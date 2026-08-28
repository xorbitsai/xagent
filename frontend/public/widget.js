(function () {
  // Config
  var scriptTag = document.currentScript;
  var host = new URL(scriptTag.src).origin;

  var SESSION_TIMEOUT_MS = 5000;          // constants appendix #16
  var SESSION_RETRY_POLICY = {
    maxAttempts: 4,
    maxRateLimitRetries: 2,
    retryDelays: [1000, 2000, 4000],
    deadlineMs: 30000
  };
  var SESSION_RETRY_AFTER_FALLBACK_MS = 1000;
  var SESSION_REFRESH_THRESHOLD_MS = 60000;
  var SESSION_ERROR_CODES = {
    grant_malformed: true,
    signature_invalid: true,
    grant_expired: true,
    grant_already_used: true,
    agent_not_granted: true,
    agent_not_available: true,
    widget_disabled: true,
    invalid_input: true,
    invalid_runtime_context: true,
    encryption_required: true,
    reconnect_invalid: true,
    session_expired: true,
    identity_mismatch: true,
    rate_limited: true
  };
  // Only diagnostic reasons registered for their session error code may be logged.
  var SESSION_ERROR_REASONS = {
    agent_not_granted: {
      origin_not_allowed: true
    }
  };
  function sessionAbortError() {
    return new DOMException('session request superseded', 'AbortError');
  }

  function sessionDelay(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal.aborted) {
        reject(sessionAbortError());
        return;
      }
      var timer = setTimeout(function () {
        signal.removeEventListener('abort', onAbort);
        resolve();
      }, ms);
      function onAbort() {
        clearTimeout(timer);
        signal.removeEventListener('abort', onAbort);
        reject(sessionAbortError());
      }
      signal.addEventListener('abort', onAbort, { once: true });
    });
  }

  function isPlainSessionErrorObject(value) {
    if (!value || typeof value !== 'object') return false;
    // Prototype identity is not affected by host-page Symbol.toStringTag pollution.
    var prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype;
  }

  function ownSessionErrorString(value, key) {
    if (!Object.prototype.hasOwnProperty.call(value, key)) return null;
    var candidate = value[key];
    return typeof candidate === 'string' ? candidate : null;
  }

  function sessionErrorEnvelope(result) {
    if (!isPlainSessionErrorObject(result.data) ||
        !Object.prototype.hasOwnProperty.call(result.data, 'error') ||
        !isPlainSessionErrorObject(result.data.error)) {
      return { code: null, reason: null };
    }
    var error = result.data.error;
    var code = ownSessionErrorString(error, 'code');
    var reason = ownSessionErrorString(error, 'reason');
    var allowedReasons = code && Object.prototype.hasOwnProperty.call(SESSION_ERROR_REASONS, code)
      ? SESSION_ERROR_REASONS[code]
      : null;
    return {
      code: code,
      reason: allowedReasons && reason && Object.prototype.hasOwnProperty.call(allowedReasons, reason)
        ? reason
        : null
    };
  }

  function isKnownSessionErrorCode(code) {
    return Boolean(code) &&
      Object.prototype.hasOwnProperty.call(SESSION_ERROR_CODES, code);
  }

  function retryAfterMs(result) {
    var header = result && result.retryAfter;
    var value = header && header.trim();
    if (value && /^-?\d+$/.test(value)) {
      var seconds = Number(value);
      return seconds >= 0
        ? seconds * 1000
        : SESSION_RETRY_AFTER_FALLBACK_MS;
    }
    var retryAt = value ? Date.parse(value) : NaN;
    return isNaN(retryAt)
      ? SESSION_RETRY_AFTER_FALLBACK_MS
      : Math.max(0, retryAt - Date.now());
  }

  function createRetryLineage() {
    return {
      startedAt: Date.now(),
      attempts: 0,
      transportRetries: 0,
      rateLimited: 0,
      lastRetryCode: 'network_unavailable'
    };
  }

  function exhaustedRetryResult(code) {
    return {
      ok: false,
      status: 0,
      data: null,
      syntheticCode: code || 'network_unavailable'
    };
  }

  // Every request counts against the supplied total-attempt and absolute-deadline
  // lineage. Exchange owns its lineage at controller scope so bfcache can
  // replace a network operation without minting a new grant budget.
  function withRetry(makeRequest, policy, signal, onRetry, lineage) {
    var retry = lineage || createRetryLineage();

    function remainingMs() {
      return Math.max(0, policy.deadlineMs - (Date.now() - retry.startedAt));
    }

    function giveUp(result, error, code) {
      if (error) throw error;
      return result || exhaustedRetryResult(code || retry.lastRetryCode);
    }

    function retryAfter(wait, result, error, code) {
      retry.lastRetryCode = code;
      if (retry.attempts >= policy.maxAttempts || wait >= remainingMs()) {
        return giveUp(result, error, code);
      }
      if (onRetry) onRetry(code);
      return sessionDelay(wait, signal).then(run);
    }

    function retryTransportOrGiveUp(result, error) {
      if (retry.transportRetries >= policy.retryDelays.length) {
        return giveUp(result, error, 'network_unavailable');
      }
      var wait = policy.retryDelays[retry.transportRetries];
      retry.transportRetries += 1;
      return retryAfter(wait, result, error, 'network_unavailable');
    }

    function run() {
      if (signal.aborted) return Promise.reject(sessionAbortError());
      if (retry.attempts >= policy.maxAttempts) {
        return Promise.resolve(exhaustedRetryResult(retry.lastRetryCode));
      }
      var requestTimeoutMs = Math.min(SESSION_TIMEOUT_MS, remainingMs());
      if (requestTimeoutMs <= 0) {
        return Promise.resolve(exhaustedRetryResult(retry.lastRetryCode));
      }
      retry.attempts += 1;
      return makeRequest(requestTimeoutMs).then(function (result) {
        if (signal.aborted) throw sessionAbortError();
        var code = sessionErrorEnvelope(result).code;
        if (code === 'rate_limited') {
          if (retry.rateLimited >= policy.maxRateLimitRetries) return result;
          retry.rateLimited += 1;
          return retryAfter(retryAfterMs(result), result, null, 'rate_limited');
        }
        if (!isKnownSessionErrorCode(code) && result.status >= 500) {
          return retryTransportOrGiveUp(result, null);
        }
        return result;
      }, function (error) {
        if (signal.aborted) throw error;
        return retryTransportOrGiveUp(null, error);
      });
    }

    return run();
  }

  // Single mode branch point: a delegated context grant switches the whole
  // identity channel. Either factory returns null after reporting a
  // fail-closed integration error, in which case nothing is rendered at all.
  var mode = scriptTag.hasAttribute('data-encrypted-context')
    ? createSessionMode(scriptTag, host)
    : createGuestMode(scriptTag, host);
  if (!mode) {
    return;
  }

  // Visual Configurations
  var buttonSize = scriptTag.getAttribute('data-button-size') || '60px';
  var buttonColor = scriptTag.getAttribute('data-button-color') || '#000';
  var iconColor = scriptTag.getAttribute('data-icon-color') || '#fff';
  var panelBgColor = scriptTag.getAttribute('data-panel-bg-color') || '#fff';

  // Read by use-websocket.ts off the iframe URL, where it outranks the browser
  // zone. Optional: absent means the iframe URL is unchanged.
  var embedTimezone = (scriptTag.getAttribute('data-timezone') || '').trim();

  function withTimezone(url) {
    if (!embedTimezone) return url;
    var encoded;
    try {
      // A lone UTF-16 surrogate survives trim() and makes encodeURIComponent
      // throw; the override must degrade to the browser/UTC path, never abort
      // iframe setup.
      encoded = encodeURIComponent(embedTimezone);
    } catch {
      return url;
    }
    return url + (url.indexOf('?') === -1 ? '?' : '&') + 'timezone=' + encoded;
  }

  // Panel width: resizable via the left-edge drag handle, persisted across
  // reloads. Inline width is only ever applied above MOBILE_BREAKPOINT --
  // below it the CSS media query owns sizing, and inline width would win
  // over that media query by specificity if left in place.
  var WIDTH_STORAGE_KEY = 'xagent_widget_width';
  var DEFAULT_PANEL_WIDTH = 380;
  var MIN_PANEL_WIDTH = 320;
  var MAX_PANEL_WIDTH = 720;
  var MOBILE_BREAKPOINT = 480;
  var HORIZONTAL_VIEWPORT_MARGIN = 40;

  function isMobileViewport() {
    return window.innerWidth <= MOBILE_BREAKPOINT;
  }

  function clampPanelWidth(width) {
    var viewportMax = window.innerWidth - HORIZONTAL_VIEWPORT_MARGIN;
    return Math.min(Math.max(width, MIN_PANEL_WIDTH), Math.min(MAX_PANEL_WIDTH, viewportMax));
  }

  function readStoredWidth() {
    var raw;
    try {
      raw = localStorage.getItem(WIDTH_STORAGE_KEY);
    } catch (e) {
      return DEFAULT_PANEL_WIDTH;
    }
    var parsed = raw ? parseInt(raw, 10) : NaN;
    // Bounds-check against the static MIN/MAX only (never the viewport --
    // that's applyPanelWidth's job) so a corrupted or stale-schema stored
    // value self-heals on the next load instead of being echoed back forever.
    if (isNaN(parsed) || parsed < MIN_PANEL_WIDTH || parsed > MAX_PANEL_WIDTH) {
      return DEFAULT_PANEL_WIDTH;
    }
    return parsed;
  }

  function persistWidth(width) {
    try {
      localStorage.setItem(WIDTH_STORAGE_KEY, String(width));
    } catch (e) {
      // Storage can be unavailable (private browsing, quota); resizing still
      // works for the session, it just won't survive a reload.
    }
  }

  // Raw preferred width, deliberately NOT clamped here: applyPanelWidth below
  // clamps only for rendering, without writing the clamped value back here,
  // so a viewport narrower than the stored preference at load never
  // overwrites it (e.g. via a later, otherwise-unmoved drag release).
  var panelWidth = readStoredWidth();

  // Styles
  var style = document.createElement('style');
  style.innerHTML = `
    .xagent-widget-container {
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 999999;
      font-family: system-ui, -apple-system, sans-serif;
    }

    .xagent-widget-fab {
      width: ${buttonSize};
      height: ${buttonSize};
      border-radius: 50%;
      background-color: ${buttonColor};
      color: ${iconColor};
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transition: transform 0.2s ease, opacity 0.2s ease;
      border: none;
      outline: none;
      padding: 0;
    }

    .xagent-widget-fab:hover {
      transform: scale(1.05);
      opacity: 0.9;
    }

    .xagent-widget-fab svg {
      width: calc(${buttonSize} * 0.53);
      height: calc(${buttonSize} * 0.53);
      fill: currentColor;
    }

    .xagent-widget-panel {
      /* Pin the box model so the resize drag's width math (which reads
         computed CSS width) can't drift from the host page's own
         box-sizing default. */
      box-sizing: border-box;
      position: absolute;
      bottom: calc(${buttonSize} + 20px);
      right: 0;
      width: ${DEFAULT_PANEL_WIDTH}px;
      height: 600px;
      max-height: calc(100vh - 100px);
      background: ${panelBgColor};
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
      overflow: hidden;
      opacity: 0;
      visibility: hidden;
      transform: translateY(20px);
      transition: opacity 0.3s ease, transform 0.3s ease, visibility 0.3s;
      border: 1px solid rgba(0,0,0,0.1);
    }

    .xagent-widget-panel.open {
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }

    .xagent-widget-iframe {
      width: 100%;
      height: 100%;
      border: none;
      background: transparent;
    }

    .xagent-widget-resize-handle {
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
      width: 6px;
      cursor: ew-resize;
      z-index: 2;
      touch-action: none;
    }

    .xagent-widget-resize-handle:hover,
    .xagent-widget-resize-handle:active {
      background: rgba(0, 0, 0, 0.15);
    }

    /* visibility transitions to a closed panel keep it hit-testable for the
       duration of the transition (per spec, as long as either end of the
       transition is 'visible') -- without this, the handle stays draggable
       and cursor-hinting during that window even though the panel is
       already fading out. */
    .xagent-widget-panel:not(.open) .xagent-widget-resize-handle {
      display: none;
    }

    @media (max-width: ${MOBILE_BREAKPOINT}px) {
      .xagent-widget-panel {
        position: fixed;
        inset: 0;
        width: 100%;
        height: 100%;
        max-height: 100%;
        border: none;
        border-radius: 0;
        box-shadow: none;
        /* Full-screen means edge-to-edge, including under a notch/home
           indicator or, in landscape, a side notch -- the fallback keeps
           unsupporting browsers at 0. */
        padding-top: env(safe-area-inset-top, 0px);
        padding-bottom: env(safe-area-inset-bottom, 0px);
        padding-left: env(safe-area-inset-left, 0px);
        padding-right: env(safe-area-inset-right, 0px);
      }

      .xagent-widget-resize-handle {
        display: none;
      }

      /* Hide the FAB only once the iframe has confirmed (via
         widget_chrome_ready in onChromeMessage) that its own in-header
         close control is actually mounted -- loading, auth-failure, and
         degraded/terminal Session states render no header at all, and
         without this guard the FAB would disappear as soon as the panel
         opens regardless, leaving a full-screen overlay with no dismiss
         action at all in those states. */
      .xagent-widget-panel.open.xagent-chrome-ready ~ .xagent-widget-fab {
        display: none;
      }
    }
  `;
  document.head.appendChild(style);

  // Container
  var container = document.createElement('div');
  container.className = 'xagent-widget-container';

  // Panel
  var panel = document.createElement('div');
  panel.className = 'xagent-widget-panel';

  // Iframe
  var iframe = document.createElement('iframe');
  iframe.className = 'xagent-widget-iframe';
  panel.appendChild(iframe);

  // Left-edge resize handle
  var resizeHandle = document.createElement('div');
  resizeHandle.className = 'xagent-widget-resize-handle';
  panel.appendChild(resizeHandle);

  // Clamp on render rather than mutating panelWidth here: a temporary narrow
  // viewport (a shrunk window, a rotated device) must not permanently shrink
  // the user's stored preference once the viewport widens again.
  function applyPanelWidth() {
    panel.style.width = isMobileViewport() ? '' : clampPanelWidth(panelWidth) + 'px';
  }
  applyPanelWidth();

  var dragState = null;
  // Set once this instance is torn down (panelRemovalObserver below) so a
  // stray pointerdown can never start a new, permanently uncleanable drag --
  // e.g. if a host SPA re-inserts this same panel node into the DOM after
  // removing it, which flips panel.isConnected back to true but does not
  // re-arm the (already-disconnected) observer or any removed listener.
  var torndown = false;

  resizeHandle.addEventListener('pointerdown', function (event) {
    // event.button is 0 for touch/pen contact and the primary mouse button;
    // a nonzero value here is a right-click, middle-click, or pen barrel
    // button, none of which should start a resize. Guard against a
    // non-numeric value (a synthetic event with no button field) the same
    // way: only a real, explicit non-primary button should block the drag.
    // Ignore a second simultaneous pointer (e.g. a touchscreen device) so it
    // can't hijack dragState mid-drag and strand the first pointer's cleanup.
    // The injected stylesheet already hides the handle while the panel is
    // closed (see .xagent-widget-panel:not(.open) above), but a host page
    // whose CSP blocks that inline <style> would leave it visually
    // draggable with no CSS backing it up -- check the open class directly
    // too, as defense in depth that doesn't depend on the stylesheet loading.
    if (torndown || isMobileViewport() || dragState || (event.button || 0) !== 0 || !panel.classList.contains('open')) return;
    dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      // getBoundingClientRect().width is always border-box regardless of
      // box-sizing, but panel.style.width sets content-box width unless
      // box-sizing: border-box is in effect -- computed style avoids that
      // mismatch (a jump by the border width on every drag start) even if
      // something on the host page ends up overriding our own box-sizing rule.
      startWidth: parseInt(window.getComputedStyle(panel).width, 10) || DEFAULT_PANEL_WIDTH,
      // The drag's live candidate. panelWidth itself is never written until
      // a real release commits it (see finishDrag) -- cancelling a drag
      // (blur, a mid-drag viewport change, pointercancel, DOM removal) then
      // needs no rollback of anything, since the committed preference was
      // never touched in the first place.
      lastWidth: null,
      // Whether this drag ever visibly rendered a width other than
      // startWidth -- a release with this still false is a no-op (a plain
      // click, or a drag fully absorbed by the viewport ceiling) and must
      // leave the existing preference alone rather than re-commit whatever
      // startWidth happened to be.
      moved: false,
      originalCursor: document.body.style.cursor,
      originalUserSelect: document.body.style.userSelect
    };
    // Optimization, not the drag's only path: pointermove/pointerup/
    // pointercancel are bound to `document` below specifically so the drag
    // still works if this capture silently fails.
    try {
      resizeHandle.setPointerCapture(event.pointerId);
    } catch (e) {}
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    // Without this, the iframe (a sibling of the handle, not an ancestor)
    // would intercept pointer events once the cursor moves over it mid-drag;
    // disabling its pointer-events lets those events fall through to panel
    // and bubble to the document-level listeners below instead of stalling
    // on the iframe.
    iframe.style.pointerEvents = 'none';
    event.preventDefault();
  });

  function onPointerMove(event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    // Panel is right-anchored, so dragging the left edge left (clientX
    // decreasing) should grow the panel.
    var delta = dragState.startX - event.clientX;
    var candidate = clampPanelWidth(dragState.startWidth + delta);
    panel.style.width = candidate + 'px';
    dragState.lastWidth = candidate;
    // A drag that started already pinned to the viewport ceiling (the raw
    // preference is wider than what currently fits) and never visibly moves
    // the render away from that ceiling hasn't expressed a new choice --
    // leave the existing preference alone rather than re-committing the
    // ceiling value on release. Always tracking the *latest* candidate here
    // (not a running min/max across the drag) is what makes an overshoot
    // that gets corrected back commit the corrected value, not the peak.
    if (candidate !== dragState.startWidth) dragState.moved = true;
  }

  function restoreDragSideEffects() {
    // Only restore if nothing else changed it in the meantime -- a host page
    // that set its own userSelect/cursor after this drag started (e.g. while
    // rebuilding its own UI mid-drag) must not have that value clobbered by
    // a stale snapshot from before it ran.
    if (document.body.style.userSelect === 'none') {
      document.body.style.userSelect = dragState.originalUserSelect;
    }
    if (document.body.style.cursor === 'ew-resize') {
      document.body.style.cursor = dragState.originalCursor;
    }
    iframe.style.pointerEvents = '';
    // The browser normally releases capture implicitly on pointerup/
    // pointercancel, but not on blur or a plain resize event -- release
    // explicitly so a drag ended that way can't keep this pointer's future
    // events (and thus other elements on the host page) captured to the
    // handle until an eventual real pointerup arrives.
    try {
      resizeHandle.releasePointerCapture(dragState.pointerId);
    } catch (e) {}
  }

  // A real release (pointerup/pointercancel... see below) commits whatever
  // was last rendered; any other way a drag ends only cancels. Both share
  // this one path so the guard, cleanup, and re-render can't drift apart
  // between the two the way they did across earlier rounds.
  function finishDrag(commit, event) {
    if (!dragState || (event && event.pointerId !== dragState.pointerId)) return;
    restoreDragSideEffects();
    // dragState.moved false means this release never actually rendered a
    // different width (a plain click, or a drag fully absorbed by the
    // viewport ceiling) -- leave the existing preference untouched rather
    // than re-committing startWidth as if it were a real choice. Skip
    // committing entirely once the panel has left the DOM: any width here
    // reflects a stale, no-longer-visible instance rather than a choice the
    // user can still see or reconsider.
    if (commit && dragState.moved && panel.isConnected) {
      panelWidth = dragState.lastWidth;
      persistWidth(panelWidth);
    }
    dragState = null;
    applyPanelWidth();
  }
  function endDrag(event) {
    finishDrag(true, event);
  }
  // Cancels rather than commits: window blur, a viewport change invalidating
  // this drag's frozen geometry, DOM removal, and pointercancel (an
  // involuntary interruption -- a touch gesture reinterpreted as a scroll,
  // an OS-level interrupt, capture lost to another element -- rather than
  // the user confirming a release) are none of them the user releasing the
  // handle, so none of them persist a width.
  function cancelDrag(event) {
    finishDrag(false, event);
  }
  document.addEventListener('pointermove', onPointerMove);
  document.addEventListener('pointerup', endDrag);
  document.addEventListener('pointercancel', cancelDrag);

  function onWindowBlur() {
    // Focus moving into this widget's own iframe (e.g. the chat composer
    // autofocusing) also fires a parent-window blur; the window itself
    // hasn't actually lost focus, so this isn't the interruption below
    // exists for.
    if (document.hasFocus()) return;
    // A drag released off-window (e.g. Alt+Tab away mid-drag) never delivers
    // pointerup/pointercancel, which would otherwise strand userSelect:
    // 'none' on the host page indefinitely. A no-op when no drag is active.
    cancelDrag();
  }
  window.addEventListener('blur', onWindowBlur);

  var lastViewportWidth = window.innerWidth;
  function onWindowResize() {
    var currentViewportWidth = window.innerWidth;
    var widthChanged = currentViewportWidth !== lastViewportWidth;
    lastViewportWidth = currentViewportWidth;
    // Only a horizontal change affects anything here -- isMobileViewport()/
    // clampPanelWidth() and a drag's frozen startX/startWidth anchors all
    // depend solely on innerWidth. A resize event with an unchanged
    // innerWidth (e.g. a mobile on-screen keyboard opening, which only moves
    // innerHeight) must be a complete no-op: applyPanelWidth() would just
    // re-render the same value when idle, but during an active drag it would
    // clobber the live preview (it always renders from the last *committed*
    // width, not dragState's in-progress one) without actually cancelling
    // the drag -- worse than doing nothing.
    if (!widthChanged) return;
    applyPanelWidth();
    // A genuine width change mid-drag invalidates this drag's frozen startX/
    // startWidth anchors -- rather than let a later pointermove misread the
    // viewport's own movement as the user shrinking the panel (and persist
    // that on release), end the drag cleanly here. The user can just grab
    // the handle again if they still want to resize.
    cancelDrag();
  }
  window.addEventListener('resize', onWindowResize);

  // Teardown the instant the panel leaves the DOM (a microtask later, per
  // MutationObserver) -- covers both leak prevention (the listeners above
  // are otherwise GC roots keeping this whole closure, panel included, alive
  // indefinitely) and the drag-interrupted-by-removal case (blur alone
  // doesn't fire just because a host SPA removes the widget, so waiting for
  // it would leave userSelect stranded on the host page until an unrelated
  // future blur happens to occur, if ever). Matches the isConnected-driven
  // detection this file already uses for the session controller's own
  // listeners.
  var panelRemovalObserver = new MutationObserver(function () {
    if (panel.isConnected) return;
    torndown = true;
    panelRemovalObserver.disconnect();
    window.removeEventListener('resize', onWindowResize);
    window.removeEventListener('blur', onWindowBlur);
    window.removeEventListener('message', onChromeMessage);
    document.removeEventListener('pointermove', onPointerMove);
    document.removeEventListener('pointerup', endDrag);
    document.removeEventListener('pointercancel', cancelDrag);
    cancelDrag();
    // Belt-and-suspenders for a host SPA that re-inserts this same node
    // later (torndown permanently blocks a new drag regardless, but without
    // this the handle would still show its ew-resize cursor and hover
    // highlight, looking interactive while doing nothing).
    resizeHandle.style.display = 'none';
  });

  // FAB
  var fab = document.createElement('button');
  fab.className = 'xagent-widget-fab';
  // Chat icon SVG
  var chatIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
  // Close icon SVG
  var closeIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';

  fab.innerHTML = chatIcon;

  var isOpen = false;

  function openPanel() {
    isOpen = true;
    panel.classList.add('open');
    fab.innerHTML = closeIcon;
  }

  function closePanel() {
    isOpen = false;
    panel.classList.remove('open');
    fab.innerHTML = chatIcon;
  }

  fab.onclick = function () {
    if (isOpen) {
      closePanel();
    } else {
      openPanel();
    }
  };

  container.appendChild(panel);
  container.appendChild(fab);
  document.body.appendChild(container);

  // Armed only after container is actually in the tree: subtree: true is
  // needed because "direct child of body" alone only catches container's
  // own removal -- a host page reaching further in (removing panel from
  // container directly, or wiping a wrapper it added around container)
  // would otherwise mutate a node this observer never inspects, and the
  // callback would simply never fire.
  // documentElement, not body: a host framework that replaces <body> wholesale
  // on navigation (e.g. Turbo Drive, htmx boosted navigation) never mutates
  // the childList of the *old* body node that still holds container -- only
  // <html>'s own childList changes when body itself is swapped out.
  panelRemovalObserver.observe(document.documentElement, { childList: true, subtree: true });

  // The header's close control lives inside the iframe's React app and has
  // no direct handle to panel/fab, so it signals intent back here over
  // postMessage instead. Named (not inline) so panelRemovalObserver can
  // remove it below, matching every other listener's teardown in this file
  // -- otherwise a host SPA that removes the widget without a full page
  // reload leaves this listener retaining the closure indefinitely.
  function onChromeMessage(event) {
    if (torndown || event.origin !== host || !iframe.contentWindow || event.source !== iframe.contentWindow) return;
    var data = event.data;
    if (!data || data.xagent !== true || data.v !== 1) return;
    if (data.type === 'widget_close') {
      closePanel();
    } else if (data.type === 'widget_chrome_ready') {
      // The child's own close control only exists once its "active" header
      // has actually mounted -- loading/error/expired states render none.
      // Until this arrives (or after it's revoked below), the FAB stays the
      // parent-owned fallback so a stuck child state can never fully trap
      // the visitor behind a full-screen mobile panel with no dismiss action.
      panel.classList.add('xagent-chrome-ready');
    } else if (data.type === 'widget_chrome_not_ready') {
      panel.classList.remove('xagent-chrome-ready');
    }
  }
  window.addEventListener('message', onChromeMessage);

  mode.attach(iframe);

  function createGuestMode(scriptTag, host) {
    var token = scriptTag.getAttribute('data-token') || 'default';
    var widgetKey = scriptTag.getAttribute('data-widget-key');

    if (!widgetKey && token === 'default') {
      console.error('Xagent Widget: Missing data-widget-key attribute. Re-copy the embed snippet from the agent widget settings.');
      return null;
    }

    return {
      attach: function (iframe) {
        // Generate guest_id if not exists
        var guestId = localStorage.getItem('xagent_guest_id');
        if (!guestId) {
          guestId = 'guest_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
          localStorage.setItem('xagent_guest_id', guestId);
        }

        function loadIframe(ticket, agentId) {
          // The widget key is deliberately NOT placed in the iframe URL: the ticket
          // is sufficient to authenticate, and keeping the key out of the frame
          // means the embedded widget has no credential to fall back on.
          var url = host + '/widget/chat/' + token + '?guest_id=' + guestId;
          if (agentId) {
            url += '&agent_id=' + encodeURIComponent(agentId);
          }
          if (ticket) {
            url += '&embed_ticket=' + encodeURIComponent(ticket);
          }
          iframe.src = withTimezone(url);
        }

        // Request a short-lived embed ticket from the top-level page. This fetch
        // carries the embedding page's real, browser-enforced Origin header, which
        // the backend validates against allowed_domains before signing the ticket.
        // Fetches inside the iframe carry the iframe's own origin instead, so the
        // ticket is how the validated embedding origin reaches the auth call.
        if (widgetKey) {
          fetch(host + '/api/widget/embed-ticket', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ widget_key: widgetKey })
          })
            .then(function (res) {
              if (!res.ok) {
                // Fail closed: no ticket means auth would fail, and loading the
                // iframe anyway would let a non-allowlisted embed slip through the
                // direct-visit path. Surface an actionable error instead — 429 is
                // rate limiting (transient, high traffic), distinct from the
                // domain-allowlist / stale-snippet causes behind a 403.
                if (res.status === 429) {
                  console.error('Xagent Widget: embed authorization rate-limited (HTTP 429), usually transient under high traffic. Reload the page to try again.');
                } else {
                  console.error('Xagent Widget: embed authorization failed (HTTP ' + res.status + '). Check that this page is in the agent\'s allowed domains and that the embed snippet is current.');
                }
                return null;
              }
              return res.json();
            })
            .then(function (data) {
              if (!data || !data.ticket) {
                return;
              }
              loadIframe(data.ticket, data.agent_id);
            })
            .catch(function (err) {
              console.error('Xagent Widget: embed authorization request failed (' + err + ').');
            });
        } else {
          // Deprecated data-token channel (dead server-side); loaded without a ticket.
          loadIframe(null, null);
        }
      }
    };
  }

  function xagentSessionRegistry() {
    if (!window.__xagentWidgetGrants) {
      window.__xagentWidgetGrants = {};
    }
    return window.__xagentWidgetGrants;
  }

  // 32-bit FNV-1a: a fast, dependency-free, synchronous string digest. This is
  // NOT a security boundary (no crypto guarantees) -- it only exists so the
  // dedupe registry never stores the grant plaintext itself, so that a
  // third-party script scraping window.__xagentWidgetGrants can't recover the
  // grant the way it could if the raw string were used as the key.
  function hashGrant(text) {
    var hash = 0x811c9dc5;
    for (var i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return 'g' + hash.toString(16);
  }

  function logSession(level, text, code, status, reason) {
    var suffix = status ? ' (HTTP ' + status + ')' : '';
    var diagnostic = reason ? code + '/' + reason : code;
    console[level]('Xagent Widget: ' + text + ' [' + diagnostic + ']' + suffix + '.');
  }

  function createSessionMode(scriptTag, host) {
    var grant = scriptTag.getAttribute('data-encrypted-context') || '';
    // Remove the credential before every possible exit. Conflict and duplicate
    // embeds are fail-closed too, but must not leave the raw grant in the DOM.
    scriptTag.removeAttribute('data-encrypted-context');

    // Fail closed before any network call, and render nothing at all: these are
    // integration mistakes on the embedding page, mirroring the missing
    // data-widget-key precedent. Never fall back to guest mode.
    if (!grant.trim()) {
      logSession('error', 'chat unavailable, the grant attribute is empty', 'grant_malformed');
      return null;
    }
    if (scriptTag.hasAttribute('data-widget-key') || scriptTag.hasAttribute('data-token')) {
      logSession('error', 'chat unavailable, remove the legacy widget attribute', 'attribute_conflict');
      return null;
    }

    var registry = xagentSessionRegistry();
    var registryKey = hashGrant(grant);
    if (registry[registryKey]) {
      logSession('warn', 'this grant is single-use; mint a fresh grant before remounting the widget', 'duplicate_init');
      return null;
    }
    registry[registryKey] = true;

    return createSessionController(grant, host);
  }

  function createSessionController(grant, host) {
    var state = {
      grant: grant,
      iframe: null,
      ready: false,
      session: null,
      reconnectToken: null,
      terminalCode: null,
      recoverableCode: null,
      exchangeRetry: createRetryLineage(),
      reconnectRetry: null,
      detached: false,
      observer: null,
      inflight: { exchange: null, reconnect: null },
      recoveryFlight: null,
      restorePending: false,
      frozenInflight: { exchange: null, reconnect: null },
      frozenRecoveryFlight: null,
      rapidRecoveries: 0,
      deliverySequence: 0,
      awaitedDeliveryId: null,
      invalidatedDeliveryId: null,
      stableDeliveryId: null,
      stabilityTimer: null
    };

    function attach(iframe) {
      state.iframe = iframe;
      iframe.src = withTimezone(host + '/widget/chat/session');
      window.addEventListener('message', onMessage);
      window.addEventListener('pageshow', onPageShow);
      window.addEventListener('pagehide', onPageHide);
      state.observer = new MutationObserver(onDomMutation);
      // documentElement, not body: see panelRemovalObserver's identical
      // reasoning above -- a full <body> replacement never mutates the old
      // body's own childList.
      state.observer.observe(document.documentElement, { childList: true, subtree: true });
      runLoadFlow();
    }

    function runLoadFlow() {
      return exchange();
    }

    function runRecoveryFlow(isResume) {
      if (state.terminalCode) {
        flush();
        return Promise.resolve();
      }
      // Join the existing parent owner before touching the rapid-recovery
      // circuit: iframe close/error bursts do not consume logical attempts.
      if (state.recoveryFlight) return state.recoveryFlight;
      var initialExchange = !state.session && inflightPromise('exchange');
      if (initialExchange) return initialExchange;
      if (!isResume) {
        invalidateStability();
        if (state.rapidRecoveries >= 3) {
          recordFailure('network_unavailable');
          return Promise.resolve();
        }
        state.rapidRecoveries += 1;
      }
      var recovery = isNonBlankString(state.reconnectToken) ? reconnect() : exchange();
      var recoveryFlight;
      recoveryFlight = Promise.resolve(recovery).then(function () {
        if (state.recoveryFlight === recoveryFlight) state.recoveryFlight = null;
      }, function () {
        if (state.recoveryFlight === recoveryFlight) state.recoveryFlight = null;
      });
      state.recoveryFlight = recoveryFlight;
      return recoveryFlight;
    }

    function onDomMutation() {
      if (state.iframe && !state.iframe.isConnected) detach();
    }

    function detach() {
      if (state.detached) return;
      state.detached = true;
      invalidateStability();
      cancelInflight('exchange');
      cancelInflight('reconnect');
      window.removeEventListener('message', onMessage);
      window.removeEventListener('pageshow', onPageShow);
      window.removeEventListener('pagehide', onPageHide);
      if (state.observer) state.observer.disconnect();
      state.observer = null;
      state.iframe = null;
      state.ready = false;
    }

    // Capture exactly which operations the browser freezes. A later pageshow
    // may cancel those operations, but must not cancel recovery started by an
    // earlier pageshow for the same navigation.
    function onPageHide(event) {
      if (!event.persisted || state.detached) return;
      invalidateStability();
      state.restorePending = true;
      state.frozenInflight.exchange = state.inflight.exchange;
      state.frozenInflight.reconnect = state.inflight.reconnect;
      state.frozenRecoveryFlight = state.recoveryFlight;
    }

    // Scripts do not re-run on a bfcache restore. Consume each persisted
    // pagehide once, retire only work frozen by that transition, and keep a
    // still-fresh confirmed session instead of speculatively replaying its
    // rotate-on-use reconnect credential.
    function onPageShow(event) {
      if (!event.persisted) return;
      if (state.detached) return;
      if (!state.restorePending) return;
      state.restorePending = false;
      var frozenExchange = state.frozenInflight.exchange;
      var frozenReconnect = state.frozenInflight.reconnect;
      var frozenRecoveryFlight = state.frozenRecoveryFlight;
      state.frozenInflight.exchange = null;
      state.frozenInflight.reconnect = null;
      state.frozenRecoveryFlight = null;
      if (state.terminalCode) return;
      cancelInflightIfCurrent('exchange', frozenExchange);
      cancelInflightIfCurrent('reconnect', frozenReconnect);
      if (state.recoveryFlight === frozenRecoveryFlight) state.recoveryFlight = null;
      state.recoverableCode = null;
      if (state.session && !isStale(state.session.session_token_expires_at)) {
        // A persisted restore reuses the token lineage but starts a new
        // parent/iframe delivery epoch; the pagehide-invalidated ID cannot
        // establish post-restore stability.
        state.session.session_delivery_id = mintDeliveryId();
        flush();
        return;
      }
      runRecoveryFlow(Boolean(frozenRecoveryFlight));
    }

    function onMessage(event) {
      if (state.detached || !state.iframe || event.source !== state.iframe.contentWindow) return;
      if (event.origin !== host) return;
      var data = event.data;
      if (!data || data.xagent !== true || data.v !== 1) return;

      if (data.type === 'ready') {
        state.ready = true;
        flush();
      } else if (data.type === 'reconnect_request') {
        runRecoveryFlow();
      } else if (data.type === 'session_connection_open') {
        if (!isNonBlankString(data.session_delivery_id)) return;
        beginStability(data.session_delivery_id);
      }
    }

    function send(message) {
      var iframe = state.iframe;
      if (state.detached || !iframe || !iframe.isConnected || !iframe.contentWindow) {
        detach();
        return;
      }
      message.xagent = true;
      message.v = 1;
      iframe.contentWindow.postMessage(message, host);
    }

    function isStale(expiresAt) {
      var at = Date.parse(expiresAt);
      if (isNaN(at)) return true;
      return at - Date.now() < SESSION_REFRESH_THRESHOLD_MS;
    }

    // Level-triggered: every ready re-sends whatever the current state is, and
    // only the latest state is ever sent. Replaying an older session_update
    // would hand the iframe an already-rotated token.
    function flush() {
      if (!state.ready) return;
      if (state.terminalCode) {
        send({ type: 'session_terminal', code: state.terminalCode });
        return;
      }
      if (state.recoverableCode) {
        send({ type: 'session_degraded', code: state.recoverableCode });
        return;
      }
      if (!state.session) return;
      // Rule 3: refresh before any use of the token, including handing it over.
      if (isStale(state.session.session_token_expires_at)) {
        runRecoveryFlow();
        return;
      }
      var deliveryId = state.session.session_delivery_id;
      if (deliveryId !== state.stableDeliveryId && deliveryId !== state.invalidatedDeliveryId) {
        state.awaitedDeliveryId = deliveryId;
      }
      send({
        type: 'session_update',
        session_delivery_id: deliveryId,
        session_token: state.session.session_token,
        session_token_expires_at: state.session.session_token_expires_at,
        absolute_expires_at: state.session.absolute_expires_at,
        agent: state.session.agent
      });
    }

    function isRecord(value) {
      return value !== null && typeof value === 'object' && !Array.isArray(value);
    }

    function isNonBlankString(value) {
      return typeof value === 'string' && value.trim().length > 0;
    }

    function isTimestamp(value) {
      return isNonBlankString(value) && !isNaN(Date.parse(value));
    }

    function parseSessionPayload(data) {
      if (!isRecord(data) ||
          !isNonBlankString(data.session_token) ||
          !isTimestamp(data.session_token_expires_at) ||
          !isNonBlankString(data.reconnect_token) ||
          !isRecord(data.session) ||
          !isTimestamp(data.session.absolute_expires_at) ||
          !isRecord(data.session.agent)) {
        return null;
      }
      return {
        session: {
          session_token: data.session_token,
          session_token_expires_at: data.session_token_expires_at,
          absolute_expires_at: data.session.absolute_expires_at,
          agent: data.session.agent
        },
        reconnectToken: data.reconnect_token
      };
    }

    function applySession(payload) {
      var reconnectTokenChanged = payload.reconnectToken !== state.reconnectToken;
      state.session = payload.session;
      state.session.session_delivery_id = mintDeliveryId();
      state.awaitedDeliveryId = null;
      state.reconnectToken = payload.reconnectToken;
      if (reconnectTokenChanged) state.reconnectRetry = null;
      state.recoverableCode = null;
    }

    function mintDeliveryId() {
      state.deliverySequence += 1;
      return 'widget-delivery-' + Date.now().toString(36) + '-' + state.deliverySequence;
    }

    function invalidateStability() {
      if (state.stabilityTimer) {
        clearTimeout(state.stabilityTimer);
        state.stabilityTimer = null;
      }
      if (state.awaitedDeliveryId) {
        state.invalidatedDeliveryId = state.awaitedDeliveryId;
        state.awaitedDeliveryId = null;
      }
    }

    function beginStability(deliveryId) {
      if (deliveryId !== state.awaitedDeliveryId || deliveryId === state.invalidatedDeliveryId) return;
      if (state.stabilityTimer || state.stableDeliveryId === deliveryId) return;
      var timer = setTimeout(function () {
        if (state.stabilityTimer !== timer || state.awaitedDeliveryId !== deliveryId) return;
        state.stabilityTimer = null;
        state.awaitedDeliveryId = null;
        state.stableDeliveryId = deliveryId;
        state.rapidRecoveries = 0;
      }, 15000);
      state.stabilityTimer = timer;
    }

    function reconnectRetryLineage(reconnectToken) {
      var current = state.reconnectRetry;
      if (!current || current.reconnectToken !== reconnectToken) {
        current = {
          reconnectToken: reconnectToken,
          lineage: createRetryLineage()
        };
        state.reconnectRetry = current;
      }
      return current.lineage;
    }

    function recordRetry(code) {
      if (state.terminalCode) return;
      state.recoverableCode = code;
      logSession('warn', 'session recovery is retrying', code);
      flush();
    }

    function recordFailure(code, status, reason) {
      if (state.terminalCode) return;
      state.terminalCode = code;
      state.recoverableCode = null;
      logSession('error', 'chat unavailable', code, status, reason);
      flush();
    }

    function classifySessionFailure(result) {
      var syntheticCode = ownSessionErrorString(result, 'syntheticCode');
      if (syntheticCode) return { code: syntheticCode, reason: null };
      var envelope = sessionErrorEnvelope(result);
      if (isKnownSessionErrorCode(envelope.code)) return envelope;
      if (result.status >= 500) return { code: 'network_unavailable', reason: null };
      return { code: 'unexpected_error', reason: null };
    }

    function handleResult(result, phase) {
      var payload = result.ok ? parseSessionPayload(result.data) : null;
      var isSuccess = payload !== null;
      if (isSuccess) {
        if (phase === 'reconnect' && isStale(payload.session.session_token_expires_at)) {
          recordFailure('unexpected_error', result.status);
          return;
        }
        applySession(payload);
        flush();
        return;
      }

      var failure = classifySessionFailure(result);
      recordFailure(failure.code, result.status, failure.reason);
    }

    function postJson(url, body, timeoutMs, operationSignal) {
      if (operationSignal.aborted) return Promise.reject(sessionAbortError());
      var controller = new AbortController();
      var timer;
      var timeout = new Promise(function (_resolve, reject) {
        timer = setTimeout(function () {
          controller.abort();
          reject(new DOMException('session request deadline exceeded', 'AbortError'));
        }, timeoutMs);
      });
      var onOperationAbort;
      var superseded = new Promise(function (_resolve, reject) {
        onOperationAbort = function () {
          controller.abort();
          reject(sessionAbortError());
        };
        operationSignal.addEventListener('abort', onOperationAbort, { once: true });
      });

      function cleanup() {
        clearTimeout(timer);
        operationSignal.removeEventListener('abort', onOperationAbort);
      }

      var request;
      try {
        request = fetch(url, {
          method: 'POST',
          credentials: 'omit',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal
        });
      } catch (error) {
        cleanup();
        return Promise.reject(error);
      }
      request = request.then(function (res) {
        return res.json().catch(function (error) {
          // A complete non-JSON response is a protocol error handled from its
          // HTTP status. Body-stream failures and aborts remain transport
          // failures so withRetry can apply the bounded retry policy.
          if (error && error.name === 'SyntaxError') return null;
          throw error;
        }).then(function (data) {
          return {
            ok: res.ok,
            status: res.status,
            data: data,
            retryAfter: res.headers.get('Retry-After')
          };
        });
      });
      return Promise.race([request, timeout, superseded]).then(function (result) {
        cleanup();
        return result;
      }, function (err) {
        cleanup();
        throw err;
      });
    }

    function singleFlight(key, makePromise) {
      var current = state.inflight[key];
      if (current) return current.promise;

      var operation = { controller: new AbortController(), promise: null };
      state.inflight[key] = operation;
      var result;
      try {
        result = makePromise(operation.controller.signal);
      } catch (error) {
        result = Promise.reject(error);
      }
      operation.promise = Promise.resolve(result).then(function () {
        if (state.inflight[key] === operation) state.inflight[key] = null;
      }, function () {
        if (state.inflight[key] === operation) state.inflight[key] = null;
      });
      return operation.promise;
    }

    function inflightPromise(key) {
      var operation = state.inflight[key];
      return operation && operation.promise;
    }

    function cancelInflight(key) {
      var operation = state.inflight[key];
      if (!operation) return;
      state.inflight[key] = null;
      operation.controller.abort();
    }

    function cancelInflightIfCurrent(key, operation) {
      if (!operation || state.inflight[key] !== operation) return;
      cancelInflight(key);
    }

    // Exchange and reconnect differ only in endpoint/body/lineage; both keep
    // retry publication fenced by the operation signal.
    function requestSessionWithRetry(url, body, retryLineage, signal) {
      return withRetry(function (timeoutMs) {
        return postJson(url, body, timeoutMs, signal);
      }, SESSION_RETRY_POLICY, signal, function (code) {
        if (!signal.aborted) recordRetry(code);
      }, retryLineage);
    }

    function exchange() {
      return singleFlight('exchange', function (signal) {
        return requestSessionWithRetry(
          host + '/v1/external/chat/sessions',
          { encrypted_context: state.grant },
          state.exchangeRetry,
          signal
        ).then(function (result) {
          handleResult(result, 'exchange');
        }, function () {
          if (!signal.aborted) recordFailure('network_unavailable');
        });
      });
    }

    function reconnect() {
      if (state.terminalCode) {
        flush();
        return Promise.resolve();
      }
      if (state.recoverableCode) {
        return inflightPromise('reconnect') || inflightPromise('exchange') || Promise.resolve();
      }
      var reconnectToken = state.reconnectToken;
      // Defense in depth: every internal reconnect caller must revalidate the
      // rotate-on-use credential immediately before constructing the request.
      if (!isNonBlankString(reconnectToken)) {
        recordFailure('reconnect_invalid');
        return Promise.resolve();
      }
      return singleFlight('reconnect', function (signal) {
        var body = { reconnect_token: reconnectToken };
        body.encrypted_context = state.grant;
        var retryLineage = reconnectRetryLineage(reconnectToken);
        return requestSessionWithRetry(
          host + '/v1/external/chat/sessions/reconnect',
          body,
          retryLineage,
          signal
        ).then(function (result) {
          handleResult(result, 'reconnect');
        }, function () {
          if (!signal.aborted) recordFailure('network_unavailable');
        });
      });
    }

    return { attach: attach };
  }
})();
