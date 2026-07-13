(function () {
  // Config
  var scriptTag = document.currentScript;
  var token = scriptTag.getAttribute('data-token') || 'default';
  var widgetKey = scriptTag.getAttribute('data-widget-key');
  var host = new URL(scriptTag.src).origin;

  if (!widgetKey && token === 'default') {
    console.error('Xagent Widget: Missing data-widget-key attribute. Re-copy the embed snippet from the agent widget settings.');
    return;
  }

  // Visual Configurations
  var buttonSize = scriptTag.getAttribute('data-button-size') || '60px';
  var buttonColor = scriptTag.getAttribute('data-button-color') || '#000';
  var iconColor = scriptTag.getAttribute('data-icon-color') || '#fff';
  var panelBgColor = scriptTag.getAttribute('data-panel-bg-color') || '#fff';

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
      position: absolute;
      bottom: calc(${buttonSize} + 20px);
      right: 0;
      width: 380px;
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

    @media (max-width: 480px) {
      .xagent-widget-panel {
        width: calc(100vw - 40px);
        height: calc(100vh - 120px);
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

  // Identify the guest. If the embedding page knows who the visitor is
  // (e.g. it renders the snippet server-side for a logged-in user), it can
  // pass that identity via data-end-user-id so the widget session is scoped
  // to that specific user instead of an anonymous per-browser id. This value
  // is supplied by the embedding page on every load, so it is not cached.
  //
  // An identity claim is only trustworthy if it's signed: without
  // data-end-user-signature, anyone viewing this page's source (or calling
  // /api/widget/auth directly) could claim to be any other user and read
  // their conversation history. The signature must be computed server-side
  // by the embedding site using the agent's widget_end_user_secret (from
  // App Widget settings) -- never compute it in browser JS, which would
  // expose the secret to the same visitors it's meant to authenticate.
  var endUserId = (scriptTag.getAttribute('data-end-user-id') || '').trim();
  var endUserSignature = (scriptTag.getAttribute('data-end-user-signature') || '').trim();
  var guestId = null;
  if (endUserId && endUserId.length > 256) {
    // Silently truncating would still send the truncated id, but the
    // embedding server signed the FULL id -- the signature would no longer
    // match it, surfacing as a confusing "Invalid end-user signature" error
    // instead of an actionable one here.
    console.error('Xagent Widget: data-end-user-id exceeds 256 characters; ignoring it and falling back to an anonymous session.');
    endUserId = '';
  }
  if (endUserId && !endUserSignature) {
    console.error('Xagent Widget: data-end-user-id was provided without data-end-user-signature; ignoring it and falling back to an anonymous session. Sign end_user_id server-side with the agent\'s end-user secret (see App Widget settings).');
    endUserId = '';
  }
  if (!endUserId) {
    // No verified end-user identity: fall back to an anonymous per-browser
    // id, generated once and cached so the same browser resumes the same
    // conversation across page loads.
    guestId = localStorage.getItem('xagent_guest_id');
    if (!guestId) {
      guestId = 'guest_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
      localStorage.setItem('xagent_guest_id', guestId);
    }
  }

  // Iframe
  var iframe = document.createElement('iframe');
  iframe.className = 'xagent-widget-iframe';
  panel.appendChild(iframe);

  function loadIframe(ticket, agentId) {
    // The widget key is deliberately NOT placed in the iframe URL: the ticket
    // is sufficient to authenticate, and keeping the key out of the frame
    // means the embedded widget has no credential to fall back on.
    var url = host + '/widget/chat/' + token;
    if (endUserId) {
      url += '?end_user_id=' + encodeURIComponent(endUserId)
        + '&end_user_signature=' + encodeURIComponent(endUserSignature);
    } else {
      url += '?guest_id=' + encodeURIComponent(guestId);
    }
    if (agentId) {
      url += '&agent_id=' + encodeURIComponent(agentId);
    }
    if (ticket) {
      url += '&embed_ticket=' + encodeURIComponent(ticket);
    }
    iframe.src = url;
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
          // direct-visit path. Surface an actionable error instead.
          console.error('Xagent Widget: embed authorization failed (HTTP ' + res.status + '). Check that this page is in the agent\'s allowed domains and that the embed snippet is current.');
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

  // FAB
  var fab = document.createElement('button');
  fab.className = 'xagent-widget-fab';
  // Chat icon SVG
  var chatIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
  // Close icon SVG
  var closeIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';

  fab.innerHTML = chatIcon;

  var isOpen = false;
  fab.onclick = function () {
    isOpen = !isOpen;
    if (isOpen) {
      panel.classList.add('open');
      fab.innerHTML = closeIcon;
    } else {
      panel.classList.remove('open');
      fab.innerHTML = chatIcon;
    }
  };

  container.appendChild(panel);
  container.appendChild(fab);
  document.body.appendChild(container);
})();
