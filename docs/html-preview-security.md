# HTML preview security and compatibility

Generated HTML is untrusted. The in-app preview uses `srcdoc` with exactly
`sandbox="allow-scripts"`, without same-origin, popup or top-navigation access.
Its document starts with a `no-referrer` meta policy for subresource requests;
the iframe also declares `referrerPolicy="no-referrer"`.
HTML preprocessing uses parse5's data tree, not a browser DOM that might load
untrusted resources before the preview policy is installed.

This deliberately gives previews an opaque origin. Browser storage APIs can
throw `SecurityError`, and ES modules, fonts and JavaScript fetches require
their resource server to accept a `null` origin. Classic scripts and images
have different browser loading rules. Not all generated web applications will
work unchanged. The jsdom tests check markup contracts, not browser sandbox
enforcement or full application compatibility.

Do not restore `allow-same-origin` or add wildcard CORS to private file APIs to
work around these restrictions. A separately deployed preview origin and full
CSP require a deployment design and are not implemented by this PR. Untrusted
scripts can still make network requests: the referrer policy is not an egress
control or a replacement for CSP.
