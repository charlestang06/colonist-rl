# Optional real Playwright transport

`colonist_rl.browser` supplies actual Playwright Chromium page control for the
user-requested web player. Importing it starts no browser and needs no Playwright
installation. Using `PlaywrightSession` explicitly opens a browser. The transport
uses public screenshot, mouse, locator, and browser lifecycle APIs; it does not
use private game APIs, bot-evasion settings, or existing browser sessions.

Install the optional dependency and Chromium in the environment that will run it:

```sh
python -m pip install -e '.[browser]'
python -m playwright install chromium
```

A headed browser with an isolated context is the default:

```python
from colonist_rl.browser import PlaywrightSession

with PlaywrightSession() as session:  # Defaults to https://colonist.io/.
    page = session.page
    transport = session.transport
    capture = session.capture("captures/colonist.png")
    print(capture.image_path, capture.metadata_path)
```

An existing synchronous Playwright `Page` can instead be injected directly:

```python
from colonist_rl.browser import PlaywrightTransport

transport = PlaywrightTransport(page, timeout_ms=10_000)
png_bytes = transport.screenshot()
visible_text = transport.read_visible_text()
# A caller that has established a valid UI target can use transport.click(x, y).
```

`PlaywrightTransport` implements the existing `BrowserTransport` protocol.
Screenshots use PNG bytes for the current viewport. Text uses only the rendered
body's `inner_text`; cookies, storage, hidden game state, and session tokens are
not inspected. `capture(output_path, metadata_path=None)` writes the screenshot
and a JSON sidecar containing its SHA-256, visible text, timestamp, and origin.
URL paths, query strings, fragments, and credentials are not copied to metadata.
The sidecar is explicitly `verified=False`; it is capture evidence, not an
invented board observation or automatic annotation.

The current page must stay connected and use `https://colonist.io` or a loopback
fixture host (`localhost`, `127.0.0.1`, or `::1`). Unknown origins, `file:` URLs,
credential-bearing URLs, disconnected pages, and invalid screenshot output are
rejected. The origin is checked before and after captures/text and after clicks;
redirected captures are rejected before files are written. This is an operation
boundary, not a general browser request firewall.

`PlaywrightSession(url=..., headed=False, timeout_ms=...)` can run a headless
fixture. Passing `user_data_dir="/explicit/profile/path"` is the only way to opt
into persistence. No profile directory is discovered or reused automatically.
Browser launch, navigation, screenshots, and visible-text reads use a finite
10-second timeout by default; the page also receives default action/navigation
timeouts. Playwright's public `Mouse.click` method has no individual timeout
argument, so this wrapper does not claim a hard wall-clock deadline for a stalled
browser connection. It does not move synchronous Playwright objects across
threads. A process-level watchdog is appropriate when a hard execution deadline
is required.

Exiting the session closes its context, browser, and Playwright driver. Failed
launch/navigation paths attempt every acquired cleanup and preserve the original
failure. `close()` is idempotent. Closing an injected transport closes its page,
while session cleanup owns the context and browser.

## Connecting the guarded adapter

The transport is real page control; it does not solve general board recognition
or autonomous play. Use `BrowserAdapter` with a reader that produces an actual
verified, stable, confident current observation, calibrated controls, and the
appropriate target mode. The adapter still requires explicit enablement and
authorization, legal-action and offer/version checks, and a verified refreshed
state after every action. Unknown screenshot states remain blocked by the
annotated-capture reader. A game should not be described as played or completed
based on opening the page or capturing an image.

```python
from colonist_rl.perception import BrowserAdapter

with PlaywrightSession(url="https://colonist.io/") as session:
    adapter = BrowserAdapter(
        transport=session.transport,
        reader=verified_reader,
        calibration=calibration,
        controls=exact_controls,
        enabled=True,
        authorized=True,
    )
    # Execute only after the reader/calibration represent this current UI.
    # refreshed = adapter.execute(current_observation, legal_action)
```

No live browser was opened by this implementation worker. Unit tests use an
injected fake Page to verify the real API call boundary, host/connection checks,
capture output, isolated/headed defaults, explicit profiles, and failure cleanup.
A separate opt-in smoke test starts a loopback HTML fixture and actual Chromium:

```sh
COLONIST_RL_BROWSER_SMOKE=1 python -m pytest -q tests/test_browser.py -k optional_local_html
```

It skips when the package or Chromium executable is unavailable. The default
test run skips this browser test, and the fixture test does not contact Colonist.
