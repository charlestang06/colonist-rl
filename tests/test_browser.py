import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

from colonist_rl.browser import BrowserConnectionError, PlaywrightSession, PlaywrightTransport


class FakePage:
    def __init__(self, url="https://colonist.io/", *, connected=True):
        self.url = url
        self.closed = False
        self.calls = []
        self.context = SimpleNamespace(browser=SimpleNamespace(is_connected=lambda: connected))
        self.mouse = SimpleNamespace(click=lambda x, y: self.calls.append(("click", x, y)))

    def is_closed(self):
        return self.closed

    def set_default_timeout(self, timeout):
        self.calls.append(("default_timeout", timeout))

    def set_default_navigation_timeout(self, timeout):
        self.calls.append(("navigation_timeout", timeout))

    def screenshot(self, **kwargs):
        self.calls.append(("screenshot", kwargs))
        return b"\x89PNG\r\n\x1a\nfixture"

    def locator(self, selector):
        assert selector == "body"
        return SimpleNamespace(inner_text=lambda **kwargs: "Visible game controls")

    def close(self, **kwargs):
        self.closed = True
        self.calls.append(("close", kwargs))


def test_injected_page_uses_png_mouse_and_visible_body_public_apis():
    page = FakePage()
    transport = PlaywrightTransport(page, timeout_ms=1234)
    assert transport.screenshot().startswith(b"\x89PNG")
    transport.click(10, 20)
    assert transport.read_visible_text() == "Visible game controls"
    assert ("click", 10.0, 20.0) in page.calls
    screenshot_call = next(call for call in page.calls if call[0] == "screenshot")
    assert screenshot_call[1] == {"type": "png", "full_page": False, "timeout": 1234}
    assert ("default_timeout", 1234) in page.calls
    transport.close()
    transport.close()
    assert page.closed
    assert len([call for call in page.calls if call[0] == "close"]) == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "https://colonist.io.evil.test/",
        "file:///tmp/game.html",
        "about:blank",
    ],
)
def test_transport_rejects_other_origins_before_actions(url):
    page = FakePage(url)
    transport = PlaywrightTransport(page)
    for operation in [
        transport.screenshot,
        lambda: transport.click(10, 20),
        transport.read_visible_text,
    ]:
        with pytest.raises(BrowserConnectionError):
            operation()
    assert not any(call[0] in ("click", "screenshot") for call in page.calls)


@pytest.mark.parametrize(
    "url", ["http://localhost:1234/", "http://127.0.0.1:1234/", "http://[::1]:1234/"]
)
def test_transport_allows_explicit_local_fixture_hosts(url):
    assert PlaywrightTransport(FakePage(url)).read_visible_text() == "Visible game controls"


def test_transport_rechecks_origin_and_connection_for_each_operation():
    page = FakePage()
    transport = PlaywrightTransport(page)
    transport.screenshot()
    page.url = "https://other.test/"
    with pytest.raises(BrowserConnectionError):
        transport.click(1, 2)
    with pytest.raises(BrowserConnectionError):
        PlaywrightTransport(FakePage(connected=False)).screenshot()
    page.url = "https://colonist.io/"
    page.closed = True
    with pytest.raises(BrowserConnectionError):
        transport.screenshot()


def test_capture_writes_real_transport_bytes_and_sanitized_metadata(tmp_path):
    page = FakePage("https://colonist.io/?session_token=do-not-store#secret")
    transport = PlaywrightTransport(page)
    artifact = transport.capture(tmp_path / "capture.png")
    assert artifact.image_path.read_bytes() == b"\x89PNG\r\n\x1a\nfixture"
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["origin"] == "https://colonist.io"
    assert metadata["visible_text"] == "Visible game controls"
    assert metadata["verified"] is False
    assert "do-not-store" not in artifact.metadata_path.read_text()
    assert metadata["screenshot_sha256"] == artifact.screenshot_sha256


def test_malformed_png_and_invalid_coordinates_are_rejected(tmp_path):
    page = FakePage()
    page.screenshot = lambda **kwargs: "not bytes"
    transport = PlaywrightTransport(page)
    with pytest.raises(BrowserConnectionError):
        transport.capture(tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()
    for point in [(float("nan"), 1), (-1, 2), (1, float("inf"))]:
        with pytest.raises(ValueError):
            transport.click(*point)


def fake_runtime(*, fail_launch=False, fail_navigation=False):
    calls = []
    page = FakePage("about:blank")

    def goto(url, **kwargs):
        calls.append(("goto", url, kwargs))
        if fail_navigation:
            raise RuntimeError("navigation failed")
        page.url = url

    page.goto = goto
    context = SimpleNamespace(
        new_page=lambda: page, pages=[], close=lambda: calls.append("context_close")
    )
    browser = SimpleNamespace(
        new_context=lambda: context,
        close=lambda: calls.append("browser_close"),
        is_connected=lambda: True,
    )
    context.browser = browser

    def launch(**kwargs):
        calls.append(("launch", kwargs))
        if fail_launch:
            raise RuntimeError("launch failed")
        return browser

    def launch_persistent_context(path, **kwargs):
        calls.append(("persistent", path, kwargs))
        return context

    runtime = SimpleNamespace(
        chromium=SimpleNamespace(
            launch=launch, launch_persistent_context=launch_persistent_context
        ),
        stop=lambda: calls.append("runtime_stop"),
    )
    return runtime, calls


def test_session_is_headed_and_isolated_by_default_and_cleans_up(monkeypatch):
    runtime, calls = fake_runtime()
    monkeypatch.setattr("colonist_rl.browser._start_playwright", lambda: runtime)
    with PlaywrightSession() as session:
        assert session.page.url == "https://colonist.io/"
        assert session.transport.read_visible_text() == "Visible game controls"
    assert (
        next(call for call in calls if isinstance(call, tuple) and call[0] == "launch")[1][
            "headless"
        ]
        is False
    )
    assert "context_close" in calls and "browser_close" in calls and "runtime_stop" in calls
    assert not any(isinstance(call, tuple) and call[0] == "persistent" for call in calls)


def test_persistent_profile_is_used_only_when_explicitly_supplied(monkeypatch, tmp_path):
    runtime, calls = fake_runtime()
    monkeypatch.setattr("colonist_rl.browser._start_playwright", lambda: runtime)
    with PlaywrightSession(
        url="http://localhost:1234/", headed=False, user_data_dir=tmp_path
    ) as session:
        assert session.page.url == "http://localhost:1234/"
    persistent = next(call for call in calls if isinstance(call, tuple) and call[0] == "persistent")
    assert persistent[1] == str(tmp_path)
    assert persistent[2]["headless"] is True


@pytest.mark.parametrize("failure", ["launch", "navigation"])
def test_failed_session_creation_releases_acquired_resources(monkeypatch, failure):
    runtime, calls = fake_runtime(
        fail_launch=failure == "launch", fail_navigation=failure == "navigation"
    )
    monkeypatch.setattr("colonist_rl.browser._start_playwright", lambda: runtime)
    with pytest.raises(RuntimeError), PlaywrightSession():
        pytest.fail("failed session unexpectedly opened")
    assert "runtime_stop" in calls
    if failure == "navigation":
        assert "context_close" in calls and "browser_close" in calls


def test_session_rejects_unknown_origin_before_launch(monkeypatch):
    monkeypatch.setattr(
        "colonist_rl.browser._start_playwright", lambda: pytest.fail("unexpected launch")
    )
    with pytest.raises(BrowserConnectionError), PlaywrightSession(url="https://other.test/"):
        pass


@pytest.mark.skipif(
    os.environ.get("COLONIST_RL_BROWSER_SMOKE") != "1", reason="opt-in local Chromium smoke test"
)
def test_optional_local_html_playwright_smoke(tmp_path):
    if importlib.util.find_spec("playwright") is None:
        pytest.skip("install the optional browser dependency first")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'<html><body><button style="position:absolute;left:0;top:0;width:150px;height:80px" onclick="this.innerText=\'Clicked\'">Click fixture</button></body></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            session = PlaywrightSession(url=f"http://127.0.0.1:{server.server_port}/", headed=False)
            session.__enter__()
        except Exception as error:
            if "Executable doesn't exist" in str(error) or "browserType.launch" in str(error):
                pytest.skip("Chromium is unavailable; run playwright install chromium")
            raise
        try:
            assert "Click fixture" in session.transport.read_visible_text()
            session.transport.click(50, 40)
            assert "Clicked" in session.transport.read_visible_text()
            artifact = session.transport.capture(tmp_path / "local.png")
            assert artifact.image_path.read_bytes().startswith(b"\x89PNG")
        finally:
            session.close()
    finally:
        server.shutdown()
        server.server_close()


def test_capture_rejects_a_redirect_before_writing_artifacts(tmp_path):
    page = FakePage()

    def redirected_screenshot(**kwargs):
        page.url = "https://other.test/"
        return b"\x89PNG\r\n\x1a\nfixture"

    page.screenshot = redirected_screenshot
    with pytest.raises(BrowserConnectionError):
        PlaywrightTransport(page).capture(tmp_path / "redirect.png")
    assert not (tmp_path / "redirect.png").exists()


def test_cleanup_attempts_all_resources_and_preserves_original_launch_error(monkeypatch):
    runtime, calls = fake_runtime(fail_navigation=True)
    browser = runtime.chromium.launch()
    context = browser.new_context()

    def broken_close():
        calls.append("context_close")
        raise RuntimeError("cleanup failed")

    context.close = broken_close
    monkeypatch.setattr("colonist_rl.browser._start_playwright", lambda: runtime)
    with pytest.raises(RuntimeError, match="navigation failed"), PlaywrightSession():
        pass
    assert "context_close" in calls and "browser_close" in calls and "runtime_stop" in calls
