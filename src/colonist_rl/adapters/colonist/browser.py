"""Optional public-API Playwright transport for an explicitly requested web player.

Importing this module does not import Playwright, start a browser, or connect to
any site. Browser sessions are isolated unless a profile is explicitly supplied.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

DEFAULT_URL = "https://colonist.io/"
_ALLOWED_HOSTS = frozenset(("colonist.io", "localhost", "127.0.0.1", "::1"))
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class BrowserConnectionError(RuntimeError):
    """A transport is disconnected, redirected, or returned invalid data."""


def _origin(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if (
            parsed.hostname not in _ALLOWED_HOSTS
            or parsed.scheme not in ("http", "https")
            or (parsed.hostname == "colonist.io" and parsed.scheme != "https")
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise BrowserConnectionError("browser URL must use colonist.io or a local fixture host")
        _ = parsed.port  # Validate a malformed port before launching or writing metadata.
        return f"{parsed.scheme}://{parsed.netloc}"
    except (ValueError, TypeError) as error:
        raise BrowserConnectionError("invalid browser URL") from error


def _timeout(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("timeout_ms must be a positive integer")
    return value


@dataclass(frozen=True)
class BrowserCapture:
    image_path: Path
    metadata_path: Path
    screenshot_sha256: str


class PlaywrightTransport:
    """Implement BrowserTransport using an injected synchronous Playwright Page.

    Only public Page/Mouse/Locator APIs are used. Each operation rechecks the
    current origin and connection. ``close`` owns this page, not other pages in
    its context. No cookies, local storage, credentials, or private APIs are read.
    """

    def __init__(self, page: Any, *, timeout_ms: int = 10_000):
        self.page = page
        self.timeout_ms = _timeout(timeout_ms)
        self._closed = False
        page.set_default_timeout(self.timeout_ms)
        page.set_default_navigation_timeout(self.timeout_ms)

    def _check(self) -> str:
        if self._closed or self.page.is_closed():
            raise BrowserConnectionError("browser page is closed")
        browser = self.page.context.browser
        if browser is not None and not browser.is_connected():
            raise BrowserConnectionError("browser is disconnected")
        return _origin(self.page.url)

    def screenshot(self) -> bytes:
        before_origin = self._check()
        result = self.page.screenshot(type="png", full_page=False, timeout=self.timeout_ms)
        if self._check() != before_origin:
            raise BrowserConnectionError("page origin changed while capturing")
        if not isinstance(result, bytes) or not result.startswith(_PNG_SIGNATURE):
            raise BrowserConnectionError("Playwright screenshot must return PNG bytes")
        return result

    def click(self, x: float, y: float) -> None:
        self._check()
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value < 0
            for value in (x, y)
        ):
            raise ValueError("click coordinates must be finite nonnegative numbers")
        self.page.mouse.click(float(x), float(y))
        self._check()

    def read_visible_text(self) -> str:
        before_origin = self._check()
        # DOM innerText represents rendered body text; hidden DOM data is not read.
        result = self.page.locator("body").inner_text(timeout=self.timeout_ms)
        if self._check() != before_origin:
            raise BrowserConnectionError("page origin changed while reading visible text")
        if not isinstance(result, str):
            raise BrowserConnectionError("body inner_text must return text")
        return result

    def capture(
        self, output_path: str | Path, metadata_path: str | Path | None = None
    ) -> BrowserCapture:
        """Write an actual PNG and unverified visible-text metadata for annotation.

        Metadata stores only the origin, dropping URL paths, queries, fragments,
        and any credentials. No screenshot-derived Observation is invented.
        """
        origin = self._check()
        screenshot = self.screenshot()
        visible_text = self.read_visible_text()
        if self._check() != origin:
            raise BrowserConnectionError("page origin changed during capture")
        image_path = Path(output_path).expanduser().resolve()
        text_path = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else image_path.with_suffix(".json")
        )
        if text_path == image_path:
            raise ValueError("screenshot and metadata paths must differ")
        digest = hashlib.sha256(screenshot).hexdigest()
        metadata = {
            "format": "colonist-rl.browser-capture",
            "version": 1,
            "verified": False,
            "origin": origin,
            "captured_at": datetime.now(UTC).isoformat(),
            "screenshot": image_path.name,
            "screenshot_sha256": digest,
            "visible_text": visible_text,
        }
        image_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(screenshot)
        text_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return BrowserCapture(image_path, text_path, digest)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if not self.page.is_closed():
                self.page.close(run_before_unload=False)


def _start_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise ImportError(
            "install colonist-rl[browser] and run playwright install chromium"
        ) from error
    return sync_playwright().start()


class PlaywrightSession:
    """Context-managed Chromium session; headed and isolated by default.

    The browser package and Chromium executable are optional installations.
    Supplying ``user_data_dir`` explicitly opts into that persistent profile;
    existing browser profiles are never located or reused automatically.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        headed: bool = True,
        user_data_dir: str | Path | None = None,
        timeout_ms: int = 10_000,
    ):
        _origin(url)
        self.url, self.headed = url, headed
        self.user_data_dir = (
            Path(user_data_dir).expanduser().resolve() if user_data_dir is not None else None
        )
        self.timeout_ms = _timeout(timeout_ms)
        self.page = None
        self.transport: PlaywrightTransport | None = None
        self._runtime = self._browser = self._context = None

    def __enter__(self) -> Self:
        if self._runtime is not None:
            raise RuntimeError("browser session is already open")
        try:
            self._runtime = _start_playwright()
            if self.user_data_dir is None:
                self._browser = self._runtime.chromium.launch(
                    headless=not self.headed, timeout=self.timeout_ms
                )
                self._context = self._browser.new_context()
            else:
                self._context = self._runtime.chromium.launch_persistent_context(
                    str(self.user_data_dir),
                    headless=not self.headed,
                    timeout=self.timeout_ms,
                )
                self._browser = self._context.browser
            self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self.transport = PlaywrightTransport(self.page, timeout_ms=self.timeout_ms)
            self.page.goto(self.url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            self.transport._check()
            return self
        except BaseException:
            # Cleanup must preserve the original launch/navigation exception.
            with suppress(Exception):
                self.close()
            raise

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception_type is None:
            self.close()
        else:
            with suppress(Exception):
                self.close()
        return False

    def capture(
        self, output_path: str | Path, metadata_path: str | Path | None = None
    ) -> BrowserCapture:
        if self.transport is None:
            raise BrowserConnectionError("browser session is not open")
        return self.transport.capture(output_path, metadata_path)

    def close(self) -> None:
        runtime, browser, context = self._runtime, self._browser, self._context
        self._runtime = self._browser = self._context = None
        self.page = self.transport = None
        # ExitStack attempts every cleanup even when an earlier cleanup raises.
        with ExitStack() as cleanup:
            if runtime is not None:
                cleanup.callback(runtime.stop)
            if browser is not None:
                cleanup.callback(browser.close)
            if context is not None:
                cleanup.callback(context.close)
