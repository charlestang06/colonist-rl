"""Offline public Page/Mouse doubles exercise guarded drag input only."""

from types import SimpleNamespace

import pytest

from colonist_rl.adapters.colonist.browser import BrowserConnectionError, PlaywrightTransport


class Page:
    def __init__(self):
        self.url = "http://localhost:1234/fixture"
        self.closed = False
        self.connected = True
        self.checks = 0
        self.on_check = lambda: None
        self.calls = []
        self.on_mouse = lambda call: None
        self.context = SimpleNamespace(browser=SimpleNamespace(is_connected=lambda: self.connected))
        self.mouse = SimpleNamespace(
            move=lambda x, y, **kwargs: self.mouse_call(("move", x, y, kwargs)),
            down=lambda: self.mouse_call(("down",)),
            up=lambda: self.mouse_call(("up",)),
            click=lambda x, y: self.mouse_call(("click", x, y)),
        )

    def mouse_call(self, call):
        self.calls.append(call)
        self.on_mouse(call)

    def is_closed(self):
        self.checks += 1
        self.on_check()
        return self.closed

    def set_default_timeout(self, timeout):
        pass

    def set_default_navigation_timeout(self, timeout):
        pass

    def screenshot(self, **kwargs):
        return b"\x89PNG\r\n\x1a\nsynthetic"


DRAG = [
    ("move", 10.0, 20.0, {}),
    ("down",),
    ("move", 80.0, 90.0, {"steps": 4}),
    ("up",),
]


def test_drag_uses_exact_public_mouse_sequence_and_checks_each_boundary():
    page = Page()
    PlaywrightTransport(page).drag(10, 20, 80, 90, steps=4)
    assert page.calls == DRAG
    assert page.checks == 9  # Initial origin plus before/after each public operation.


def test_drag_defaults_steps_and_preserves_old_click_and_screenshot():
    page = Page()
    transport = PlaywrightTransport(page)
    transport.drag(10, 20, 80, 90)
    transport.click(2, 3)
    assert page.calls == [
        *DRAG[:2],
        ("move", 80.0, 90.0, {"steps": 10}),
        ("up",),
        ("click", 2.0, 3.0),
    ]
    assert transport.screenshot() == b"\x89PNG\r\n\x1a\nsynthetic"


@pytest.mark.parametrize("coordinate", [True, -1, float("nan"), float("inf"), "10", None])
@pytest.mark.parametrize("index", range(4))
def test_drag_rejects_invalid_coordinate_before_input(coordinate, index):
    page = Page()
    values = [10, 20, 80, 90]
    values[index] = coordinate
    with pytest.raises((TypeError, ValueError)):
        PlaywrightTransport(page).drag(*values)
    assert page.calls == []


@pytest.mark.parametrize("steps", [True, False, 0, -1, 1.5, "4", None])
def test_drag_rejects_invalid_steps_before_input(steps):
    page = Page()
    with pytest.raises((TypeError, ValueError)):
        PlaywrightTransport(page).drag(10, 20, 80, 90, steps=steps)
    assert page.calls == []


@pytest.mark.parametrize(
    "boundary,prefix", [(2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (7, 3), (8, 3), (9, 4)]
)
@pytest.mark.parametrize("change", ["origin", "disconnected", "closed", "forbidden"])
def test_drag_rechecks_connection_and_original_origin_before_and_after_each_step(
    boundary, prefix, change
):
    page = Page()

    def mutate():
        if page.checks == boundary:
            if change == "origin":
                page.url = "http://localhost:5678/other"  # Also allowed, but not original.
            elif change == "forbidden":
                page.url = "https://example.com/"
            elif change == "disconnected":
                page.connected = False
            else:
                page.closed = True

    page.on_check = mutate
    with pytest.raises(BrowserConnectionError):
        PlaywrightTransport(page).drag(10, 20, 80, 90, steps=4)
    assert page.calls == DRAG[:prefix]  # No cleanup release on the changed boundary.


@pytest.mark.parametrize("step", range(4))
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_failure_releases_only_when_drag_may_be_pressed_and_preserves_exception(
    step, error_type
):
    page = Page()
    original = error_type("original mouse failure")

    def fail_once(call):
        if len(page.calls) == step + 1:
            raise original

    page.on_mouse = fail_once
    with pytest.raises(error_type) as caught:
        PlaywrightTransport(page).drag(10, 20, 80, 90, steps=4)
    assert caught.value is original
    assert page.calls == DRAG[: step + 1] + ([("up",)] if step >= 1 else [])


@pytest.mark.parametrize("change", ["origin", "disconnected", "closed"])
def test_partial_failure_does_not_release_after_leaving_original_connected_origin(change):
    page = Page()
    original = RuntimeError("move failed after navigation")

    def fail(call):
        if len(page.calls) == 3:
            if change == "origin":
                page.url = "http://127.0.0.1:1234/"
            elif change == "disconnected":
                page.connected = False
            else:
                page.closed = True
            raise original

    page.on_mouse = fail
    with pytest.raises(RuntimeError) as caught:
        PlaywrightTransport(page).drag(10, 20, 80, 90, steps=4)
    assert caught.value is original
    assert page.calls == DRAG[:3]


@pytest.mark.parametrize("cleanup_error", [RuntimeError("cleanup"), KeyboardInterrupt()])
def test_cleanup_failure_cannot_replace_original_exception(cleanup_error):
    page = Page()
    original = RuntimeError("original")

    def fail(call):
        if call[0] == "down":
            raise original
        if call[0] == "up":
            raise cleanup_error

    page.on_mouse = fail
    with pytest.raises(RuntimeError) as caught:
        PlaywrightTransport(page).drag(10, 20, 80, 90, steps=4)
    assert caught.value is original
    assert page.calls == [*DRAG[:2], ("up",)]


@pytest.mark.parametrize("change", ["forbidden", "disconnected", "closed"])
def test_drag_rejects_unsafe_initial_page_before_input(change):
    page = Page()
    if change == "forbidden":
        page.url = "https://example.com/"
    elif change == "disconnected":
        page.connected = False
    else:
        page.closed = True
    with pytest.raises(BrowserConnectionError):
        PlaywrightTransport(page).drag(10, 20, 80, 90)
    assert page.calls == []


def test_drag_establishes_current_allowed_origin_at_call_start():
    page = Page()
    transport = PlaywrightTransport(page)
    page.url = "http://127.0.0.1:5678/fixture"
    transport.drag(10, 20, 80, 90, steps=4)
    assert page.calls == DRAG
