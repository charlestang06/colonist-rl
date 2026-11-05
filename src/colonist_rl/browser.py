"""Compatibility alias for :mod:`colonist_rl.adapters.colonist.browser`.

Both import paths resolve to one module so legacy patches of the optional
``_start_playwright`` factory still affect the canonical session implementation.
"""

import sys

from .adapters.colonist import browser as _browser

sys.modules[__name__] = _browser
