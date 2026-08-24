"""Mobile UI Components for iPhone 17 Pro Max simulator & smartphone devices."""

from __future__ import annotations

from ui.components.mobile.iphone_frame import render_dynamic_island, render_home_indicator
from ui.components.mobile.mobile_header import render_mobile_header
from ui.components.mobile.mobile_bottom_nav import render_mobile_bottom_nav
from ui.components.mobile.mobile_pill_row import render_mobile_pill_row
from ui.components.mobile.mobile_interaction_card import render_mobile_interaction_card
from ui.components.mobile.mobile_duplicate_card import render_mobile_duplicate_card

__all__ = [
    "render_dynamic_island",
    "render_home_indicator",
    "render_mobile_header",
    "render_mobile_bottom_nav",
    "render_mobile_pill_row",
    "render_mobile_interaction_card",
    "render_mobile_duplicate_card",
]
