from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mobile_navigation_is_rendered_outside_scroll_viewport() -> None:
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    mobile_branch = source.split('with st.container(key="iphone_17_simulator"):', maxsplit=1)[1]

    nav_position = mobile_branch.index("render_mobile_bottom_nav()")
    scroll_position = mobile_branch.index('with st.container(key="mobile_scroll_viewport")')

    assert nav_position < scroll_position


def test_phone_preview_has_one_invisible_scroll_owner() -> None:
    source = (PROJECT_ROOT / "ui" / "styles.py").read_text(encoding="utf-8")

    assert "scrollbar-width: thin" not in source
    assert ".st-key-mobile_scroll_viewport::-webkit-scrollbar" in source
    assert "scrollbar-width: none" in source
    assert '.st-key-iphone_17_simulator [data-testid="stVerticalBlock"] {' not in source


def test_mobile_controls_meet_ios_default_touch_target() -> None:
    source = (PROJECT_ROOT / "ui" / "styles.py").read_text(encoding="utf-8")

    assert "min-height: 44px !important" in source


def test_bottom_navigation_has_a_visible_selected_state() -> None:
    source = (PROJECT_ROOT / "ui" / "styles.py").read_text(encoding="utf-8")

    assert ".st-key-mobile_bottom_nav label:has(input:checked)" in source


def test_mobile_is_the_default_demo_view() -> None:
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert 'st.session_state.setdefault("view_mode", "📱 iPhone 17 Pro Max Preview")' in source
