"""Mobile Navigation Bar Component (iOS Tab Bar style)."""

from __future__ import annotations

import streamlit as st


def render_mobile_bottom_nav() -> str:
    """Render mobile segmented navigation tab bar.
    
    Tabs:
    - 'scan': Quét (Analyze)
    - 'drugs': Tra Cứu (Search)
    - 'ddi': Tương Tác (DDI Pair)
    """
    st.session_state.setdefault("mobile_active_tab", "scan")

    options_map = {
        "Quét thuốc": "scan",
        "Tra cứu": "drugs",
        "Tương tác": "ddi",
    }

    current_tab = st.session_state.get("mobile_active_tab", "scan")
    current_label = next((lbl for lbl, key in options_map.items() if key == current_tab), "Quét thuốc")

    with st.container(key="mobile_bottom_nav"):
        selected_label = st.radio(
            "Điều hướng chính",
            options=list(options_map.keys()),
            index=list(options_map.keys()).index(current_label),
            horizontal=True,
            label_visibility="collapsed",
            key="mobile_nav_selector",
        )

    chosen_tab = options_map[selected_label]
    st.session_state["mobile_active_tab"] = chosen_tab
    return chosen_tab
