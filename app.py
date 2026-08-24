"""Main entrypoint for the Pill Safety AI Clinical Decision Support Platform."""

from __future__ import annotations

import streamlit as st

from ui.components import render_disclaimer, render_header
from ui.components.mobile import (
    render_dynamic_island,
    render_home_indicator,
    render_mobile_bottom_nav,
)
from ui.model_loader import load_cv_pipeline
from ui.styles import inject_healthcare_css
from ui.views import (
    render_about_view,
    render_analyze_view,
    render_ddi_checker_view,
    render_drug_search_view,
)
from ui.views.mobile import (
    render_mobile_about_view,
    render_mobile_analyze_view,
    render_mobile_ddi_view,
    render_mobile_drug_search_view,
)


def main() -> None:
    """Streamlit application lifecycle entrypoint."""
    st.set_page_config(
        page_title="Pill Safety AI — Clinical Decision Support",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 1. Inject centralized healthcare & mobile CSS system
    inject_healthcare_css()

    # 2. Shared Session State Initialization
    st.session_state.setdefault("view_mode", "🖥️ Desktop View")
    st.session_state.setdefault("mobile_active_tab", "scan")

    # 3. Load ML models safely with caching (shared across all views)
    cv_load_result = load_cv_pipeline()

    # 4. View Switcher Toolbar Stage (Isolated above the device stage)
    st.markdown("<div class='preview-toolbar-stage'>", unsafe_allow_html=True)
    view_options = ["🖥️ Desktop View", "📱 iPhone 17 Pro Max Preview"]
    current_view_state = st.session_state.get("view_mode", "🖥️ Desktop View")
    view_index = view_options.index(current_view_state) if current_view_state in view_options else 0

    selected_view = st.radio(
        "Chế độ hiển thị",
        options=view_options,
        index=view_index,
        horizontal=True,
        label_visibility="collapsed",
        key="app_view_switcher",
    )
    st.session_state["view_mode"] = selected_view
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # VIEW MODE A: DESKTOP CLINICAL WORKSPACE
    # =========================================================================
    if selected_view == "🖥️ Desktop View":
        # Header with live Database & CV Health checks
        render_header(cv_load_result)

        # Desktop Navigation Tabs
        tabs = st.tabs([
            "🔬 Phân Tích & Đối Soát Đơn Thuốc (Analyze)",
            "📚 Tra Cứu Dược Thư (Drug Database)",
            "⚡ Kiểm Tra Tương Tác Cặp (DDI Checker)",
            "📖 Phương Pháp & Kiến Trúc (Methodology)",
        ])

        with tabs[0]:
            render_analyze_view(cv_load_result)

        with tabs[1]:
            render_drug_search_view()

        with tabs[2]:
            render_ddi_checker_view()

        with tabs[3]:
            render_about_view()

        # Persistent Clinical & Legal Disclaimer
        render_disclaimer()

    # =========================================================================
    # VIEW MODE B & C: IPHONE 17 PRO MAX SIMULATOR & MOBILE RESPONSIVE
    # =========================================================================
    else:
        with st.container(key="iphone_17_simulator"):
            # Dynamic Island at top notch
            render_dynamic_island()

            # Dedicated Inner Scroll Viewport (Hover Scroll Owner)
            with st.container(key="mobile_scroll_viewport"):
                active_tab = render_mobile_bottom_nav()

                if active_tab == "scan":
                    render_mobile_analyze_view(cv_load_result)
                elif active_tab == "drugs":
                    render_mobile_drug_search_view()
                elif active_tab == "ddi":
                    render_mobile_ddi_view()
                elif active_tab == "about":
                    render_mobile_about_view()

            # Fixed Home Indicator anchored at bottom chassis edge
            render_home_indicator()


if __name__ == "__main__":
    main()
