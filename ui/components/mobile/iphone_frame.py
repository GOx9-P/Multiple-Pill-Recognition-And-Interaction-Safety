"""iPhone 17 Pro Max Simulator Shell Components for Desktop Previews."""

from __future__ import annotations

import streamlit as st


def render_dynamic_island() -> None:
    """Render the Dynamic Island sensor notch at the top of the iPhone simulator."""
    st.markdown(
        """
        <div class="iphone-island-container">
            <div class="iphone-island">
                <span class="island-camera"></span>
                <span class="island-sensor"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_indicator() -> None:
    """Render the iOS Home Indicator bar at the bottom of the iPhone simulator."""
    st.markdown(
        """
        <div class="iphone-home-indicator-container">
            <div class="iphone-home-indicator"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
