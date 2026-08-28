"""Clinical App Header component."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_header(cv_load_result: Any = None) -> None:
    """Render a patient-facing product header without implementation telemetry."""
    st.markdown(
        """
        <div class="clinical-header">
            <div class="clinical-brand">
                <div class="clinical-brand-icon">💊</div>
                <div>
                    <h1 class="clinical-brand-title">Pill Safety</h1>
                    <p class="clinical-brand-subtitle">Medication identification and interaction screening</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
