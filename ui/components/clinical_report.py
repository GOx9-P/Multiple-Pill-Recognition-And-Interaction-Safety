"""Clinical Report Component."""

from __future__ import annotations

import streamlit as st

from ..adapters.view_models import SafetyReportViewModel


def render_clinical_report(report: SafetyReportViewModel) -> None:
    """Render the readable safety report inside its own focused scrollable canvas."""
    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">Safety summary</h3>
            <span style="font-size: 0.8rem; color: var(--text-muted);">Review before making medication decisions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_btn = st.columns([3, 1])
    with col_btn:
        st.download_button(
            "Download report (.txt)",
            data=report.formatted_report_text,
            file_name=f"clinical_pill_safety_report_{report.session_id}.txt",
            mime="text/plain",
            key=f"btn_dl_report_{report.session_id}",
            use_container_width=True,
        )

    with st.container(height=360, border=True):
        st.markdown(report.formatted_report_text)
