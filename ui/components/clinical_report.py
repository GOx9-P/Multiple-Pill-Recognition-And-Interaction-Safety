"""Clinical Report Component."""

from __future__ import annotations

import streamlit as st

from ..adapters.view_models import SafetyReportViewModel


def render_clinical_report(report: SafetyReportViewModel) -> None:
    """Render structured Markdown clinical safety summary with clean typography."""
    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">📄 3. Báo cáo tổng quan y khoa (Clinical Report)</h3>
            <span style="font-size: 0.8rem; color: var(--text-muted);">AI Decision Support Summary</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_meta, col_btn = st.columns([3, 1])
    with col_meta:
        st.caption(f"**Động cơ sinh báo cáo:** `{report.provider_used}` • **Session ID:** `{report.session_id}`")
    with col_btn:
        st.download_button(
            "📥 Tải Báo Cáo (.txt)",
            data=report.formatted_report_text,
            file_name=f"clinical_pill_safety_report_{report.session_id}.txt",
            mime="text/plain",
            key=f"btn_dl_report_{report.session_id}",
            use_container_width=True,
        )

    # Render Report Content in a readable container
    st.markdown(
        f'<div class="clinical-report-container">\n\n{report.formatted_report_text}\n\n</div>',
        unsafe_allow_html=True,
    )
