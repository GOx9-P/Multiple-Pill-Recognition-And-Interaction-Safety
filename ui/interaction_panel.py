"""Interaction-checker presentation panel."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from .config import UI_DEMO_MODE
from .demo_data import get_mock_interaction_result, pill_display_fields
from pill_safety.rag.interaction_checker import InteractionCheckResult, InteractionWarning


def _severity_color(severity: str) -> str:
    s = severity.strip().lower()
    if s == "severe":
        return "#DC2626"
    if s == "moderate":
        return "#F59E0B"
    if s == "mild":
        return "#EAB308"
    return "#3B82F6"


def _render_warning(warning: InteractionWarning) -> None:
    color = _severity_color(warning.severity)
    mechanism_html = (
        f'<div class="interaction-warning-section"><strong>Cơ chế:</strong> {warning.mechanism}</div>'
        if warning.mechanism
        else ""
    )
    risk_html = (
        f'<div class="interaction-warning-section"><strong>Nguy cơ lâm sàng:</strong> {warning.clinical_risk}</div>'
        if warning.clinical_risk
        else ""
    )
    management_html = (
        f'<div class="interaction-warning-section"><strong>Xử trí:</strong> {warning.management}</div>'
        if warning.management
        else ""
    )
    st.markdown(
        f"""
        <div class="interaction-warning-card">
            <div class="interaction-warning-header" style="background:{color}">
                <span class="interaction-warning-icon">⚠️</span>
                <span class="interaction-warning-severity">{warning.severity.capitalize()}</span>
            </div>
            <div class="interaction-warning-body">
                <div class="interaction-warning-message">{warning.message}</div>
                {mechanism_html}
                {risk_html}
                {management_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interaction_panel(check_interactions: Callable[[Any], InteractionCheckResult]) -> None:
    """Render the interaction-check flow with demo support."""

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-header">
                <span class="section-badge purple">2</span>
                <h2 class="section-title">DRUG INTERACTION CHECKER</h2>
            </div>
            <div class="purple-divider"></div>
            """,
            unsafe_allow_html=True,
        )

        cv_output = st.session_state.get("cv_output")
        demo_mode = st.session_state.get("demo_mode", UI_DEMO_MODE)
        can_check = cv_output is not None
        result = st.session_state.get("interaction_result")

        if not can_check:
            st.markdown(
                """
                <div class="interaction-empty-state">
                    <span class="interaction-empty-icon">📋</span>
                    <div class="interaction-empty-title">No interaction analysis yet</div>
                    <div class="interaction-empty-subtitle">Upload an image and recognize pills to enable interaction checking.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Kiểm tra tương tác", disabled=True, type="primary")
            return

        if result is None:
            st.markdown("#### Detected Drugs")

            pills = cv_output.pills if hasattr(cv_output, "pills") else []
            for pill in pills:
                fields = pill_display_fields(pill)
                st.markdown(
                    f'<div class="interaction-drug-item">✓ {fields["instance_id"]}</div>',
                    unsafe_allow_html=True,
                )

            if st.button("Kiểm tra tương tác", type="primary", use_container_width=True):
                # TODO: thay bằng DdiLookupService/LlmReportGenerator thật khi backend sẵn sàng
                if demo_mode and st.session_state.get("cv_output_is_demo"):
                    st.session_state["interaction_result"] = get_mock_interaction_result()
                    st.session_state["interaction_result_is_demo"] = True
                else:
                    st.session_state["interaction_result"] = check_interactions(cv_output)
                    st.session_state["interaction_result_is_demo"] = False
                st.rerun()
            return

        is_demo = st.session_state.get("interaction_result_is_demo", False)
        if is_demo:
            st.caption("Kết quả minh họa giao diện — không phải dự đoán AI thực tế.")
        else:
            st.caption("Kết quả hiện là stub xác định; chưa có hệ thống RAG/LLM.")

        st.markdown("#### Kết quả tương tác")

        if not result.warnings:
            st.success("✅ Không phát hiện tương tác đáng kể")
        else:
            for warning in result.warnings:
                _render_warning(warning)
