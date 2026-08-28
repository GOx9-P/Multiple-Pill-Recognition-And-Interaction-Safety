"""Overall Clinical Safety Banner Component."""

from __future__ import annotations

import streamlit as st

from ..adapters.view_models import SafetyReportViewModel


def render_safety_banner(report: SafetyReportViewModel) -> None:
    """Render the high-priority clinical severity banner based on the evaluation result."""
    severity = report.overall_severity.lower()

    if severity == "critical":
        title = "Critical interaction requires professional review"
        desc = "A serious medication interaction was found. Consult a qualified healthcare professional before use."
        icon = "⛔"
        css_class = "critical"
    elif severity == "moderate":
        title = "Interaction or duplicate ingredient needs review"
        desc = "A potential interaction or duplicate ingredient was found. Review the medication list with a healthcare professional."
        icon = "⚠️"
        css_class = "moderate"
    elif severity == "unresolved":
        title = "Some medications need confirmation"
        desc = "One or more medications could not be identified confidently. Confirm them manually before relying on the safety result."
        icon = "❓"
        css_class = "unresolved"
    else:
        title = "No known harmful interaction found"
        desc = "No harmful interaction was found among the medications identified in the current database."
        icon = " "
        css_class = "safe"

    st.markdown(
        f"""
        <div class="severity-banner {css_class}">
            <div class="severity-icon-badge">{icon}</div>
            <div>
                <h3 class="severity-title">{title}</h3>
                <p class="severity-description">{desc}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
