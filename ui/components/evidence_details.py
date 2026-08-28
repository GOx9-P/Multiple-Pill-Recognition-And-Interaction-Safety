"""Structured evidence review component for desktop users."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..adapters.view_models import ImageQualityViewModel, PillViewModel


def _display_status(status: str, is_manual_override: bool) -> str:
    """Convert the internal decision state into a concise user-facing label."""
    if is_manual_override:
        return "Manually confirmed"
    return {
        "accepted": "Identified",
        "ambiguous": "Needs review",
        "unresolved": "Unresolved",
        "rejected": "Not usable",
    }.get(status, "Under review")


def _format_percentage(value: float | None) -> str:
    """Format optional confidence values consistently for the evidence panel."""
    return "Not available" if value is None else f"{value * 100:.0f}%"


def render_evidence_details(
    pills: list[PillViewModel],
    quality: ImageQualityViewModel,
    raw_cv_data: Any,
) -> None:
    """Render focused, readable evidence while keeping raw diagnostics optional."""
    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">Evidence details</h3>
            <span style="font-size: 0.8rem; color: var(--text-muted);">Review the evidence behind each medication result</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pill_ids = [pill.instance_id for pill in pills]
    selected_id = st.selectbox(
        "Medication evidence",
        options=pill_ids,
        format_func=lambda instance_id: f"Medication #{pill_ids.index(instance_id) + 1} · {instance_id}",
        key="evidence_details_pill_selector",
    )
    pill = next(item for item in pills if item.instance_id == selected_id)

    with st.container(border=True):
        decision_col, visual_col, action_col = st.columns(3)
        with decision_col:
            st.caption("DECISION")
            st.markdown(f"**{_display_status(pill.status, pill.is_manual_override)}**")
            st.metric("Match confidence", _format_percentage(pill.match_confidence))
        with visual_col:
            st.caption("VISIBLE EVIDENCE")
            st.markdown(f"**Shape:** {pill.shape.title()} ({_format_percentage(pill.shape_confidence)})")
            st.markdown(f"**Color:** {pill.color_primary.title()} ({_format_percentage(pill.color_confidence)})")
            st.markdown(f"**Imprint:** `{pill.imprint_raw}` ({_format_percentage(pill.imprint_confidence)})")
        with action_col:
            st.caption("NEXT STEP")
            st.markdown(pill.required_action or "No additional action is currently required.")
            if pill.scope_warning:
                st.warning(pill.scope_warning)

    st.markdown("**Candidate comparison**")
    if pill.top_candidates:
        candidate_rows = [
            {
                "Rank": candidate.rank,
                "Medication": candidate.product_name,
                "Match score": f"{candidate.final_score * 100:.1f}%",
                "Imprint": f"{(candidate.imprint_score or 0) * 100:.0f}%",
                "Shape": f"{(candidate.shape_score or 0) * 100:.0f}%",
                "Color": f"{(candidate.color_score or 0) * 100:.0f}%",
            }
            for candidate in pill.top_candidates
        ]
        st.dataframe(candidate_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No database candidates are available for this medication yet.")

    quality_tab, raw_data_tab = st.tabs(["Image quality", "Raw technical data"])

    with quality_tab:
        quality_col, blur_col, glare_col, lighting_col = st.columns(4)
        quality_col.metric("Overall quality", quality.status.replace("_", " ").title())
        blur_col.metric("Blur score", f"{quality.blur_score:.2f}")
        glare_col.metric("Glare", "Detected" if quality.glare_detected else "Not detected")
        lighting_col.metric("Lighting", "Needs review" if quality.lighting_warning else "Normal")
        if quality.notes:
            st.caption(" · ".join(quality.notes))

    with raw_data_tab:
        st.caption("For troubleshooting only. This data is not required to interpret the medication result.")
        if isinstance(raw_cv_data, dict):
            st.json(raw_cv_data)
        else:
            st.code(str(raw_cv_data), language="json")
