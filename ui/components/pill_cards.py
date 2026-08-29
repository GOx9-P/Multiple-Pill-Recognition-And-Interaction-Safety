"""Detected pill cards for compact result review."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ..adapters.view_models import PillViewModel


def render_pill_cards(
    pills: list[PillViewModel],
    selected_pill_id: str | None,
    on_pill_selected: Callable[[str], None],
) -> None:
    """Render compact medication results with their supporting evidence."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; font-size: 1rem; color: var(--text-primary);">Recognized medications ({len(pills)})</h4>
            <span style="font-size: 0.75rem; color: var(--text-muted);">Select a medication to inspect it in the image</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pills:
        st.info("No medications were detected in this image.")
        return

    for index, pill in enumerate(pills, start=1):
        is_selected = selected_pill_id is not None and pill.instance_id == selected_pill_id
        active_class = "active" if is_selected else ""

        if pill.is_manual_override:
            badge_class = "manual"
            badge_text = "MANUALLY CONFIRMED"
        elif pill.status == "accepted":
            badge_class = "accepted"
            badge_text = "IDENTIFIED"
        elif pill.status == "ambiguous":
            badge_class = "ambiguous"
            badge_text = "NEEDS REVIEW"
        else:
            badge_class = "unresolved"
            badge_text = "UNRESOLVED"

        drug_display_name = pill.drug_name or "Medication not confidently identified"
        brand_display = f" ({pill.brand_name})" if pill.brand_name else ""
        identifier_row = ""
        if pill.rxcui:
            identifier_row = (
                f'<div class="pill-meta-row"><span>RxCUI / NDC</span>'
                f'<span>{pill.rxcui} / {pill.ndc or "N/A"}</span></div>'
            )

        st.markdown(
            f"""
            <div class="pill-card-item {active_class}">
                <div class="pill-card-top">
                    <span class="pill-instance-tag">MEDICATION #{index} | {pill.instance_id}</span>
                    <span class="pill-badge {badge_class}">{badge_text}</span>
                </div>
                <h4 class="pill-drug-name">{drug_display_name}{brand_display}</h4>
                <div class="pill-meta-row"><span>Shape</span><span>{pill.shape} ({pill.shape_confidence*100:.0f}%)</span></div>
                <div class="pill-meta-row"><span>Color</span><span>{pill.color_primary} ({pill.color_confidence*100:.0f}%)</span></div>
                <div class="pill-meta-row"><span>Imprint</span><span><code>{pill.imprint_raw}</code></span></div>
                <div class="pill-meta-row"><span>Score line</span><span>{'Visible' if pill.scoreline_visible else 'Not visible'}</span></div>
                {identifier_row}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"Evidence for medication #{index}", expanded=False):
            if pill.top_candidates:
                st.markdown("**Top database candidates**")
                for candidate in pill.top_candidates:
                    st.markdown(
                        f"- **Rank {candidate.rank}:** {candidate.product_name} "
                        f"| Match score: `{candidate.final_score * 100:.1f}%` "
                        f"(Imprint: {candidate.imprint_score or 0:.2f}, "
                        f"Shape: {candidate.shape_score or 0:.2f}, "
                        f"Color: {candidate.color_score or 0:.2f})"
                    )
            else:
                st.markdown(f"- **Shape confidence:** `{pill.shape_confidence or 0:.2f}` ({pill.shape})")
                st.markdown(f"- **Color confidence:** `{pill.color_confidence or 0:.2f}` ({pill.color_primary})")
                st.markdown(f"- **Imprint confidence:** `{pill.imprint_confidence or 0:.2f}`")
                candidates = ", ".join(pill.imprint_candidates) if pill.imprint_candidates else "Not available"
                st.markdown(f"- **Imprint candidates:** `{candidates}`")
