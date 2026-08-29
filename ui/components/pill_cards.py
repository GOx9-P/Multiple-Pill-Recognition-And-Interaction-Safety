"""Detected pill cards for compact result review."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ..adapters.view_models import PillViewModel
from .recapture_panel import render_retake_controls


def render_pill_cards(
    pills: list[PillViewModel],
    selected_pill_id: str | None,
    on_pill_selected: Callable[[str], None],
    on_retake_requested: Callable[[str], None] | None = None,
    on_recapture: Callable[[str, object], None] | None = None,
    on_manual_override: Callable[[str, str], None] | None = None,
    recapture_errors: dict[str, str] | None = None,
    focused_retake_pill_id: str | None = None,
) -> None:
    """Render compact medication results and immediate actions for each item."""
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

        focus_col, retake_col = st.columns(2, gap="small")
        with focus_col:
            st.button(
                "Show in image",
                key=f"focus_pill_{pill.instance_id}",
                on_click=on_pill_selected,
                args=(pill.instance_id,),
                use_container_width=True,
            )

        needs_retake = pill.status != "accepted" and not pill.is_manual_override
        if needs_retake and on_retake_requested is not None:
            with retake_col:
                st.button(
                    "Retake photo",
                    key=f"retake_pill_{pill.instance_id}",
                    on_click=on_retake_requested,
                    args=(pill.instance_id,),
                    type="primary",
                    use_container_width=True,
                )

            if (
                pill.instance_id == focused_retake_pill_id
                and on_recapture is not None
                and on_manual_override is not None
            ):
                with st.expander("Retake options", expanded=True):
                    render_retake_controls(
                        pill=pill,
                        on_recapture=on_recapture,
                        on_manual_override=on_manual_override,
                        errors=recapture_errors,
                    )
