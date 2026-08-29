"""Visual Evidence Viewer Component."""

from __future__ import annotations

from typing import Callable

import streamlit as st
from PIL import Image

from ..adapters.view_models import ImageQualityViewModel, PillViewModel
from ..drawing_utils import draw_cv_overlay


def render_visual_viewer(
    image: Image.Image,
    pills: list[PillViewModel],
    quality: ImageQualityViewModel,
    selected_pill_id: str | None,
    on_pill_selected: Callable[[str], None],
    highlighted_pill_ids: list[str] | None = None,
    on_clear_interaction_highlight: Callable[[], None] | None = None,
) -> None:
    """Render the interactive annotated image and image quality assessment."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; font-size: 1rem; color: var(--text-primary);">Image review</h4>
            <span style="font-size: 0.75rem; color: var(--text-muted);">Select a medication to focus the overlay</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    highlighted_pill_ids = highlighted_pill_ids or []
    if highlighted_pill_ids:
        medication_numbers = [
            str(index)
            for index, pill in enumerate(pills, start=1)
            if pill.instance_id in highlighted_pill_ids
        ]
        st.info(
            "Safety finding highlighted for medication "
            + ", ".join(f"#{number}" for number in medication_numbers)
            + "."
        )
        if on_clear_interaction_highlight is not None:
            st.button(
                "Clear safety highlight",
                key="clear_interaction_highlight",
                on_click=on_clear_interaction_highlight,
            )

    # 1. Draw image with overlay
    annotated_image = draw_cv_overlay(
        image,
        pills,
        selected_pill_id,
        highlighted_pill_ids=set(highlighted_pill_ids),
    )
    st.image(annotated_image, use_container_width=True)

    # 2. Pill Index Selector (for interactive focus)
    if pills:
        options = [p.instance_id for p in pills]
        current_idx = options.index(selected_pill_id) if selected_pill_id in options else 0

        selected_option = st.selectbox(
            "Selected medication:",
            options=options,
            index=current_idx,
            key="select_active_pill_box",
        )
        if selected_option != selected_pill_id:
            on_pill_selected(selected_option)

    # 3. Compact Image Quality Chips
    blur_label = f"Blur: {quality.blur_score:.2f} ({'Low' if quality.blur_score < 0.3 else 'High'})"
    glare_label = f"Glare: {'Detected' if quality.glare_detected else 'Not detected'}"
    light_label = f"Lighting: {'Needs review' if quality.lighting_warning else 'Normal'}"

    st.markdown(
        f"""
        <div class="quality-chip-container">
            <span class="quality-chip">Image quality: <strong>{quality.status.upper()}</strong></span>
            <span class="quality-chip">🔍 {blur_label}</span>
            <span class="quality-chip">✨ {glare_label}</span>
            <span class="quality-chip">💡 {light_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
