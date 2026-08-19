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
) -> None:
    """Render the interactive annotated image and image quality assessment."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; font-size: 1rem; color: var(--text-primary);">🔍 Bằng chứng thị giác (CV Visual Canvas)</h4>
            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">YOLOv11-Seg • OCR</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Draw image with overlay
    annotated_image = draw_cv_overlay(image, pills, selected_pill_id)
    st.image(annotated_image, use_container_width=True)

    # 2. Pill Index Selector (for interactive focus)
    if pills:
        options = [p.instance_id for p in pills]
        current_idx = options.index(selected_pill_id) if selected_pill_id in options else 0

        selected_option = st.selectbox(
            "🎯 Tiêu điểm đối soát (Chọn viên thuốc để làm nổi bật):",
            options=options,
            index=current_idx,
            key="select_active_pill_box",
        )
        if selected_option != selected_pill_id:
            on_pill_selected(selected_option)

    # 3. Compact Image Quality Chips
    blur_label = f"Độ mờ: {quality.blur_score:.2f} ({'Tốt' if quality.blur_score < 0.3 else 'Cao'})"
    glare_label = f"Chói sáng: {'Có' if quality.glare_detected else 'Không'}"
    light_label = f"Ánh sáng: {'Cảnh báo' if quality.lighting_warning else 'Bình thường'}"

    st.markdown(
        f"""
        <div class="quality-chip-container">
            <span class="quality-chip">📸 Trạng thái ảnh: <strong>{quality.status.upper()}</strong></span>
            <span class="quality-chip">🔍 {blur_label}</span>
            <span class="quality-chip">✨ {glare_label}</span>
            <span class="quality-chip">💡 {light_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
