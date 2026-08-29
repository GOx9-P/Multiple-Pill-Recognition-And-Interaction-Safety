"""Focused recapture controls for medications that still need review."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st
from PIL import Image

from ..adapters.view_models import PillViewModel


def _recapture_reason(pill: PillViewModel) -> str:
    """Return one short, user-facing reason for requesting a clearer image."""
    if pill.status == "ambiguous":
        return "The visible details match more than one possible medication."
    if not pill.imprint_raw or pill.imprint_raw.strip() in {"?", "-"}:
        return "The imprint is not clear enough to identify this medication."
    return "The current photo does not provide enough reliable detail to identify this medication."


def _show_current_crop(pill: PillViewModel) -> None:
    """Show the existing pill crop when it is available for visual reference."""
    if pill.crop_path and Path(pill.crop_path).is_file():
        st.image(pill.crop_path, caption=f"Current crop: {pill.instance_id}", use_container_width=True)
    else:
        st.caption("Current crop is unavailable.")


def render_retake_controls(
    pill: PillViewModel,
    on_recapture: Callable[[str, Image.Image], None],
    on_manual_override: Callable[[str, str], None],
    errors: dict[str, str] | None = None,
) -> None:
    """Render the focused retake and manual-confirmation controls for one pill."""
    errors = errors or {}
    image_col, detail_col = st.columns([1, 2], gap="medium")
    with image_col:
        _show_current_crop(pill)

    with detail_col:
        st.caption(_recapture_reason(pill))
        st.caption("Take one close, in-focus photo with the imprint clearly visible.")

        if pill.instance_id in errors:
            st.error(errors[pill.instance_id])

        captured = st.camera_input(
            "Capture a close-up",
            key=f"recapture_camera_{pill.instance_id}",
        )
        uploaded = st.file_uploader(
            "Or upload a close-up",
            type=["png", "jpg", "jpeg"],
            key=f"recapture_upload_{pill.instance_id}",
        )
        chosen = captured or uploaded
        if chosen is not None:
            retake_image = Image.open(chosen).convert("RGB")
            st.image(retake_image, caption="Retake preview", width=220)
            if st.button(
                "Analyze this retake",
                key=f"recapture_analyze_{pill.instance_id}",
                type="primary",
            ):
                on_recapture(pill.instance_id, retake_image)
                st.rerun()

        st.divider()
        manual_value = st.text_input(
            "Imprint or medication name",
            value=pill.imprint_raw if pill.imprint_raw not in {"?", "-"} else "",
            key=f"recapture_manual_{pill.instance_id}",
            placeholder="Example: TV5056, 84A, Aspirin",
        )
        if st.button("Confirm manually", key=f"recapture_confirm_{pill.instance_id}"):
            if manual_value.strip():
                on_manual_override(pill.instance_id, manual_value.strip())
                st.rerun()
            else:
                st.info("Enter an imprint or medication name before confirming.")
