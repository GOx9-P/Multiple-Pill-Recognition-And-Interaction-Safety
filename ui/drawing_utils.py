"""Image rendering and bounding box overlay helpers for Clinical AI visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def draw_cv_overlay(
    image: Image.Image,
    pills: Any,
    selected_pill_id: str | None = None,
) -> Image.Image:
    """Return an annotated RGB image with bounding boxes, pill numbers, and selected highlight.
    
    Accepts either a list of pills or an object with a .pills attribute.
    """
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pill_list = getattr(pills, "pills", pills)
    if not isinstance(pill_list, (list, tuple)):
        pill_list = []

    for idx, pill in enumerate(pill_list, start=1):
        # Extract bbox coordinates
        if hasattr(pill, "bbox_xyxy"):
            coords = pill.bbox_xyxy
            inst_id = getattr(pill, "instance_id", f"pill_{idx:03d}")
            mask_path_str = getattr(pill, "mask_path", None)
        elif isinstance(pill, dict):
            coords = pill.get("bbox_xyxy", [0, 0, 100, 100])
            inst_id = pill.get("instance_id", f"pill_{idx:03d}")
            mask_path_str = pill.get("mask_path")
        else:
            coords = [0, 0, 100, 100]
            inst_id = f"pill_{idx:03d}"
            mask_path_str = None

        if len(coords) < 4:
            continue

        left, top, right, bottom = (int(val) for val in coords)
        is_selected = selected_pill_id is not None and inst_id == selected_pill_id

        # Colors: Selected -> Bright Cyan/Teal; Normal -> Coral Red / Amber
        if is_selected:
            box_color = (6, 182, 212, 255)  # Cyan
            fill_color = (6, 182, 212, 45)
            line_width = 4
        else:
            box_color = (239, 68, 68, 230)  # Red / Coral
            fill_color = (239, 68, 68, 20)
            line_width = 2

        # Draw semi-transparent box fill
        draw.rectangle((left, top, right, bottom), fill=fill_color, outline=box_color, width=line_width)

        # Draw optional segmentation mask if present
        if mask_path_str and Path(mask_path_str).is_file() and right > left and bottom > top:
            try:
                with Image.open(mask_path_str) as src_mask:
                    mask = src_mask.convert("L").resize(
                        (right - left, bottom - top), Image.Resampling.NEAREST
                    )
                mask_tint = Image.new("RGBA", mask.size, box_color[:3] + (70,))
                overlay.alpha_composite(mask_tint, (left, top))
            except Exception:
                pass

        # Draw Badge Tag with Pill Number [1], [2]
        tag_text = f" #{idx} "
        tag_w = len(tag_text) * 9 + 8
        tag_h = 20
        tag_top = max(0, top - tag_h - 2)
        draw.rectangle((left, tag_top, left + tag_w, tag_top + tag_h), fill=box_color)
        draw.text((left + 4, tag_top + 2), tag_text, fill=(255, 255, 255, 255))

    return Image.alpha_composite(canvas, overlay).convert("RGB")
