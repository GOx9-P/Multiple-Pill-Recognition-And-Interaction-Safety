"""Compact Touchable Mobile Pill Row Component."""

from __future__ import annotations

from typing import Any
import streamlit as st
from ui.adapters.view_models import PillViewModel


def render_mobile_pill_row(
    pill: Any,
    is_selected: bool = False,
    on_select_callback: Any = None,
) -> None:
    """Render a compact touchable mobile row for a detected pill."""
    if isinstance(pill, PillViewModel):
        inst_id = pill.instance_id
        status = pill.status.lower()
        d_name = pill.drug_name or pill.brand_name or "Chưa định danh"
        shape = pill.shape or ""
        color = pill.color_primary or ""
        imprint = pill.imprint_raw or "—"
        strength = pill.strength or ""
        conf = f"{int(pill.match_confidence * 100)}%" if pill.match_confidence else "—"
    elif isinstance(pill, dict):
        inst_id = pill.get("instance_id", "pill_001")
        status = pill.get("status", "unresolved").lower()
        d_name = pill.get("drug_name") or pill.get("brand_name") or "Chưa định danh"
        shape = pill.get("shape", "")
        color = pill.get("color_primary", "")
        imprint = pill.get("imprint_raw", "—")
        strength = pill.strength or ""
        conf_val = pill.get("match_confidence")
        conf = f"{int(conf_val * 100)}%" if conf_val else "—"
    else:
        return

    badge_cls = "accepted" if status == "accepted" else ("ambiguous" if status == "ambiguous" else "unresolved")
    active_cls = "active" if is_selected else ""
    sel_mark = "✓ Đã chọn" if is_selected else "› Xem vị trí"

    html = (
        f'<div class="mobile-pill-row {active_cls}">'
        f'<div class="mobile-pill-row-top"><span class="mobile-pill-tag">● {inst_id}</span><span class="pill-badge {badge_cls}">{status.upper()}</span></div>'
        f'<div class="mobile-pill-name">{d_name} {strength}</div>'
        f'<div class="mobile-pill-meta"><span>{shape} · {color} · "{imprint}"</span><span class="mobile-pill-conf">{conf}</span></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    btn_label = f"{sel_mark} viên {inst_id}"
    if st.button(
        btn_label,
        key=f"m_btn_pill_{inst_id}",
        use_container_width=True,
        type="primary" if is_selected else "secondary",
    ):
        if on_select_callback:
            on_select_callback(inst_id)
