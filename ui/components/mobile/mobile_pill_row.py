"""Compact Touchable Mobile Pill Row Component."""

from __future__ import annotations

from html import escape
from typing import Any
import streamlit as st
from ui.adapters.view_models import PillViewModel
from ui.mobile_ui_logic import get_pill_status_content


def render_mobile_pill_row(
    pill: Any,
    display_index: int = 1,
) -> None:
    """Render a compact pill summary with technical evidence de-emphasized."""
    if isinstance(pill, PillViewModel):
        status = pill.status.lower()
        d_name = pill.drug_name or pill.brand_name or "Chưa định danh"
        shape = pill.shape or ""
        color = pill.color_primary or ""
        imprint = pill.imprint_raw or "—"
        strength = pill.strength or ""
        conf = f"{int(pill.match_confidence * 100)}%" if pill.match_confidence else "—"
        is_manual = pill.is_manual_override
    elif isinstance(pill, dict):
        status = pill.get("status", "unresolved").lower()
        d_name = pill.get("drug_name") or pill.get("brand_name") or "Chưa định danh"
        shape = pill.get("shape", "")
        color = pill.get("color_primary", "")
        imprint = pill.get("imprint_raw", "—")
        strength = pill.get("strength") or ""
        conf_val = pill.get("match_confidence")
        conf = f"{int(conf_val * 100)}%" if conf_val else "—"
        is_manual = bool(pill.get("is_manual_override", False))
    else:
        return

    status_content = get_pill_status_content(status, is_manual=is_manual)
    evidence = f'{shape} · {color} · mã in "{imprint}"'
    confidence_label = "Độ tin cậy cao" if conf != "—" and int(conf.rstrip("%")) >= 85 else "Cần kiểm tra"

    html = (
        f'<article class="mobile-pill-row {status_content.css_class}">'
        f'<div class="mobile-pill-row-top"><span class="mobile-pill-index">Viên {display_index}</span>'
        f'<span class="pill-badge {status_content.css_class}">{status_content.symbol} {escape(status_content.label)}</span></div>'
        f'<div class="mobile-pill-name">{escape(d_name)} {escape(strength)}</div>'
        f'<div class="mobile-pill-confidence">{escape(confidence_label)}</div>'
        f'<div class="mobile-pill-evidence">{escape(evidence)}</div>'
        f'</article>'
    )
    st.markdown(html, unsafe_allow_html=True)
