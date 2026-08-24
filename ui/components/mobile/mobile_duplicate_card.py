"""Consumer-facing duplicate-active-ingredient warning."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def render_mobile_duplicate_card(duplicate: Any) -> None:
    """Render an overdose warning when one ingredient appears in several pills."""
    ingredient = escape(getattr(duplicate, "ingredient_name", "Hoạt chất"))
    warning = escape(getattr(duplicate, "warning", "Phát hiện hoạt chất trùng lặp."))
    count = len(getattr(duplicate, "source_instances", []))
    st.markdown(
        f'<div class="mobile-duplicate-card"><div class="mobile-ddi-badge duplicate">Trùng hoạt chất</div>'
        f'<div class="mobile-ddi-pair">{ingredient} xuất hiện trong {count} viên</div>'
        f'<div class="mobile-ddi-action"><strong>Bạn cần làm gì</strong>'
        f'Kiểm tra tổng liều với bác sĩ hoặc dược sĩ trước khi dùng.</div>'
        f'<div class="mobile-ddi-section"><div class="mobile-ddi-label">Nguy cơ</div>'
        f'<div class="mobile-ddi-text">{warning}</div></div></div>',
        unsafe_allow_html=True,
    )
