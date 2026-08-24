"""Mobile Drug-Drug Interaction (DDI) Pair Checker View."""

from __future__ import annotations

import streamlit as st
from ui.adapters.pipeline_adapter import KNOWN_DDI_MATRIX
from ui.components.mobile.mobile_interaction_card import render_mobile_interaction_card

AVAILABLE_INGREDIENTS = [
    "Clopidogrel",
    "Omeprazole",
    "Aspirin",
    "Lisinopril",
    "Amiodarone",
    "Warfarin",
    "Acetaminophen",
]


def render_mobile_ddi_view() -> None:
    """Render the mobile-optimized DDI Pair Checker screen."""
    st.markdown(
        '<div class="mobile-section-header">'
        '<div class="mobile-section-title" style="font-size: 1.15rem; margin-bottom: 2px;">⚡ Kiểm Tra Tương Tác Cặp</div>'
        '<div style="font-size: 0.8rem; color: var(--text-muted);">Đối soát nhanh nguy cơ tương tác giữa 2 loại hoạt chất</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.selectbox(
            "Hoạt chất 1",
            options=AVAILABLE_INGREDIENTS,
            index=0,
            key="m_ddi_select_a",
        )
    with col2:
        drug_b = st.selectbox(
            "Hoạt chất 2",
            options=AVAILABLE_INGREDIENTS,
            index=1,
            key="m_ddi_select_b",
        )

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    pair_a = drug_a.strip().lower()
    pair_b = drug_b.strip().lower()

    if pair_a == pair_b:
        dup_html = (
            '<div class="mobile-safety-hero moderate">'
            '<div class="mobile-safety-icon">🔄</div>'
            '<div>'
            f'<div class="mobile-safety-title">CÙNG MỘT HOẠT CHẤT ({drug_a.upper()})</div>'
            '<div class="mobile-safety-desc">Nguy cơ quá liều tích lũy hoạt chất khi dùng trùng lặp trong đơn thuốc.</div>'
            '</div>'
            '</div>'
        )
        st.markdown(dup_html, unsafe_allow_html=True)
        return

    rule = KNOWN_DDI_MATRIX.get((pair_a, pair_b)) or KNOWN_DDI_MATRIX.get((pair_b, pair_a))

    if rule:
        inter_dict = {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": rule["severity"],
            "clinical_effect": rule.get("clinical_risk") or rule["message"],
            "recommendation": rule.get("management") or "Tránh dùng đồng thời.",
            "mechanism": rule.get("mechanism", ""),
        }
        render_mobile_interaction_card(inter_dict)
    else:
        safe_html = (
            f'<div class="mobile-safe-box">'
            f'🟢 <b>Không phát hiện tương tác đối kháng</b> giữa <i>{drug_a}</i> và <i>{drug_b}</i> trong cơ sở dữ liệu hiện hành.'
            f'</div>'
        )
        st.markdown(safe_html, unsafe_allow_html=True)
