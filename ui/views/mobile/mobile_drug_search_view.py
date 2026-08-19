"""Mobile Drug Database Search View."""

from __future__ import annotations

import textwrap
import streamlit as st
from ui.adapters.pipeline_adapter import KNOWN_DRUG_DATABASE


def render_mobile_drug_search_view() -> None:
    """Render the mobile-optimized Drug Directory & Imprint Search screen."""
    header_html = textwrap.dedent(
        """
        <div class="mobile-section-header">
            <div class="mobile-section-title" style="font-size: 1.15rem; margin-bottom: 2px;">📚 Tra Cứu Dược Thư</div>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Tra cứu thông tin hoạt chất, chỉ định và hình dạng nhận diện thuốc</div>
        </div>
        """
    ).strip()
    st.markdown(header_html, unsafe_allow_html=True)

    query = st.text_input(
        "Tìm kiếm thuốc",
        placeholder="Nhập tên thuốc hoặc mã khắc (84A, TV5056...)",
        key="m_drug_search_query",
        label_visibility="collapsed",
    )

    matches = []
    for imprint, drug in KNOWN_DRUG_DATABASE.items():
        name_match = (
            not query
            or query.lower() in drug["product_name"].lower()
            or query.lower() in (drug.get("brand_name") or "").lower()
            or query.lower() in (drug.get("generic_name") or "").lower()
            or query.lower() in imprint.lower()
        )
        if name_match:
            matches.append((imprint, drug))

    st.markdown(f"<div style='font-size: 0.8rem; font-weight: 600; color: var(--text-muted); margin: 8px 0;'>Tìm thấy {len(matches)} loại thuốc phù hợp:</div>", unsafe_allow_html=True)

    for imprint, drug in matches:
        brand = drug.get("brand_name") or "—"
        rxcui = drug.get("rxcui") or "—"
        strength = drug.get("strength") or ""
        card_html = textwrap.dedent(
            f"""
            <div class="mobile-pill-row" style="margin-bottom: 8px;">
                <div class="mobile-pill-name" style="font-size: 0.95rem;">{drug['product_name']}</div>
                <div style="font-size: 0.8rem; color: var(--accent-brand); font-weight: 600;">Biệt dược: {brand} · RxCUI: {rxcui}</div>
                <div class="mobile-pill-meta" style="margin-top: 4px;">
                    <span>Mã khắc: "{imprint}" · Hàm lượng: {strength}</span>
                </div>
            </div>
            """
        ).strip()
        st.markdown(card_html, unsafe_allow_html=True)
