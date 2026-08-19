"""Drug Database Search & Directory View."""

from __future__ import annotations

import streamlit as st

from ..adapters.pipeline_adapter import KNOWN_DRUG_DATABASE


def render_drug_search_view() -> None:
    """Render the pharmaceutical database search and directory lookup interface."""
    st.markdown(
        """
        <div class="clinical-card">
            <div class="card-header-row">
                <h3 class="card-title">📚 Cơ sở dữ liệu dược phẩm & Tra cứu thuốc</h3>
                <span style="font-size: 0.8rem; color: var(--text-muted);">RxNorm • DailyMed Directory</span>
            </div>
            <p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0;">
                Tra cứu thông tin định danh thuốc, mã khắc ký tự (Imprint), hình dáng, màu sắc và hàm lượng hoạt chất chuẩn hóa.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search Bar & Filter Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("🔍 Tìm theo tên thuốc hoặc mã khắc (Imprint):", placeholder="Ví dụ: Plavix, 84A, Aspirin, TV5056...")
    with c2:
        shape_filter = st.selectbox("Hình dáng (Shape):", ["Tất cả", "Round", "Oval", "Capsule"])
    with c3:
        color_filter = st.selectbox("Màu sắc (Color):", ["Tất cả", "White", "Yellow", "Orange"])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Filter and display results
    matches = []
    for imprint, drug in KNOWN_DRUG_DATABASE.items():
        name_match = (
            not query
            or query.lower() in drug["product_name"].lower()
            or query.lower() in (drug.get("brand_name") or "").lower()
            or query.lower() in imprint.lower()
        )
        if name_match:
            matches.append((imprint, drug))

    st.markdown(f"**Kết quả tra cứu ({len(matches)} loại thuốc):**")

    if not matches:
        st.warning("Không tìm thấy loại thuốc nào phù hợp với từ khóa tìm kiếm.")
        return

    cols = st.columns(2)
    for idx, (imprint, drug) in enumerate(matches):
        col = cols[idx % 2]
        with col:
            st.markdown(
                f"""
                <div class="pill-card-item">
                    <div class="pill-card-top">
                        <span class="pill-instance-tag">IMPRINT: <code>{imprint}</code></span>
                        <span class="pill-badge accepted">RXNORM</span>
                    </div>
                    <h4 class="pill-drug-name">{drug['product_name']}</h4>
                    <div class="pill-meta-row"><span>Biệt dược (Brand):</span><span>{drug.get('brand_name') or 'N/A'}</span></div>
                    <div class="pill-meta-row"><span>Tên gốc (Generic):</span><span>{drug.get('generic_name') or 'N/A'}</span></div>
                    <div class="pill-meta-row"><span>Hàm lượng (Strength):</span><span>{drug.get('strength') or 'N/A'}</span></div>
                    <div class="pill-meta-row"><span>Mã RxCUI:</span><span><code>{drug.get('rxcui') or 'N/A'}</code></span></div>
                    <div class="pill-meta-row"><span>Mã NDC:</span><span><code>{drug.get('ndc') or 'N/A'}</code></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
