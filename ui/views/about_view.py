"""About and Methodology View."""

from __future__ import annotations

import streamlit as st


def render_about_view() -> None:
    """Render the methodology, deep learning architecture, and data source disclosures."""
    st.markdown(
        """
        <div class="clinical-card">
            <div class="card-header-row">
                <h3 class="card-title">📖 Kiến trúc hệ thống & Phương pháp nghiên cứu</h3>
                <span style="font-size: 0.8rem; color: var(--text-muted);">Deep Learning + RAG + Clinical Decision Support</span>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                Hệ thống kết hợp thị giác máy tính sâu (Deep Computer Vision), Truy xuất tri thức dược phẩm (RAG) và Động cơ đánh giá tương tác thuốc:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Pipeline Stages Cards Grid
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">Module 1: Phân đoạn đối tượng (YOLOv11-Seg)</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Tách từng thực thể viên thuốc trong ảnh chụp phức tạp, xuất bounding box [x1, y1, x2, y2] và segmentation mask chính xác.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">Module 2: Đa nhiệm thuộc tính (ResNet18 Multi-Head)</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Trích xuất đặc trưng ngoại quan gồm Hình dạng (Round, Oval, Capsule...) và Màu sắc (White, Yellow, Orange...) kèm độ tin cậy.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">Module 3: OCR & Vạch chia liều (PaddleOCR + Hough)</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Nhận dạng ký tự khắc dập (Imprint) trên mặt thuốc và phát hiện vạch chia liều (Scoreline).
</p>
</div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">Module 4: Hợp nhất thị giác (CV Fusion)</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Hợp nhất dữ liệu các nhánh CV thành JSON chuẩn hóa biểu diễn toàn diện từng viên thuốc.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">Module 5: RAG Candidate Retrieval & Decision Gate</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Đối soát CSDL dược thư RxNorm theo trọng số IDF Imprint + Shape + Color. Phân loại Accepted, Ambiguous hoặc Unresolved.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">Module 6 & 7: DDI Engine & Clinical Report</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Tra cứu ma trận tương tác thuốc (DDI), phát hiện trùng lặp hoạt chất và sinh báo cáo lâm sàng chuẩn y khoa.
</p>
</div>""",
            unsafe_allow_html=True,
        )

    # 2. Data Sources & Clinical Ethics
    st.markdown(
        """<div class="clinical-card" style="margin-top: 0.75rem;">
<h4 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-size: 1rem;">2. Nguồn dữ liệu & Cơ sở Tri thức</h4>
<ul style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin: 0 0 1rem 1.25rem; padding: 0;">
<li><strong>RxNorm / NLM:</strong> Chuẩn hóa mã định danh thuốc (RxCUI) và danh mục hoạt chất quốc tế.</li>
<li><strong>FDA DailyMed:</strong> Thông tin nhãn thuốc, mã phân loại NDC và đặc điểm hình dạng/màu sắc.</li>
<li><strong>NLM Drug Interaction Database:</strong> Dữ liệu tương tác thuốc được thẩm định lâm sàng theo các mức độ nghiêm trọng.</li>
</ul>

<h4 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-size: 1rem;">3. Quy chuẩn An toàn Lâm sàng & Human-in-the-Loop</h4>
<p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; margin: 0;">
Nhằm ngăn chặn tối đa sai sót y tế do ảnh mờ hoặc chữ khắc bị che khuất, hệ thống triển khai cổng quyết định an toàn (Decision Gate). 
Khi độ tin cậy không đạt ngưỡng an toàn, hệ thống lập tức kích hoạt trạng thái <code>unresolved</code> và mở form nhập bổ sung thủ công 
để nhân viên y tế / người dùng xác nhận trước khi tiếp tục phân tích tương tác.
</p>
</div>""",
        unsafe_allow_html=True,
    )
