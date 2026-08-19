"""Mobile Architecture & Methodology View."""

from __future__ import annotations

import textwrap
import streamlit as st


def render_mobile_about_view() -> None:
    """Render the mobile-optimized Methodology & System Architecture screen."""
    header_html = textwrap.dedent(
        """
        <div class="mobile-section-header">
            <div class="mobile-section-title" style="font-size: 1.15rem; margin-bottom: 2px;">📖 Kiến Trúc & Phương Pháp</div>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Hệ thống AI Đa tầng hỗ trợ nhận diện và kiểm tra an toàn tương tác thuốc</div>
        </div>
        """
    ).strip()
    st.markdown(header_html, unsafe_allow_html=True)

    modules = [
        {
            "num": "01",
            "title": "Phát hiện & Phân đoạn (Instance Segmentation)",
            "model": "YOLOv11-Seg",
            "desc": "Tách từng viên thuốc khỏi nền ảnh phức tạp, tạo bounding box và mặt nạ phân đoạn chính xác theo pixel.",
        },
        {
            "num": "02",
            "title": "Nhận diện Thuộc tính (Attribute Classification)",
            "model": "ResNet18 Multi-Head",
            "desc": "Dự đoán đồng thời 4 đặc trưng quang học: Hình dạng (Shape), Màu sắc (Color), Vạch chia (Scoreline), Khắc chìm (Imprint).",
        },
        {
            "num": "03",
            "title": "Trích xuất Ký tự Khắc chìm (OCR)",
            "model": "PaddleOCR v3 (Fine-tuned)",
            "desc": "Trích xuất ký tự chữ và số dập nổi trên bề mặt viên thuốc, lọc nhiễu theo độ tin cậy.",
        },
        {
            "num": "04",
            "title": "Truy xuất Dược thư (Knowledge Retrieval / RAG)",
            "model": "RxNorm & DailyMed",
            "desc": "Đối soát đa trường (Multi-field) với cơ sở dữ liệu thuốc chuẩn FDA để xác định mã RxCUI và tên thuốc chính thức.",
        },
        {
            "num": "05",
            "title": "Động cơ Cảnh báo Tương tác (DDI Engine)",
            "model": "Clinical DDI Matrix",
            "desc": "Kiểm tra tương tác đối kháng, ức chế enzyme gan CYP450 và phát hiện trùng lặp hoạt chất gây quá liều tích lũy.",
        },
    ]

    for m in modules:
        with st.expander(f"🔹 Mô-đun {m['num']}: {m['title']}", expanded=False):
            mod_html = textwrap.dedent(
                f"""
                <div style="font-size: 0.825rem; line-height: 1.5; color: var(--text-secondary);">
                    <div><b>Model:</b> <code style="color: var(--accent-brand);">{m['model']}</code></div>
                    <div style="margin-top: 4px;">{m['desc']}</div>
                </div>
                """
            ).strip()
            st.markdown(mod_html, unsafe_allow_html=True)

    disc_html = textwrap.dedent(
        """
        <div class="mobile-guidance-box" style="margin-top: 16px; border-left: 3px solid var(--sev-critical);">
            <div style="font-weight: 700; color: var(--sev-critical-text); margin-bottom: 4px;">⚖️ Tuyên Bố Miễn Trừ Trách Nhiệm Y Tế</div>
            <div style="font-size: 0.775rem; line-height: 1.45; color: var(--text-muted);">
                Hệ thống chỉ phục vụ mục đích nghiên cứu học thuật và hỗ trợ ra quyết định lâm sàng (Clinical Decision Support). Kết quả từ AI không thay thế cho chỉ định hoặc tư vấn y khoa của bác sĩ/dược sĩ chuyên môn.
            </div>
        </div>
        """
    ).strip()
    st.markdown(disc_html, unsafe_allow_html=True)
