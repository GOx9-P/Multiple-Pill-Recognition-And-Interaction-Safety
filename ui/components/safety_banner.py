"""Overall Clinical Safety Banner Component."""

from __future__ import annotations

import streamlit as st

from ..adapters.view_models import SafetyReportViewModel


def render_safety_banner(report: SafetyReportViewModel) -> None:
    """Render the high-priority clinical severity banner based on the evaluation result."""
    severity = report.overall_severity.lower()

    if severity == "critical":
        title = "NGUY HIỂM CAO — PHÁT HIỆN TƯƠNG TÁC THUỐC CHỐNG CHỈ ĐỊNH"
        desc = "Đơn thuốc có chứa cặp tương tác đối kháng nghiêm trọng. Khuyến cáo kiểm tra chuyên môn lâm sàng ngay lập tức trước khi cấp phát hoặc sử dụng."
        icon = "⛔"
        css_class = "critical"
    elif severity == "moderate":
        title = "CẢNH BÁO — PHÁT HIỆN TƯƠNG TÁC THUỐC HOẶC TRÙNG LẶP HOẠT CHẤT"
        desc = "Phát hiện tương tác mức độ trung bình hoặc trùng lặp hoạt chất cần theo dõi liều lượng và giãn cách thời gian uống phù hợp."
        icon = "⚠️"
        css_class = "moderate"
    elif severity == "unresolved":
        title = "CHÚ Ý — CÓ VIÊN THUỐC CHƯA THỂ ĐỊNH DANH CHẮC CHẮN"
        desc = "Một số viên thuốc có ảnh mờ hoặc chữ khắc không rõ ràng. Vui lòng sử dụng tính năng 'Nhập bù thủ công' để hoàn tất phân tích an toàn."
        icon = "❓"
        css_class = "unresolved"
    else:
        title = "AN TOÀN — KHÔNG PHÁT HIỆN TƯƠNG TÁC BẤT LỢI TRONG CSDL"
        desc = "Không ghi nhận tương tác đối kháng nguy hiểm nào giữa các thuốc được nhận diện trong cơ sở dữ liệu dược thư hiện hành."
        icon = " "
        css_class = "safe"

    st.markdown(
        f"""
        <div class="severity-banner {css_class}">
            <div class="severity-icon-badge">{icon}</div>
            <div>
                <h3 class="severity-title">{title}</h3>
                <p class="severity-description">{desc}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
