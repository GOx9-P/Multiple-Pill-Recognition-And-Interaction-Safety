"""Medical Legal Disclaimer Component."""

from __future__ import annotations

import streamlit as st


def render_disclaimer() -> None:
    """Render the persistent clinical and legal disclaimer."""
    st.markdown(
        """
        <div class="clinical-disclaimer">
            <span style="font-size: 1.25rem;">ℹ️</span>
            <div>
                <strong>Miễn trừ trách nhiệm y tế (Medical Disclaimer):</strong>
                Hệ thống AI này được thiết kế với mục đích hỗ trợ ra quyết định lâm sàng và tham khảo thông tin nhận diện dược phẩm. 
                Kết quả phân tích <em>không thay thế</em> chẩn đoán, chỉ định hay tư vấn trực tiếp từ bác sĩ hoặc dược sĩ chuyên môn. 
                Tuyệt đối không tự ý ngưng, thay đổi liều lượng hoặc phối hợp thuốc khi chưa có ý kiến của nhân viên y tế.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
