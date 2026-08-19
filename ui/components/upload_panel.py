"""Input Zone: Real Image Upload and Camera Capture."""

from __future__ import annotations

from typing import Callable

import streamlit as st
from PIL import Image


def render_upload_panel(
    on_image_selected: Callable[[Image.Image, str], None]
) -> None:
    """Render file upload and camera capture without mock preset buttons."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">📷 Tải ảnh chụp viên thuốc thực tế để nhận diện</h3>
                <span style="font-size: 0.8rem; color: var(--text-muted);">Định dạng hỗ trợ: PNG, JPG, JPEG • Tự động nhận diện qua YOLOv11 + ResNet18 + OCR</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File Uploader and Camera Tabs
    tab_upload, tab_camera = st.tabs(["📁 Tải tệp từ máy tính", "📸 Chụp từ Camera"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Chọn ảnh chụp chứa các viên thuốc:",
            type=["png", "jpg", "jpeg"],
            key="main_file_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(img, caption="Ảnh đã chọn", width=180)
                with c2:
                    st.markdown(f"**Tên file:** `{uploaded_file.name}` ({img.width}x{img.height} px)")
                    if st.button("🚀 Bắt Đầu Phân Tích Bằng AI", type="primary", key="btn_run_uploaded"):
                        on_image_selected(img, uploaded_file.name)
                        st.rerun()
            except Exception as e:
                st.error(f"Không thể đọc file ảnh: {e}")

    with tab_camera:
        camera_file = st.camera_input("Chụp ảnh trực tiếp viên thuốc:", key="main_camera_input")
        if camera_file is not None:
            try:
                img = Image.open(camera_file).convert("RGB")
                if st.button("🚀 Phân Tích Ảnh Vừa Chụp Bằng AI", type="primary", key="btn_run_camera"):
                    on_image_selected(img, "camera_capture.jpg")
                    st.rerun()
            except Exception as e:
                st.error(f"Không thể xử lý ảnh camera: {e}")

