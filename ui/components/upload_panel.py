"""Input Zone: Upload, Camera, and Sleek Preset Scenarios."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st
from PIL import Image

from ..demo_data import get_preset_scenario


def render_upload_panel(
    on_image_selected: Callable[[Image.Image, str, dict[str, Any] | None], None]
) -> None:
    """Render file upload, camera capture, and sleek demo presets."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">📷 Tải ảnh đơn thuốc & Kịch bản mẫu</h3>
                <span style="font-size: 0.8rem; color: var(--text-muted);">PNG, JPG, JPEG • 1-Click Fast Demo</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Sleek Demo Scenario Presets Bar
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        if st.button(
            "🔴 Critical\nClopidogrel + Omeprazole",
            key="btn_preset_crit",
            use_container_width=True,
            help="Xung đột đối kháng ức chế CYP2C19 nguy hiểm cao",
        ):
            data = get_preset_scenario("critical")
            img = Image.new("RGB", (800, 500), color=(241, 245, 249))
            on_image_selected(img, "scenario_critical.json", data)
            st.rerun()

    with p2:
        if st.button(
            "🟡 Moderate\nAspirin + Lisinopril",
            key="btn_preset_mod",
            use_container_width=True,
            help="Tương tác trung bình ảnh hưởng huyết áp & thận",
        ):
            data = get_preset_scenario("moderate")
            img = Image.new("RGB", (800, 500), color=(241, 245, 249))
            on_image_selected(img, "scenario_moderate.json", data)
            st.rerun()

    with p3:
        if st.button(
            "❓ Unresolved\nẢnh mờ / Cần nhập tay",
            key="btn_preset_unres",
            use_container_width=True,
            help="Viên thuốc bị mờ kích hoạt Human-in-the-loop",
        ):
            data = get_preset_scenario("unresolved")
            img = Image.new("RGB", (800, 500), color=(241, 245, 249))
            on_image_selected(img, "scenario_unresolved.json", data)
            st.rerun()

    with p4:
        if st.button(
            "🟢 Safe\nĐơn thuốc tương thích",
            key="btn_preset_safe",
            use_container_width=True,
            help="Đơn thuốc an toàn không có tương tác đối kháng",
        ):
            data = get_preset_scenario("safe")
            img = Image.new("RGB", (800, 500), color=(241, 245, 249))
            on_image_selected(img, "scenario_safe.json", data)
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. File Uploader and Camera Tabs
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
                if st.button("🚀 Bắt Đầu Phân Tích Ảnh", type="primary", key="btn_run_uploaded"):
                    on_image_selected(img, uploaded_file.name, None)
                    st.rerun()
            except Exception as e:
                st.error(f"Không thể đọc file ảnh: {e}")

    with tab_camera:
        camera_file = st.camera_input("Chụp ảnh trực tiếp viên thuốc:", key="main_camera_input")
        if camera_file is not None:
            try:
                img = Image.open(camera_file).convert("RGB")
                if st.button("🚀 Phân Tích Ảnh Vừa Chụp", type="primary", key="btn_run_camera"):
                    on_image_selected(img, "camera_capture.jpg", None)
                    st.rerun()
            except Exception as e:
                st.error(f"Không thể xử lý ảnh camera: {e}")
