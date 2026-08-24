"""Mobile Analyze View — Zalo-Inspired Media Entry & Camera-First Clinical Flow.

Implements Zalo Interaction Pattern:
1. Single Media Entry Area (Tap to open Media Chooser)
2. Media Chooser Surface (Camera first, File Uploader second, Recent Session Grid 3-6)
3. Photo Review State with single prominent Primary CTA
4. Clinical Decision Hierarchy (Safety Banner -> Detected Pills -> DDI Alerts -> Recommendations -> Visual Evidence)
"""

from __future__ import annotations

import datetime
from typing import Any
from PIL import Image
import streamlit as st

from ui.adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output
from ui.components.mobile.mobile_pill_row import render_mobile_pill_row
from ui.components.mobile.mobile_interaction_card import render_mobile_interaction_card
from ui.drawing_utils import draw_cv_overlay


def _add_to_recent_images(image: Image.Image, name: str, preset_data: dict[str, Any] | None = None) -> None:
    """Store up to 6 recent session images with thumbnails for quick Zalo-style chooser grid."""
    st.session_state.setdefault("recent_images", [])
    thumb = image.copy()
    thumb.thumbnail((120, 120))
    now_str = datetime.datetime.now().strftime("%H:%M")
    
    filtered = [item for item in st.session_state.recent_images if item.get("name") != name]
    entry = {
        "image": image,
        "name": name,
        "thumb": thumb,
        "preset_data": preset_data,
        "time": now_str,
    }
    st.session_state.recent_images = [entry] + filtered[:5]


def render_mobile_analyze_view(cv_load_result: Any) -> None:
    """Render the Zalo-inspired Camera-First Mobile Analyze View."""
    st.session_state.setdefault("current_image", None)
    st.session_state.setdefault("current_image_name", None)
    st.session_state.setdefault("raw_cv_data", None)
    st.session_state.setdefault("pending_image", None)
    st.session_state.setdefault("pending_image_name", None)
    st.session_state.setdefault("pending_cv_data", None)
    st.session_state.setdefault("selected_pill_id", None)
    st.session_state.setdefault("manual_overrides", {})
    st.session_state.setdefault("recent_images", [])
    st.session_state.setdefault("chooser_open", False)
    st.session_state.setdefault("mobile_input_mode", "entry")  # "entry" | "chooser" | "camera" | "uploader"

    def execute_analysis(image: Image.Image, image_name: str) -> None:
        st.session_state.current_image = image
        st.session_state.current_image_name = image_name
        st.session_state.pending_image = None
        st.session_state.pending_image_name = None
        st.session_state.pending_cv_data = None
        st.session_state.manual_overrides = {}
        st.session_state.selected_pill_id = None
        st.session_state.chooser_open = False
        st.session_state.mobile_input_mode = "entry"
        st.session_state.cv_error = None

        if cv_load_result and cv_load_result.available:
            try:
                import tempfile
                from pathlib import Path
                from uuid import uuid4
                from pill_safety.schemas import SegmentationInferenceRequest

                temp_dir = Path(tempfile.gettempdir()) / "pill_safety_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file = temp_dir / f"upload_{uuid4().hex[:8]}.png"
                image.save(temp_file)

                req = SegmentationInferenceRequest(
                    request_id=str(uuid4()),
                    session_id=str(uuid4()),
                    image_id=temp_file.stem,
                    image_path=str(temp_file),
                )
                with st.spinner("🤖 AI đang nhận diện thuốc (YOLOv11 + ResNet18 + OCR)..."):
                    artifacts = cv_load_result.pipeline.predict_with_artifacts(req)
                    st.session_state.raw_cv_data = artifacts.output
                    st.session_state.cv_error = None
            except Exception as err:
                import traceback
                st.session_state.raw_cv_data = None
                st.session_state.cv_error = f"Lỗi AI CV: {err}\n{traceback.format_exc()}"
        else:
            st.session_state.raw_cv_data = None
            st.session_state.cv_error = cv_load_result.error if cv_load_result else "Mô hình Computer Vision chưa được nạp thành công."

    current_image = st.session_state.get("current_image", None)
    pending_image = st.session_state.get("pending_image", None)
    input_mode = st.session_state.get("mobile_input_mode", "entry")

    # =========================================================================
    # STATE 1: IDLE / MEDIA ENTRY AREA & ZALO MEDIA CHOOSER
    # =========================================================================
    if current_image is None and pending_image is None:
        st.markdown(
            '<div class="mobile-section-header">'
            '<div class="mobile-section-title" style="font-size: 1.15rem; margin-bottom: 2px;">Kiểm Tra An Toàn Thuốc</div>'
            '<div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.45;">Chụp hoặc tải ảnh để kiểm tra tương tác thuốc tức thì.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # On-Demand Mode: Camera Stream
        if input_mode == "camera":
            st.markdown("<div class='mobile-section-title'>📷 Chụp Ảnh Thuốc</div>", unsafe_allow_html=True)
            cam_img = st.camera_input("Chụp ảnh", key="m_active_cam", label_visibility="collapsed")
            if st.button("✕ Huỷ chụp", key="m_btn_cancel_cam", use_container_width=True):
                st.session_state.mobile_input_mode = "entry"
                st.rerun()

            if cam_img is not None:
                pil_img = Image.open(cam_img).convert("RGB")
                st.session_state.pending_image = pil_img
                st.session_state.pending_image_name = "camera_photo.jpg"
                st.session_state.pending_cv_data = None
                _add_to_recent_images(pil_img, "Chụp Camera")
                st.session_state.mobile_input_mode = "entry"
                st.session_state.chooser_open = False
                st.rerun()

        # On-Demand Mode: File Uploader
        elif input_mode == "uploader":
            st.markdown("<div class='mobile-section-title'>📁 Chọn Ảnh Từ Thư Viện</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Tải ảnh từ máy",
                type=["jpg", "jpeg", "png", "webp"],
                key="m_active_up",
                label_visibility="collapsed",
            )
            if st.button("✕ Huỷ chọn", key="m_btn_cancel_up", use_container_width=True):
                st.session_state.mobile_input_mode = "entry"
                st.rerun()

            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file).convert("RGB")
                st.session_state.pending_image = pil_img
                st.session_state.pending_image_name = uploaded_file.name
                st.session_state.pending_cv_data = None
                _add_to_recent_images(pil_img, uploaded_file.name)
                st.session_state.mobile_input_mode = "entry"
                st.session_state.chooser_open = False
                st.rerun()

        # Default State: SINGLE Media Entry Hub (Zalo Pattern)
        else:
            # Media Entry Point Button / Card
            if st.button("🖼️  Thêm Ảnh Thuốc (Chạm để chọn hoặc chụp)", key="m_btn_open_chooser", type="primary", use_container_width=True):
                st.session_state.chooser_open = not st.session_state.get("chooser_open", False)
                st.rerun()

            # Zalo-Inspired Media Chooser Surface (Toggled open)
            if st.session_state.get("chooser_open", False):
                with st.container():
                    st.markdown("<div style='font-weight: 700; font-size: 0.875rem; margin-bottom: 6px;'>Chọn nguồn ảnh:</div>", unsafe_allow_html=True)
                    
                    c_cam, c_lib = st.columns(2)
                    with c_cam:
                        if st.button("📷 Chụp Ảnh", key="m_btn_choose_cam", use_container_width=True):
                            st.session_state.mobile_input_mode = "camera"
                            st.rerun()
                    with c_lib:
                        if st.button("📁 Thư Viện Ảnh", key="m_btn_choose_lib", use_container_width=True):
                            st.session_state.mobile_input_mode = "uploader"
                            st.rerun()

                    # Recent Session Images Shelf (Cells 3-6)
                    recent_list = st.session_state.get("recent_images", [])
                    if recent_list:
                        st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: var(--text-muted); margin: 8px 0 4px 0;'>🕒 Ảnh gần đây trong phiên:</div>", unsafe_allow_html=True)
                        cols = st.columns(min(len(recent_list), 4))
                        for idx, r_item in enumerate(recent_list[:4]):
                            with cols[idx]:
                                st.image(r_item["thumb"], use_container_width=True)
                                if st.button(f"Chọn #{idx+1}", key=f"m_recent_sel_{idx}", use_container_width=True):
                                    st.session_state.pending_image = r_item["image"]
                                    st.session_state.pending_image_name = r_item["name"]
                                    st.session_state.pending_cv_data = r_item.get("preset_data")
                                    st.session_state.chooser_open = False
                                    st.rerun()

                    if st.button("✕ Đóng bảng chọn", key="m_btn_close_chooser", use_container_width=True):
                        st.session_state.chooser_open = False
                        st.rerun()

            # Compact Optical Guidance Note
            st.markdown(
                '<div class="mobile-guidance-box">'
                '<b>💡 Mẹo chụp đạt độ chính xác cao:</b> Nền sáng đơn sắc · Đủ ánh sáng · Tách rời các viên thuốc.'
                '</div>',
                unsafe_allow_html=True,
            )

    # =========================================================================
    # STATE 2: PHOTO REVIEW STATE (Single Prominent Primary CTA)
    # =========================================================================
    elif pending_image is not None and current_image is None:
        st.markdown(
            '<div class="mobile-section-header">'
            '<div class="mobile-section-title" style="font-size: 1.1rem; margin-bottom: 2px;">🔍 Xem Lại Ảnh Đã Chọn</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.image(pending_image, use_container_width=True)

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        
        # ONE Prominent Primary CTA
        if st.button("🚀 Phân Tích Đơn Thuốc Ngay", key="m_btn_run_analysis", type="primary", use_container_width=True):
            execute_analysis(
                image=st.session_state.pending_image,
                image_name=st.session_state.pending_image_name or "pill_photo.jpg",
            )
            st.rerun()

        if st.button("🔄 Chọn Ảnh Khác", key="m_btn_change_photo", use_container_width=True):
            st.session_state.pending_image = None
            st.session_state.pending_image_name = None
            st.session_state.pending_cv_data = None
            st.session_state.mobile_input_mode = "entry"
            st.rerun()

    # =========================================================================
    # STATE 3: RESULTS WORKSPACE (After AI Analysis)
    # =========================================================================
    else:
        if st.session_state.get("cv_error"):
            st.error(f"❌ **Lỗi AI Computer Vision:**\n\n```\n{st.session_state.cv_error}\n```")
            if st.button("🔄 Thử Lại Với Ảnh Khác", key="m_btn_retry_after_err", use_container_width=True):
                st.session_state.current_image = None
                st.session_state.raw_cv_data = None
                st.session_state.cv_error = None
                st.rerun()
            return

        pills, quality = parse_cv_output(st.session_state.raw_cv_data)

        if not pills:
            st.warning("⚠️ Không phát hiện viên thuốc nào trong ảnh chụp. Vui lòng chụp rõ nét hơn.")
            if st.button("📸 Chụp Lại", key="m_btn_retake_empty", use_container_width=True):
                st.session_state.current_image = None
                st.session_state.raw_cv_data = None
                st.rerun()
            return

        report = evaluate_safety_and_report(
            pills=pills,
            manual_overrides=st.session_state.manual_overrides,
        )

        # 1. Overall Safety Hero Banner
        sev = report.overall_severity.lower()
        banner_cls = "critical" if sev == "critical" else ("moderate" if sev == "moderate" else ("unresolved" if sev == "unresolved" else "safe"))
        
        severity_titles = {
            "critical": "CẢNH BÁO NGUY HIỂM CAO (CRITICAL)",
            "moderate": "CẦN THEO DÕI ĐẶC BIỆT (MODERATE)",
            "unresolved": "CẦN BỔ SUNG THÔNG TIN THUỐC",
            "safe": "ĐƠN THUỐC TƯƠNG THÍCH AN TOÀN",
        }
        severity_descriptions = {
            "critical": "Phát hiện tương tác đối kháng nghiêm trọng. Khuyến cáo không sử dụng đồng thời.",
            "moderate": "Phát hiện tương tác trung bình hoặc trùng lặp hoạt chất cần lưu ý.",
            "unresolved": "Một số viên thuốc chưa đủ độ tin cậy để nhận diện. Vui lòng xác nhận thủ công.",
            "safe": "Không ghi nhận tương tác bất lợi giữa các thuốc trong cơ sở dữ liệu.",
        }

        st.markdown(
            f'<div class="mobile-safety-hero {banner_cls}">'
            f'<div class="mobile-safety-icon">{"🔴" if banner_cls == "critical" else ("🟡" if banner_cls == "moderate" else ("❓" if banner_cls == "unresolved" else "🟢"))}</div>'
            f'<div><div class="mobile-safety-title">{severity_titles.get(sev, sev.upper())}</div>'
            f'<div class="mobile-safety-desc">{severity_descriptions.get(sev, "")}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 2. Detected Pills List (Touchable Rows)
        st.markdown(
            f"<div class='mobile-section-title'>💊 Danh Sách Thuốc Nhận Diện ({len(pills)})</div>",
            unsafe_allow_html=True,
        )

        selected_id = st.session_state.get("selected_pill_id")

        def handle_select_pill(inst_id: str) -> None:
            st.session_state.selected_pill_id = inst_id
            st.rerun()

        for pill in pills:
            is_active = (selected_id == pill.instance_id)
            render_mobile_pill_row(pill, is_selected=is_active, on_select_callback=handle_select_pill)

        # 3. Unresolved Human-in-the-loop Form
        unresolved_pills = [p for p in pills if p.status in ("unresolved", "ambiguous")]
        if unresolved_pills:
            with st.expander("❓ Xác Nhận Thuốc Thủ Công", expanded=True):
                st.markdown("<div style='font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 6px;'>Nhập tên thuốc cho các viên chưa rõ:</div>", unsafe_allow_html=True)
                for p in unresolved_pills:
                    curr_val = st.session_state.manual_overrides.get(p.instance_id, "")
                    new_val = st.text_input(
                        f"Thuốc cho {p.instance_id}",
                        value=curr_val,
                        key=f"m_override_{p.instance_id}",
                        placeholder="Ví dụ: Plavix, Aspirin...",
                    )
                    if new_val != curr_val:
                        st.session_state.manual_overrides[p.instance_id] = new_val

                if st.button("✓ Cập nhật & Kiểm tra lại", key="m_btn_apply_override", type="primary", use_container_width=True):
                    st.rerun()

        # 4. Drug Interactions (DDI)
        if report.interactions:
            st.markdown(
                f"<div class='mobile-section-title' style='color: var(--sev-critical-text); margin-top: 6px;'>⚡ Cảnh Báo Tương Tác ({len(report.interactions)})</div>",
                unsafe_allow_html=True,
            )
            for idx, inter in enumerate(report.interactions, start=1):
                render_mobile_interaction_card(inter, index=idx)
        else:
            st.markdown(
                '<div class="mobile-safe-box">'
                '🟢 <b>Không phát hiện tương tác đối kháng nguy hiểm</b> giữa các thuốc.'
                '</div>',
                unsafe_allow_html=True,
            )

        # 5. Clinical Summary & Full Report Expander
        report_text = getattr(report, "formatted_report_text", "")
        if report_text:
            with st.expander("📋 Báo Cáo Lâm Sàng Chi Tiết", expanded=False):
                st.markdown(
                    f"<div class='mobile-report-text'>{report_text}</div>",
                    unsafe_allow_html=True,
                )

        # 6. Visual Evidence Canvas
        with st.expander("📷 Bằng Chứng Thị Giác AI", expanded=False):
            annotated_img = draw_cv_overlay(
                current_image,
                pills,
                selected_pill_id=selected_id,
            )
            st.image(annotated_img, use_container_width=True)

        # 7. Action CTA: Reset & New Scan
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Chụp Hoặc Phân Tích Đơn Thuốc Khác", key="m_btn_reset_scan", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.current_image_name = None
            st.session_state.raw_cv_data = None
            st.session_state.pending_image = None
            st.session_state.pending_image_name = None
            st.session_state.pending_cv_data = None
            st.session_state.selected_pill_id = None
            st.session_state.manual_overrides = {}
            st.session_state.mobile_input_mode = "entry"
            st.session_state.chooser_open = False
            st.rerun()
