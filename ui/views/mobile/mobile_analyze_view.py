"""Consumer-focused mobile flow for pill recognition and interaction safety."""

from __future__ import annotations

from html import escape
from typing import Any

from PIL import Image
import streamlit as st

from ui.adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output
from ui.components.mobile.mobile_duplicate_card import render_mobile_duplicate_card
from ui.components.mobile.mobile_interaction_card import render_mobile_interaction_card
from ui.components.mobile.mobile_pill_row import render_mobile_pill_row
from ui.demo_data import DEMO_SCENARIO_LABELS, get_demo_image, get_preset_scenario
from ui.drawing_utils import draw_cv_overlay
from ui.mobile_ui_logic import (
    get_mobile_severity_content,
    get_recognition_progress,
    should_show_image_quality_warning,
    sort_interactions_by_severity,
)


def _render_app_bar() -> None:
    st.markdown(
        """
        <div class="mobile-app-bar">
            <div>
                <div class="mobile-app-kicker">PILL SAFETY</div>
                <h1>Kiểm tra thuốc</h1>
            </div>
            <div class="mobile-app-mark" aria-hidden="true">+</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _reset_scan() -> None:
    for key in (
        "current_image", "current_image_name", "raw_cv_data", "pending_image",
        "pending_image_name", "pending_cv_data", "selected_pill_id", "cv_error",
    ):
        st.session_state[key] = None
    st.session_state.manual_overrides = {}
    st.session_state.mobile_input_mode = "entry"


def render_mobile_analyze_view(cv_load_result: Any) -> None:
    """Render capture, review, verification, and safety-result states."""
    defaults = {
        "current_image": None,
        "current_image_name": None,
        "raw_cv_data": None,
        "pending_image": None,
        "pending_image_name": None,
        "pending_cv_data": None,
        "selected_pill_id": None,
        "manual_overrides": {},
        "mobile_input_mode": "entry",
        "cv_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    def execute_analysis(
        image: Image.Image,
        image_name: str,
        preset_data: dict[str, Any] | None = None,
    ) -> None:
        st.session_state.current_image = image
        st.session_state.current_image_name = image_name
        st.session_state.pending_image = None
        st.session_state.pending_image_name = None
        st.session_state.pending_cv_data = None
        st.session_state.manual_overrides = {}
        st.session_state.selected_pill_id = None
        st.session_state.mobile_input_mode = "entry"
        st.session_state.cv_error = None

        if preset_data is not None:
            st.session_state.raw_cv_data = preset_data
            return

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
                request = SegmentationInferenceRequest(
                    request_id=str(uuid4()),
                    session_id=str(uuid4()),
                    image_id=temp_file.stem,
                    image_path=str(temp_file),
                )
                with st.spinner("Đang tìm các viên thuốc trong ảnh..."):
                    artifacts = cv_load_result.pipeline.predict_with_artifacts(request)
                    st.session_state.raw_cv_data = artifacts.output
            except Exception as error:
                st.session_state.raw_cv_data = None
                st.session_state.cv_error = str(error)
        else:
            st.session_state.raw_cv_data = None
            st.session_state.cv_error = cv_load_result.error if cv_load_result else "Hệ thống nhận diện chưa sẵn sàng."

    _render_app_bar()
    current_image = st.session_state.current_image
    pending_image = st.session_state.pending_image
    input_mode = st.session_state.mobile_input_mode

    if current_image is None and pending_image is None:
        st.markdown(
            '<section class="mobile-intro"><h2>Kiểm tra các thuốc bạn sắp dùng</h2>'
            '<p>Chụp rõ tất cả viên thuốc trong cùng một ảnh để nhận diện và kiểm tra tương tác.</p></section>',
            unsafe_allow_html=True,
        )

        if input_mode == "camera":
            st.markdown('<h2 class="mobile-section-title">Chụp ảnh thuốc</h2>', unsafe_allow_html=True)
            camera_file = st.camera_input("Chụp ảnh", key="m_active_cam", label_visibility="collapsed")
            if camera_file is not None:
                image = Image.open(camera_file).convert("RGB")
                st.session_state.pending_image = image
                st.session_state.pending_image_name = "camera_photo.jpg"
                st.session_state.pending_cv_data = None
                st.session_state.mobile_input_mode = "entry"
                st.rerun()
            if st.button("Quay lại", key="m_btn_cancel_cam", use_container_width=True):
                st.session_state.mobile_input_mode = "entry"
                st.rerun()
        elif input_mode == "uploader":
            st.markdown('<h2 class="mobile-section-title">Tải ảnh từ thiết bị</h2>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Chọn ảnh thuốc", type=["jpg", "jpeg", "png", "webp"],
                key="m_active_up", label_visibility="collapsed",
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.pending_image = image
                st.session_state.pending_image_name = uploaded_file.name
                st.session_state.pending_cv_data = None
                st.session_state.mobile_input_mode = "entry"
                st.rerun()
            if st.button("Quay lại", key="m_btn_cancel_up", use_container_width=True):
                st.session_state.mobile_input_mode = "entry"
                st.rerun()
        else:
            camera_col, upload_col = st.columns(2, gap="small")
            with camera_col:
                if st.button("Chụp ảnh", key="m_btn_choose_cam", type="primary", use_container_width=True):
                    st.session_state.mobile_input_mode = "camera"
                    st.rerun()
            with upload_col:
                if st.button("Tải ảnh", key="m_btn_choose_lib", use_container_width=True):
                    st.session_state.mobile_input_mode = "uploader"
                    st.rerun()
            st.markdown(
                '<div class="mobile-photo-guide"><strong>Để nhận diện rõ hơn</strong>'
                '<span>Nền sáng, đủ ánh sáng và các viên không chạm nhau.</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Dùng dữ liệu mẫu để xem nhanh", expanded=False):
                label_to_key = {label: key for key, label in DEMO_SCENARIO_LABELS.items()}
                selected_label = st.selectbox("Kịch bản", options=list(label_to_key), key="mobile_demo_scenario")
                if st.button("Mở kịch bản mẫu", key="m_btn_demo", use_container_width=True):
                    scenario_key = label_to_key[selected_label]
                    st.session_state.pending_image = get_demo_image(scenario_key)
                    st.session_state.pending_image_name = f"demo_{scenario_key}.png"
                    st.session_state.pending_cv_data = get_preset_scenario(scenario_key)
                    st.rerun()
        return

    if pending_image is not None and current_image is None:
        st.markdown(
            '<section class="mobile-intro compact"><h2>Kiểm tra lại ảnh</h2>'
            '<p>Đảm bảo nhìn thấy rõ từng viên thuốc trước khi tiếp tục.</p></section>',
            unsafe_allow_html=True,
        )
        st.image(pending_image, use_container_width=True)
        if st.button("Nhận diện thuốc", key="m_btn_run_analysis", type="primary", use_container_width=True):
            execute_analysis(
                image=pending_image,
                image_name=st.session_state.pending_image_name or "pill_photo.jpg",
                preset_data=st.session_state.pending_cv_data,
            )
            st.rerun()
        if st.button("Chọn ảnh khác", key="m_btn_change_photo", use_container_width=True):
            st.session_state.pending_image = None
            st.session_state.pending_image_name = None
            st.session_state.pending_cv_data = None
            st.rerun()
        return

    if st.session_state.cv_error:
        st.markdown(
            '<div class="mobile-empty-state error"><strong>Chưa thể phân tích ảnh</strong>'
            '<span>Hệ thống nhận diện đang gặp sự cố. Hãy thử ảnh khác hoặc dùng dữ liệu mẫu.</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Chi tiết kỹ thuật", expanded=False):
            st.code(st.session_state.cv_error)
        if st.button("Thử ảnh khác", key="m_btn_retry_after_err", type="primary", use_container_width=True):
            _reset_scan()
            st.rerun()
        return

    pills, quality = parse_cv_output(st.session_state.raw_cv_data)
    if not pills:
        st.markdown(
            '<div class="mobile-empty-state"><strong>Không tìm thấy viên thuốc</strong>'
            '<span>Hãy chụp gần hơn, dùng nền trơn và đặt các viên tách rời nhau.</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Chụp hoặc chọn lại", key="m_btn_retake_empty", type="primary", use_container_width=True):
            _reset_scan()
            st.rerun()
        return

    report = evaluate_safety_and_report(pills=pills, manual_overrides=st.session_state.manual_overrides)
    progress = get_recognition_progress(pills)
    severity = get_mobile_severity_content(report.overall_severity, progress.has_unresolved)

    if should_show_image_quality_warning(quality):
        st.markdown(
            '<div class="mobile-image-warning"><strong>Ảnh có thể chưa đủ rõ</strong>'
            '<span>Hãy kiểm tra kỹ các viên được đánh dấu cần xác nhận.</span></div>',
            unsafe_allow_html=True,
        )

    completeness_html = (
        f'<div class="mobile-completeness-note">{escape(severity.completeness_note)}</div>'
        if severity.completeness_note else ""
    )
    st.markdown(
        f'<section class="mobile-safety-hero {severity.css_class}">'
        f'<div class="mobile-safety-symbol" aria-hidden="true">{severity.symbol}</div>'
        f'<div class="mobile-safety-content"><div class="mobile-safety-eyebrow">{escape(severity.eyebrow)}</div>'
        f'<h2>{escape(severity.title)}</h2><p>{escape(severity.description)}</p>'
        f'<div class="mobile-safety-action"><strong>Bạn cần làm gì</strong>{escape(severity.action)}</div>'
        f'{completeness_html}</div></section>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="mobile-section-heading"><div><span>THUỐC TRONG ẢNH</span>'
        f'<h2>{escape(progress.label)}</h2></div><div class="mobile-progress-count">'
        f'{progress.resolved}/{progress.total}</div></div>',
        unsafe_allow_html=True,
    )
    for index, pill in enumerate(pills, start=1):
        render_mobile_pill_row(pill, display_index=index)
        if pill.status in ("unresolved", "ambiguous"):
            with st.form(key=f"mobile_verify_{index}", clear_on_submit=False):
                current_value = st.session_state.manual_overrides.get(pill.instance_id, "")
                user_value = st.text_input(
                    f"Tên thuốc hoặc mã in trên viên {index}", value=current_value,
                    placeholder="Ví dụ: 84A hoặc Aspirin",
                )
                if st.form_submit_button("Xác nhận viên này", use_container_width=True):
                    if user_value.strip():
                        st.session_state.manual_overrides[pill.instance_id] = user_value.strip()
                    st.rerun()
            if pill.required_action:
                st.warning(pill.required_action)

    ordered_interactions = sort_interactions_by_severity(report.interactions)
    critical_interactions = [item for item in ordered_interactions if item.severity.lower() in ("critical", "contraindicated")]
    other_interactions = [item for item in ordered_interactions if item not in critical_interactions]
    if ordered_interactions or report.duplicate_warnings:
        st.markdown(
            '<div class="mobile-section-heading simple"><div><span>CẢNH BÁO</span>'
            '<h2>Tương tác cần lưu ý</h2></div></div>', unsafe_allow_html=True,
        )
        for interaction in critical_interactions:
            render_mobile_interaction_card(interaction)
        for duplicate in report.duplicate_warnings:
            render_mobile_duplicate_card(duplicate)
        for interaction in other_interactions:
            render_mobile_interaction_card(interaction)

    with st.expander("Xem ảnh và chi tiết nhận diện", expanded=False):
        selected_id = st.session_state.selected_pill_id
        if len(pills) > 1:
            options = {f"Viên {index}": pill.instance_id for index, pill in enumerate(pills, start=1)}
            labels = ["Tất cả viên", *options.keys()]
            current_label = next((label for label, value in options.items() if value == selected_id), "Tất cả viên")
            selected_label = st.selectbox("Làm nổi bật trên ảnh", options=labels, index=labels.index(current_label))
            selected_id = options.get(selected_label)
            st.session_state.selected_pill_id = selected_id
        annotated_image = draw_cv_overlay(current_image, pills, selected_pill_id=selected_id)
        st.image(annotated_image, use_container_width=True)
        for index, pill in enumerate(pills, start=1):
            confidence = f"{pill.match_confidence * 100:.0f}%" if pill.match_confidence is not None else "Chưa đủ dữ liệu"
            st.markdown(
                f"**Viên {index}:** mã in `{escape(pill.imprint_raw)}` · {escape(pill.shape)} · "
                f"{escape(pill.color_primary)} · độ tin cậy {confidence}"
            )

    st.markdown(
        '<div class="mobile-medical-disclaimer"><strong>Giới hạn của kết quả</strong>'
        '<span>Thông tin này chỉ hỗ trợ tham khảo và chỉ dựa trên các thuốc đã nhận diện. '
        'Không tự ý ngừng, đổi hoặc phối hợp thuốc khi chưa hỏi bác sĩ hoặc dược sĩ.</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("Kiểm tra ảnh khác", key="m_btn_reset_scan", use_container_width=True):
        _reset_scan()
        st.rerun()
