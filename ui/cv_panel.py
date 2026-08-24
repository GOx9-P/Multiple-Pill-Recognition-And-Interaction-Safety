"""Image-upload and CV inference panel."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
from PIL import Image

from .config import UI_DEMO_MODE
from .demo_data import DemoCVOutput, build_demo_cv_output, pill_display_fields
from .drawing_utils import draw_cv_overlay
from .model_loader import CVPipelineLoadResult


def _persist_upload(uploaded_file: Any) -> tuple[Path, str]:
    """Persist an uploaded image because the real request API requires image_path."""

    contents = uploaded_file.getvalue()
    digest = hashlib.sha256(contents).hexdigest()
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    directory = Path(tempfile.gettempdir()) / "pill_safety_streamlit_uploads"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}{suffix}"
    if not destination.exists():
        destination.write_bytes(contents)
    return destination, digest


def _render_pill_card(pill: Any) -> None:
    fields = pill_display_fields(pill)
    st.markdown(
        f"""
        <div class="pill-card">
            <div class="pill-card-title">
                {fields["instance_id"]}
                <span class="pill-badge pill-badge-detected">{fields["status"]}</span>
            </div>
            <div class="pill-card-row"><span>Hình dạng</span><span>{fields["shape"]}</span></div>
            <div class="pill-card-row"><span>Màu sắc</span><span>{fields["color"]}</span></div>
            <div class="pill-card-row"><span>Khắc chữ</span><span>{fields["imprint"]}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_real_inference(cv_pipeline: Any, image_path: Path, image_digest: str) -> None:
    from pill_safety.schemas import SegmentationInferenceRequest

    request = SegmentationInferenceRequest(
        request_id=str(uuid4()),
        session_id=st.session_state.setdefault("cv_session_id", str(uuid4())),
        image_id=image_digest,
        image_path=str(image_path),
    )
    try:
        with st.spinner("Đang chạy phân đoạn, thuộc tính và OCR..."):
            artifacts = cv_pipeline.predict_with_artifacts(request)
    except Exception as error:
        st.error(f"Không thể chạy suy luận CV: {error}")
        return

    st.session_state["cv_output"] = artifacts.output
    st.session_state["cv_artifacts"] = artifacts
    st.session_state["cv_output_is_demo"] = False
    st.session_state["interaction_result"] = None
    st.session_state["interaction_result_is_demo"] = False


def _run_demo_inference() -> None:
    st.session_state["cv_output"] = build_demo_cv_output()
    st.session_state["cv_artifacts"] = None
    st.session_state["cv_output_is_demo"] = True
    st.session_state["interaction_result"] = None
    st.session_state["interaction_result_is_demo"] = False


def render_cv_panel(cv_load_result: CVPipelineLoadResult) -> None:
    """Render upload controls and CV results; gracefully handle missing models."""

    st.markdown('<div class="cv-panel-wrapper">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">
                <span class="section-badge blue">1</span>
                <h2 class="section-title">PILL RECOGNITION</h2>
            </div>
            <p class="section-subtitle">Upload an image containing one or more pills</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not cv_load_result.available:
        st.markdown(
            """
            <div class="pill-warning-box">
                <strong>CV model chưa sẵn sàng</strong><br>
                Model artifacts chưa được cấu hình. Giao diện vẫn có thể xem và kiểm tra luồng hoạt động.
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "Chọn ảnh thuốc",
        type=["jpg", "jpeg", "png"],
        help="Hỗ trợ JPG, JPEG, PNG",
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    image_path, image_digest = _persist_upload(uploaded_file)
    if st.session_state["uploaded_image_key"] != image_digest:
        st.session_state["uploaded_image_key"] = image_digest
        st.session_state["uploaded_image_path"] = str(image_path)
        st.session_state["cv_output"] = None
        st.session_state["cv_artifacts"] = None
        st.session_state["cv_output_is_demo"] = False
        st.session_state["interaction_result"] = None
        st.session_state["interaction_result_is_demo"] = False

    with Image.open(image_path) as uploaded_image:
        source_image = uploaded_image.convert("RGB")
    st.session_state["uploaded_image"] = source_image.copy()
    st.image(source_image, caption="Ảnh gốc", use_container_width=True)

    demo_mode = st.session_state.get("demo_mode", UI_DEMO_MODE)
    pipeline_available = cv_load_result.available

    if demo_mode:
        if st.button("🧠 Nhận diện thuốc (Demo UI)", type="primary"):
            _run_demo_inference()
            st.rerun()
    elif pipeline_available:
        if st.button("🧠 Nhận diện thuốc", type="primary"):
            _run_real_inference(cv_load_result.pipeline, image_path, image_digest)
            st.rerun()
    else:
        st.button("🧠 Nhận diện thuốc", type="primary", disabled=True)
        st.caption("Nút bị vô hiệu hóa vì model artifacts chưa sẵn sàng.")

    cv_output = st.session_state.get("cv_output")
    if cv_output is None:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    is_demo = st.session_state.get("cv_output_is_demo", False)
    if is_demo:
        st.markdown(
            '<span class="pill-badge pill-badge-demo">Demo UI</span>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Kết quả nhận diện")

    has_real_overlay = (
        not is_demo
        and not isinstance(cv_output, DemoCVOutput)
        and getattr(cv_output, "pills", None)
    )
    if has_real_overlay:
        st.image(
            draw_cv_overlay(source_image, cv_output),
            caption="Overlay phân đoạn và bounding box",
            use_container_width=True,
        )
    else:
        st.image(source_image, caption="Ảnh gốc", use_container_width=True)
        st.markdown(
            """
            <div class="pill-overlay-placeholder">
                Overlay sẽ xuất hiện sau khi mô hình CV hoàn tất.
            </div>
            """,
            unsafe_allow_html=True,
        )

    pills = cv_output.pills if hasattr(cv_output, "pills") else []
    st.success(f"Đã phát hiện {len(pills)} viên/đối tượng.")
    for pill in pills:
        _render_pill_card(pill)

    st.markdown('</div>', unsafe_allow_html=True)
