"""Main Clinical Workspace: Analyze View."""

from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image

from ..adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output
from ..adapters.view_models import ImageQualityViewModel, PillViewModel, SafetyReportViewModel
from ..components import (
    render_clinical_report,
    render_interaction_cards,
    render_pill_cards,
    render_safety_banner,
    render_upload_panel,
    render_visual_viewer,
)


def render_analyze_view(cv_load_result: Any) -> None:
    """Render the end-to-end pill recognition, RAG identification, and safety analysis workspace."""
    # Ensure session state variables
    st.session_state.setdefault("current_image", None)
    st.session_state.setdefault("current_image_name", None)
    st.session_state.setdefault("raw_cv_data", None)
    st.session_state.setdefault("cv_error", None)
    st.session_state.setdefault("selected_pill_id", None)
    st.session_state.setdefault("manual_overrides", {})
    st.session_state.setdefault("pipeline_running", False)

    # 1. Callback when user uploads an image or captures from camera
    def on_image_selected(image: Image.Image, image_name: str) -> None:
        st.session_state.current_image = image
        st.session_state.current_image_name = image_name
        st.session_state.manual_overrides = {}
        st.session_state.selected_pill_id = None
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
                with st.spinner("🤖 Đang chạy mô hình AI nhận diện (YOLOv11-Seg + ResNet18 + PaddleOCR)..."):
                    artifacts = cv_load_result.pipeline.predict_with_artifacts(req)
                    st.session_state.raw_cv_data = artifacts.output
                    st.session_state.cv_error = None
            except Exception as err:
                import traceback
                st.session_state.raw_cv_data = None
                st.session_state.cv_error = f"Lỗi trong quá trình chạy Inference: {err}\n{traceback.format_exc()}"
        else:
            st.session_state.raw_cv_data = None
            st.session_state.cv_error = cv_load_result.error if cv_load_result else "Mô hình Computer Vision chưa được khởi tạo thành công."

    # 2. Render Upload Panel (Upload / Camera)
    with st.container():
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        render_upload_panel(on_image_selected)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Handle Errors or Empty State
    if st.session_state.cv_error:
        st.error(f"❌ **Lỗi phân tích Computer Vision:**\n\n```\n{st.session_state.cv_error}\n```\n\n*Gợi ý: Kiểm tra file weights YOLOv11 (.pt) và ResNet-18 (.pt, .json) trên môi trường chạy.*")

    if st.session_state.current_image is None:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 20px; background: white; border: 1.5px dashed var(--border-medium); border-radius: 12px; margin-top: 16px;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">💊</div>
                <h4 style="margin: 0 0 6px 0; color: var(--text-primary);">Chưa có ảnh nào được chọn</h4>
                <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted); max-width: 500px; margin: 0 auto;">
                    Vui lòng tải tệp ảnh chụp hoặc dùng Camera ở khung phía trên để hệ thống bắt đầu quét viên thuốc, định danh RxNorm và phát hiện tương tác đối kháng DDI.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 4. Main Analysis Section if real image and CV data are ready
    if st.session_state.current_image is not None and st.session_state.raw_cv_data is not None:
        st.success("✅ **AI Pipeline Thành Công:** Ảnh đã được nhận diện trực tiếp bằng YOLOv11 Segmentation + ResNet-18 Multi-Head + PaddleOCR!")
        pills, quality = parse_cv_output(st.session_state.raw_cv_data)

        if not pills:
            st.warning("⚠️ Mô hình YOLOv11 không phát hiện viên thuốc nào trong bức ảnh này. Vui lòng chụp rõ nét hơn hoặc đặt thuốc trên nền tương phản sáng.")
            return

        # Apply manual overrides and evaluate DDI & Report
        report: SafetyReportViewModel = evaluate_safety_and_report(
            pills=pills,
            manual_overrides=st.session_state.manual_overrides,
        )

        # Priority 1: Overall Clinical Safety Banner
        render_safety_banner(report)

        # Priority 2: Split-View Workspace (Left: Evidence Viewer; Right: Detected Pills)
        col_left, col_right = st.columns([1.1, 1.0], gap="large")

        with col_left:
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            render_visual_viewer(
                image=st.session_state.current_image,
                pills=pills,
                quality=quality,
                selected_pill_id=st.session_state.selected_pill_id,
                on_pill_selected=lambda pid: setattr(st.session_state, "selected_pill_id", pid),
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            def handle_manual_override(inst_id: str, val: str) -> None:
                st.session_state.manual_overrides[inst_id] = val

            render_pill_cards(
                pills=pills,
                selected_pill_id=st.session_state.selected_pill_id,
                on_pill_selected=lambda pid: setattr(st.session_state, "selected_pill_id", pid),
                on_manual_override=handle_manual_override,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Priority 3: Drug Interaction & Duplicate Ingredient Cards
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        render_interaction_cards(
            interactions=report.interactions,
            duplicates=report.duplicate_warnings,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Priority 4: Full Clinical Report
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        render_clinical_report(report)
        st.markdown('</div>', unsafe_allow_html=True)

        # Priority 5: Advanced Developer / XAI Trace Expander
        with st.expander("🛠️ Advanced Developer & XAI Pipeline Trace", expanded=False):
            st.markdown("**Raw Structured CV Output JSON:**")
            if isinstance(st.session_state.raw_cv_data, dict):
                st.json(st.session_state.raw_cv_data)
            else:
                st.write(str(st.session_state.raw_cv_data))
    else:
        # Clean Empty State
        st.markdown(
            """
            <div class="clinical-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🔬</div>
                <h3 style="color: var(--text-primary); margin-bottom: 0.35rem; font-size: 1.25rem;">Sẵn sàng phân tích đơn thuốc</h3>
                <p style="color: var(--text-muted); max-width: 500px; margin: 0 auto 1.25rem auto; font-size: 0.875rem;">
                    Hãy tải lên ảnh chụp từ máy tính / camera, hoặc bấm chọn nhanh một trong các 
                    <strong>Kịch bản mẫu</strong> ở trên để trải nghiệm quy trình nhận diện và đối soát an toàn.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
