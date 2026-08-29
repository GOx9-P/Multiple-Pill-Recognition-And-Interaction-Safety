"""Main Clinical Workspace: Analyze View."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import streamlit as st
from streamlit.components import v1 as components
from PIL import Image

from ..adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output
from ..adapters.view_models import ImageQualityViewModel, PillViewModel, SafetyReportViewModel
from ..components import (
    render_clinical_report,
    render_evidence_details,
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
    st.session_state.setdefault("recapture_results", {})
    st.session_state.setdefault("recapture_errors", {})
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("highlighted_pill_ids", [])
    st.session_state.setdefault("active_interaction_index", None)
    st.session_state.setdefault("scroll_to_medication_photo", False)
    st.session_state.setdefault("focused_retake_pill_id", None)

    # 1. Callback when user uploads an image or captures from camera
    def on_image_selected(image: Image.Image, image_name: str) -> None:
        st.session_state.current_image = image
        st.session_state.current_image_name = image_name
        st.session_state.manual_overrides = {}
        st.session_state.recapture_results = {}
        st.session_state.recapture_errors = {}
        st.session_state.selected_pill_id = None
        st.session_state.highlighted_pill_ids = []
        st.session_state.active_interaction_index = None
        st.session_state.scroll_to_medication_photo = False
        st.session_state.focused_retake_pill_id = None
        st.session_state.pop("select_active_pill_box", None)
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
                with st.spinner("Analyzing the medication image..."):
                    artifacts = cv_load_result.pipeline.predict_with_artifacts(req)
                    st.session_state.raw_cv_data = artifacts.output
                    st.session_state.cv_error = None
            except Exception as err:
                import traceback
                st.session_state.raw_cv_data = None
                st.session_state.cv_error = f"Analysis could not be completed: {err}\n{traceback.format_exc()}"
        else:
            st.session_state.raw_cv_data = None
            st.session_state.cv_error = cv_load_result.error if cv_load_result else "The medication analysis service could not be started."

    def on_pill_recaptured(instance_id: str, image: Image.Image) -> None:
        """Run the full pipeline for one close-up and retain its original medication ID."""
        if not cv_load_result or not cv_load_result.available:
            st.session_state.recapture_errors[instance_id] = (
                "The medication analysis service is not available for this retake."
            )
            return

        try:
            import tempfile
            from pathlib import Path
            from uuid import uuid4

            from pill_safety.schemas import SegmentationInferenceRequest

            temp_dir = Path(tempfile.gettempdir()) / "pill_safety_retakes"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"retake_{instance_id}_{uuid4().hex[:8]}.png"
            image.save(temp_file)

            request = SegmentationInferenceRequest(
                request_id=str(uuid4()),
                session_id=str(uuid4()),
                image_id=temp_file.stem,
                image_path=str(temp_file),
            )
            with st.spinner(f"Rechecking {instance_id}..."):
                artifacts = cv_load_result.pipeline.predict_with_artifacts(request)
                retake_pills, _ = parse_cv_output(artifacts.output)

            if len(retake_pills) != 1:
                st.session_state.recapture_errors[instance_id] = (
                    "The retake must show exactly one medication. Please capture only this medication."
                )
                return

            st.session_state.recapture_results[instance_id] = retake_pills[0]
            st.session_state.recapture_errors.pop(instance_id, None)
        except Exception:
            st.session_state.recapture_errors[instance_id] = (
                "This retake could not be analyzed. Please try a sharper close-up."
            )

    # 2. Render Upload Panel (Upload / Camera)
    with st.container():
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        render_upload_panel(on_image_selected)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Handle Errors or Empty State
    if st.session_state.cv_error:
        st.error(f"**Analysis error:**\n\n```\n{st.session_state.cv_error}\n```\n\nPlease try another image or contact the application administrator.")

    if st.session_state.current_image is None:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 20px; background: white; border: 1.5px dashed var(--border-medium); border-radius: 12px; margin-top: 16px;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">💊</div>
                <h4 style="margin: 0 0 6px 0; color: var(--text-primary);">No image selected</h4>
                <p style="margin: 0; font-size: 0.875rem; color: var(--text-muted); max-width: 500px; margin: 0 auto;">
                    Upload a medication photo or use the camera above to identify visible medications and screen for known interactions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 4. Main Analysis Section if real image and CV data are ready
    if st.session_state.current_image is not None and st.session_state.raw_cv_data is not None:
        st.success("Analysis complete. Review the identified medications and safety findings below.")
        pills, quality = parse_cv_output(st.session_state.raw_cv_data)

        # A successful close-up updates only the matching original instance, never its ID or position.
        for index, pill in enumerate(pills):
            retake_result = st.session_state.recapture_results.get(pill.instance_id)
            if retake_result is not None:
                pills[index] = replace(
                    retake_result,
                    instance_id=pill.instance_id,
                    bbox_xyxy=pill.bbox_xyxy,
                    mask_path=pill.mask_path,
                    crop_path=pill.crop_path,
                )

        if not pills:
            st.warning("No medications were detected. Try a sharper photo with the medications separated from the background.")
            return

        # Apply manual overrides and evaluate DDI & Report
        report: SafetyReportViewModel = evaluate_safety_and_report(
            pills=pills,
            manual_overrides=st.session_state.manual_overrides,
        )

        # Priority 1: Overall Clinical Safety Banner
        render_safety_banner(report)
        if report.interactions or report.duplicate_warnings:
            st.link_button("View safety findings", "#interaction-review")

        def handle_manual_override(inst_id: str, val: str) -> None:
            """Store a user-confirmed value and keep the original medication ID."""
            st.session_state.manual_overrides[inst_id] = val

        def focus_pill(instance_id: str) -> None:
            """Focus one medication in the photo and clear any prior DDI pair highlight."""
            st.session_state.selected_pill_id = instance_id
            st.session_state.highlighted_pill_ids = []
            st.session_state.active_interaction_index = None
            st.session_state.select_active_pill_box = instance_id

        def request_retake(instance_id: str) -> None:
            """Open the matching retake menu directly beneath its medication card."""
            focus_pill(instance_id)
            st.session_state.focused_retake_pill_id = instance_id

        def focus_interaction(interaction_index: int) -> None:
            """Highlight exactly the detected pills that produced one DDI finding."""
            interaction = report.interactions[interaction_index]
            source_instances = interaction.source_instances
            st.session_state.highlighted_pill_ids = source_instances
            st.session_state.active_interaction_index = interaction_index
            st.session_state.selected_pill_id = source_instances[0] if source_instances else None
            if source_instances:
                st.session_state.select_active_pill_box = source_instances[0]
            st.session_state.scroll_to_medication_photo = bool(source_instances)

        def clear_interaction_focus() -> None:
            """Remove the temporary interaction highlight without changing analysis results."""
            st.session_state.highlighted_pill_ids = []
            st.session_state.active_interaction_index = None

        # A normal anchor supports the top-level shortcut to the safety review.
        st.markdown('<div id="medication-photo"></div>', unsafe_allow_html=True)
        if st.session_state.scroll_to_medication_photo:
            components.html(
                """
                <script>
                window.setTimeout(() => {
                    const target = window.parent.document.getElementById("medication-photo");
                    if (target) target.scrollIntoView({behavior: "smooth", block: "start"});
                }, 150);
                </script>
                """,
                height=0,
            )
            st.session_state.scroll_to_medication_photo = False

        # Keep image evidence and medication results in parallel, independently readable panels.
        col_left, col_right = st.columns([1.1, 1.0], gap="large")

        with col_left:
            with st.container(height=680, border=True):
                render_visual_viewer(
                    image=st.session_state.current_image,
                    pills=pills,
                    quality=quality,
                    selected_pill_id=st.session_state.selected_pill_id,
                    on_pill_selected=focus_pill,
                    highlighted_pill_ids=st.session_state.highlighted_pill_ids,
                    on_clear_interaction_highlight=clear_interaction_focus,
                )

        with col_right:
            with st.container(height=680, border=True):
                render_pill_cards(
                    pills=pills,
                    selected_pill_id=st.session_state.selected_pill_id,
                    on_pill_selected=focus_pill,
                    on_retake_requested=request_retake,
                    on_recapture=on_pill_recaptured,
                    on_manual_override=handle_manual_override,
                    recapture_errors=st.session_state.recapture_errors,
                    focused_retake_pill_id=st.session_state.focused_retake_pill_id,
                )

        # Group the safety findings into one focused workspace below the two primary panels.
        st.markdown('<div id="interaction-review"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_interaction_cards(
                interactions=report.interactions,
                duplicates=report.duplicate_warnings,
                on_interaction_selected=focus_interaction,
                active_interaction_index=st.session_state.active_interaction_index,
            )
            st.divider()
            render_clinical_report(report)

        # Keep implementation diagnostics available without making the user-facing result look like raw logs.
        with st.expander("Evidence details", expanded=False):
            render_evidence_details(
                pills=pills,
                quality=quality,
                raw_cv_data=st.session_state.raw_cv_data,
            )
    else:
        # Clean Empty State
        st.markdown(
            """
            <div class="clinical-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🔬</div>
                <h3 style="color: var(--text-primary); margin-bottom: 0.35rem; font-size: 1.25rem;">Ready to analyze medications</h3>
                <p style="color: var(--text-muted); max-width: 500px; margin: 0 auto 1.25rem auto; font-size: 0.875rem;">
                    Upload a photo from your computer or camera to identify medications and review potential safety findings.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
