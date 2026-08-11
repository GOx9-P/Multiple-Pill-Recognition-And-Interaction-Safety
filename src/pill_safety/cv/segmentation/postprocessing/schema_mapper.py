"""Ánh xạ dữ liệu nội bộ sang public schema của Module 1."""

from __future__ import annotations

from ....schemas import (
    ImageQuality,
    SegmentationEvidence,
    SegmentationInferenceOutput,
    SegmentationInferenceRequest,
    SegmentationInstance,
)

from .instances import ProcessedInstance


def build_segmentation_instance(
    *,
    instance_id: str,
    instance_token: str,
    mask_path: str,
    crop_path: str,
    processed: ProcessedInstance,
    quality_flags: list[str],
) -> SegmentationInstance:
    """Chuyển một instance nội bộ thành đúng schema công khai của Module 1."""

    return SegmentationInstance(
        instance_id=instance_id,
        instance_token=instance_token,
        bbox_xyxy=list(processed.bbox_xyxy),
        mask_path=mask_path,
        crop_path=crop_path,
        segmentation=SegmentationEvidence(
            confidence=round(processed.confidence, 4),
            occlusion_estimate=processed.occlusion_estimate,
            possible_merged_instance=processed.possible_merged_instance,
            possible_non_pill=processed.possible_non_pill,
        ),
        quality_flags=quality_flags,
    )


def build_segmentation_output(
    request: SegmentationInferenceRequest,
    image_quality: ImageQuality,
    instances: list[SegmentationInstance],
) -> SegmentationInferenceOutput:
    """Giữ định danh request và chỉ công khai các trường đã được quy định."""

    return SegmentationInferenceOutput(
        request_id=request.request_id,
        session_id=request.session_id,
        image_id=request.image_id,
        image_quality=image_quality,
        instances=instances,
    )
