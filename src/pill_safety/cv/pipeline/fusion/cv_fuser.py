"""Fusion strict cac output CV da tinh xong thanh Module 4 schema."""

from __future__ import annotations

from typing import TypeVar

from pill_safety.schemas import CVPill, CVPipelineInput, CVPipelineOutput
from pill_safety.schemas.attribute import AttributeInferenceOutput
from pill_safety.schemas.ocr import OCRInferenceOutput
from pill_safety.schemas.segmentation import SegmentationInferenceOutput, SegmentationInstance


TokenOutput = TypeVar("TokenOutput", AttributeInferenceOutput, OCRInferenceOutput)


def _index_by_instance_token(
    outputs: list[TokenOutput],
    module_name: str,
) -> dict[str, TokenOutput]:
    """Lap chi muc token va tu choi hai output cung dai dien mot vien thuoc."""

    indexed: dict[str, TokenOutput] = {}
    for output in outputs:
        token = output.instance_token
        if token in indexed:
            raise ValueError(
                f"{module_name} contains duplicate instance_token: {token!r}."
            )
        indexed[token] = output
    return indexed


def _validate_output_identity(
    output: AttributeInferenceOutput | OCRInferenceOutput,
    segmentation: SegmentationInferenceOutput,
    instance: SegmentationInstance,
    module_name: str,
) -> None:
    """Dam bao output phu thuoc cung request, image va instance voi segmentation."""

    expected = {
        "request_id": segmentation.request_id,
        "session_id": segmentation.session_id,
        "image_id": segmentation.image_id,
    }
    actual = {
        "request_id": output.request_id,
        "session_id": output.session_id,
        "image_id": output.image_id,
    }
    if actual != expected:
        raise ValueError(
            f"{module_name} identity does not match segmentation output: "
            f"expected {expected}, got {actual}."
        )
    if output.instance_id != instance.instance_id:
        raise ValueError(
            f"{module_name} instance_id does not match token {instance.instance_token!r}: "
            f"expected {instance.instance_id!r}, got {output.instance_id!r}."
        )


def _derive_cv_status(
    instance: SegmentationInstance,
    segmentation: SegmentationInferenceOutput,
    ocr: OCRInferenceOutput,
) -> str:
    """Gan trang thai theo cac safety signal da duoc CV_Module.md quy dinh."""

    if instance.segmentation.possible_non_pill:
        return "unknown_object"
    if segmentation.image_quality.status == "unusable":
        return "insufficient_visual_evidence"
    if instance.segmentation.possible_merged_instance or not ocr.imprint.visible:
        return "partial_features"
    return "features_ready"


def fuse_cv_outputs(value: CVPipelineInput | dict) -> CVPipelineOutput:
    """Ghep Module 1, 2, 3 theo token va de OCR lam source of truth cho scoreline."""

    payload = CVPipelineInput.model_validate(value)
    segmentation = payload.segmentation_output
    attributes = _index_by_instance_token(payload.attribute_outputs, "Attribute output")
    ocr_results = _index_by_instance_token(payload.ocr_outputs, "OCR output")
    segmentation_tokens = {item.instance_token for item in segmentation.instances}

    for module_name, indexed in (("Attribute output", attributes), ("OCR output", ocr_results)):
        unexpected = sorted(set(indexed) - segmentation_tokens)
        if unexpected:
            raise ValueError(
                f"{module_name} has instance_token not emitted by segmentation: {unexpected}."
            )

    pills: list[CVPill] = []
    for instance in segmentation.instances:
        token = instance.instance_token
        attribute = attributes.get(token)
        ocr = ocr_results.get(token)
        if attribute is None or ocr is None:
            missing = []
            if attribute is None:
                missing.append("attribute")
            if ocr is None:
                missing.append("ocr")
            raise ValueError(
                f"Cannot fuse {instance.instance_id!r}: missing {', '.join(missing)} output."
            )
        _validate_output_identity(attribute, segmentation, instance, "Attribute output")
        _validate_output_identity(ocr, segmentation, instance, "OCR output")

        pills.append(
            CVPill(
                instance_id=instance.instance_id,
                instance_token=token,
                cv_status=_derive_cv_status(instance, segmentation, ocr),
                bbox_xyxy=instance.bbox_xyxy,
                mask_path=instance.mask_path,
                crop_path=instance.crop_path,
                segmentation=instance.segmentation,
                shape=attribute.shape,
                color=attribute.color,
                dosage_form=attribute.dosage_form,
                # OCR owns this field; Attribute only provided an unknown placeholder.
                scoreline=ocr.scoreline,
                logo_or_symbol=attribute.logo_or_symbol,
                damage_or_occlusion=attribute.damage_or_occlusion,
                imprint_visibility=ocr.imprint_visibility,
                imprint={
                    "visible": ocr.imprint.visible,
                    "raw": ocr.imprint.raw,
                    "confidence": ocr.imprint.confidence,
                    "ocr_observations": ocr.imprint.ocr_observations,
                    "normalized_candidates": ocr.imprint.normalized_candidates,
                },
                quality_flags=instance.quality_flags,
            )
        )

    return CVPipelineOutput(
        request_id=segmentation.request_id,
        session_id=segmentation.session_id,
        image_id=segmentation.image_id,
        image_quality=segmentation.image_quality,
        pills=pills,
    )
