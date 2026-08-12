"""Dieu phoi end-to-end Module 1, 2, 3 va fusion Module 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pill_safety.schemas import (
    AttributeInferenceRequest,
    CVPipelineInput,
    CVPipelineOutput,
    OCRInferenceRequest,
    SegmentationInferenceRequest,
)

from .cv_pipeline import CVPipelineArtifacts, CVPipelineAssembler


class _Predictor(Protocol):
    """Giao dien toi thieu cua predictor co the tra output kem artifact."""

    def predict_with_artifacts(self, request: Any) -> Any:
        """Chay inference cho mot request va tra object co truong output."""

        ...


@dataclass(frozen=True)
class FullCVPipelineArtifacts:
    """Gom output Module 4 va artifact trung gian cua mot lan chay full CV."""

    output: CVPipelineOutput
    segmentation_artifacts: Any
    attribute_artifacts: list[Any]
    ocr_artifacts: list[Any]
    pipeline_artifacts: CVPipelineArtifacts


class FullCVPipeline:
    """Chay segmentation mot lan, sau do chay Attribute va OCR cho tung crop."""

    def __init__(
        self,
        segmentation_predictor: _Predictor,
        attribute_predictor: _Predictor,
        ocr_predictor: _Predictor,
        pipeline_assembler: CVPipelineAssembler,
    ):
        """Nhan predictor da khoi tao de tranh nap lai model cho tung vien."""

        self.segmentation_predictor = segmentation_predictor
        self.attribute_predictor = attribute_predictor
        self.ocr_predictor = ocr_predictor
        self.pipeline_assembler = pipeline_assembler

    def predict(
        self, request: SegmentationInferenceRequest | dict[str, Any]
    ) -> CVPipelineOutput:
        """Chay full pipeline va chi tra cv_output_v1."""

        return self.predict_with_artifacts(request).output

    def predict_with_artifacts(
        self, request: SegmentationInferenceRequest | dict[str, Any]
    ) -> FullCVPipelineArtifacts:
        """Chay ba module CV theo dependency crop/mask roi fusion output cuoi."""

        request = SegmentationInferenceRequest.model_validate(request)
        segmentation_artifacts = self.segmentation_predictor.predict_with_artifacts(
            request
        )
        segmentation_output = segmentation_artifacts.output
        attribute_artifacts: list[Any] = []
        ocr_artifacts: list[Any] = []

        for instance in segmentation_output.instances:
            shared_request = {
                "request_id": segmentation_output.request_id,
                "session_id": segmentation_output.session_id,
                "image_id": segmentation_output.image_id,
                "instance_id": instance.instance_id,
                "instance_token": instance.instance_token,
                "crop_path": instance.crop_path,
                "mask_path": instance.mask_path,
            }
            attribute_artifacts.append(
                self.attribute_predictor.predict_with_artifacts(
                    AttributeInferenceRequest.model_validate(shared_request)
                )
            )
            ocr_artifacts.append(
                self.ocr_predictor.predict_with_artifacts(
                    OCRInferenceRequest.model_validate(shared_request)
                )
            )

        pipeline_artifacts = self.pipeline_assembler.predict_with_artifacts(
            CVPipelineInput(
                segmentation_output=segmentation_output,
                attribute_outputs=[artifact.output for artifact in attribute_artifacts],
                ocr_outputs=[artifact.output for artifact in ocr_artifacts],
            )
        )
        return FullCVPipelineArtifacts(
            output=pipeline_artifacts.output,
            segmentation_artifacts=segmentation_artifacts,
            attribute_artifacts=attribute_artifacts,
            ocr_artifacts=ocr_artifacts,
            pipeline_artifacts=pipeline_artifacts,
        )
