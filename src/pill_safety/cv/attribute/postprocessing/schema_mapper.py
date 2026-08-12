"""Ánh xạ prediction shape/color sang contract công khai của Module 2."""

from __future__ import annotations

from typing import Any

from pill_safety.schemas import AttributeInferenceOutput, AttributeInferenceRequest


def build_attribute_output(
    request: AttributeInferenceRequest,
    prediction: dict[str, Any],
) -> AttributeInferenceOutput:
    """Gắn ID từ Module 1 vào prediction và từ chối field ngoài schema Module 2."""

    return AttributeInferenceOutput.model_validate(
        {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "image_id": request.image_id,
            "instance_id": request.instance_id,
            "instance_token": request.instance_token,
            **prediction,
        }
    )
