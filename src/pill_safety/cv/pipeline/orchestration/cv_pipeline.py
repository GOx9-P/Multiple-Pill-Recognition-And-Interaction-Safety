"""Entrypoint logic cua Module 4: luu CV output da fusion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pill_safety.schemas import CVPipelineInput, CVPipelineOutput

from ..config import CVPipelineConfig
from ..fusion import fuse_cv_outputs


@dataclass(frozen=True)
class CVPipelineArtifacts:
    """Output Module 4 va duong dan JSON artifact phuc vu RAG/debug."""

    output: CVPipelineOutput
    schema_json_path: Path


def _safe_directory_name(value: str) -> str:
    """Chuyen ID thanh ten thu muc an toan de artifact cac request khong ghi de nhau."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "request"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 de RAG va cong cu audit co the doc truc tiep."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


class CVPipelineAssembler:
    """Fusion output da co cua ba module CV, khong chay lai model inference."""

    def __init__(self, config: CVPipelineConfig | None = None):
        """Khoi tao assembler voi noi luu artifact cua Module 4."""

        self.config = config or CVPipelineConfig()

    def predict(self, value: CVPipelineInput | dict) -> CVPipelineOutput:
        """Tra ve CV output da fusion ma khong ghi artifact ra dia."""

        return fuse_cv_outputs(value)

    def predict_with_artifacts(
        self, value: CVPipelineInput | dict
    ) -> CVPipelineArtifacts:
        """Fusion output va luu mot JSON cv_output_v1 namespace theo request/image."""

        output = fuse_cv_outputs(value)
        output_directory = (
            self.config.output_dir
            / "predictions"
            / "cv_pipeline"
            / _safe_directory_name(output.request_id)
            / _safe_directory_name(output.image_id)
        )
        schema_json_path = output_directory / "cv_output.json"
        _write_json(schema_json_path, output.model_dump(mode="json"))
        return CVPipelineArtifacts(output=output, schema_json_path=schema_json_path)
