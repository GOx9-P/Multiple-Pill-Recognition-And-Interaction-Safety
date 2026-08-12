"""Cau hinh runtime cua Module 4 CV Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CVPipelineConfig:
    """Chi chua noi luu artifact vi Module 4 khong nap model moi."""

    output_dir: Path = Path("outputs")

    def with_output_dir(self, output_dir: str | Path) -> "CVPipelineConfig":
        """Tra ve config moi voi thu muc artifact duoc thay the."""

        return replace(self, output_dir=Path(output_dir))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CVPipelineConfig":
        """Doc YAML va giu default khi artifact config chua khai bao."""

        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        artifacts = raw.get("artifacts", {})
        return cls(output_dir=Path(artifacts.get("output_dir", "outputs")))
