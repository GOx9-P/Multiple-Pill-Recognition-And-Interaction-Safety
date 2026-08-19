"""Coordinate the post-CV medication identification and safety workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pill_safety.schemas.cv_pipeline import CVPipelineOutput
from pill_safety.schemas.rag import ContextBuilderInput, DdiRequest, RagIdentifyRequest

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class EndToEndArtifacts:
    """Return the complete output and the audit artifacts written for one run."""

    output: dict[str, Any]
    output_dir: Path
    paths: dict[str, Path]


class EndToEndPostCvPipeline:
    """Run RAG, DDI, and LLM reporting from one validated ``cv_output_v1``."""

    def __init__(
        self,
        identification_service: Any,
        ddi_lookup_service: Any,
        context_builder: Any | None = None,
        report_generator: Any | None = None,
    ) -> None:
        """Receive ready services so tests can replace external dependencies safely."""

        self.identification_service = identification_service
        self.ddi_lookup_service = ddi_lookup_service
        if context_builder is None:
            from pill_safety.rag.reporting.context_builder import ContextBuilderService

            context_builder = ContextBuilderService()
        if report_generator is None:
            from pill_safety.rag.reporting.llm_report_generator import LlmReportGenerator

            report_generator = LlmReportGenerator()
        self.context_builder = context_builder
        self.report_generator = report_generator

    @classmethod
    def from_database_session(
        cls,
        db: "Session",
        *,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
    ) -> "EndToEndPostCvPipeline":
        """Build production services that share one SQLAlchemy database session."""

        from pill_safety.rag.ddi.ddi_lookup_service import DdiLookupService
        from pill_safety.rag.identification_service import IdentificationService
        from pill_safety.rag.reporting.llm_report_generator import LlmReportGenerator

        return cls(
            identification_service=IdentificationService(db),
            ddi_lookup_service=DdiLookupService(db),
            report_generator=LlmReportGenerator(
                provider_name=llm_provider,
                api_key=llm_api_key,
            ),
        )

    def run(
        self,
        cv_output: CVPipelineOutput | Mapping[str, Any],
        *,
        market: str = "US",
        known_drug_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run every LLM/RAG stage while preserving the original CV output."""

        cv_payload = self._validate_cv_output(cv_output)
        request_id = cv_payload["request_id"]
        session_id = cv_payload["session_id"]

        rag_request_payload = {
            "schema_version": "rag_request_v1",
            "request_id": request_id,
            "session_id": session_id,
            "market": market,
            "known_drug_names": known_drug_names or [],
            "cv_output": cv_payload,
        }
        # Validate the public contract, but pass Module 4's original JSON through
        # unchanged so optional CV evidence is never discarded before retrieval.
        RagIdentifyRequest.model_validate(rag_request_payload)
        rag_identification = self.identification_service.identify(rag_request_payload)

        ddi_request = DdiRequest.model_validate(
            {
                "schema_version": "ddi_request_v0",
                "request_id": request_id,
                "session_id": session_id,
                "identified_products": self._accepted_products(rag_identification),
            }
        ).model_dump(mode="json")
        ddi_output = self.ddi_lookup_service.lookup_ddi(ddi_request)

        context_input = ContextBuilderInput.model_validate(
            {
                "schema_version": "context_builder_input_v0",
                "request_id": request_id,
                "session_id": session_id,
                "cv_output": cv_payload,
                "rag_identification": rag_identification,
                "ddi_output": ddi_output,
            }
        ).model_dump(mode="json")
        llm_context = self.context_builder.build_context(context_input)
        llm_report = self.report_generator.generate_report(llm_context)

        return {
            "schema_version": "end_to_end_result_v1",
            "request_id": request_id,
            "session_id": session_id,
            "image_id": cv_payload["image_id"],
            "cv_output": cv_payload,
            "rag_identification": rag_identification,
            "ddi_output": ddi_output,
            "pill_summary": self._build_pill_summary(
                cv_output=cv_payload,
                rag_identification=rag_identification,
                ddi_output=ddi_output,
            ),
            "llm_context": llm_context,
            "llm_report": llm_report,
        }

    def run_with_artifacts(
        self,
        cv_output: CVPipelineOutput | Mapping[str, Any],
        *,
        output_dir: Path,
        market: str = "US",
        known_drug_names: list[str] | None = None,
    ) -> EndToEndArtifacts:
        """Run the workflow and persist independently inspectable JSON and CSV files."""

        output = self.run(
            cv_output,
            market=market,
            known_drug_names=known_drug_names,
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payloads = {
            "cv_output": output["cv_output"],
            "rag_identification": output["rag_identification"],
            "ddi_output": output["ddi_output"],
            "llm_context": output["llm_context"],
            "llm_report": output["llm_report"],
            "end_to_end_result": output,
        }
        paths: dict[str, Path] = {}
        for name, payload in payloads.items():
            path = output_dir / f"{name}.json"
            self._write_json(path, payload)
            paths[name] = path

        summary_path = output_dir / "pill_summary.csv"
        self._write_summary_csv(summary_path, output["pill_summary"])
        paths["pill_summary"] = summary_path

        report_path = output_dir / "llm_report.txt"
        report_path.write_text(
            str(output["llm_report"]["formatted_report_text"]),
            encoding="utf-8",
        )
        paths["llm_report_text"] = report_path
        return EndToEndArtifacts(output=output, output_dir=output_dir, paths=paths)

    @staticmethod
    def _validate_cv_output(
        cv_output: CVPipelineOutput | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reject non-Module-4 payloads before they can reach retrieval or DDI."""

        if isinstance(cv_output, CVPipelineOutput):
            validated = cv_output
        else:
            validated = CVPipelineOutput.model_validate(dict(cv_output))
        return validated.model_dump(mode="json")

    @staticmethod
    def _accepted_products(rag_identification: Mapping[str, Any]) -> list[dict[str, str]]:
        """Map only accepted RAG products to the exact identifier expected by DDI."""

        products: list[dict[str, str]] = []
        for result in rag_identification.get("pill_results") or []:
            if result.get("identification_status") != "identified":
                continue
            accepted = result.get("accepted_product") or {}
            drug_id = accepted.get("drug_id")
            product_code = accepted.get("product_code")
            if drug_id is not None:
                product_id = f"drug_{drug_id}"
            elif product_code:
                product_id = str(product_code)
            else:
                raise ValueError(
                    "Identified RAG result is missing both drug_id and product_code."
                )
            products.append(
                {
                    "instance_id": str(result["instance_id"]),
                    "product_id": product_id,
                }
            )
        return products

    @staticmethod
    def _build_pill_summary(
        *,
        cv_output: Mapping[str, Any],
        rag_identification: Mapping[str, Any],
        ddi_output: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Create one join-safe table row per CV pill using ``instance_id`` only."""

        rag_by_instance = {
            str(item.get("instance_id")): item
            for item in rag_identification.get("pill_results") or []
        }
        interactions_by_instance: dict[str, list[dict[str, Any]]] = {}
        for interaction in ddi_output.get("interactions") or []:
            for instance_id in interaction.get("source_instance_ids") or []:
                interactions_by_instance.setdefault(str(instance_id), []).append(interaction)
        duplicates_by_instance: dict[str, list[dict[str, Any]]] = {}
        for warning in ddi_output.get("duplicate_ingredient_warnings") or []:
            for instance_id in warning.get("source_instance_ids") or []:
                duplicates_by_instance.setdefault(str(instance_id), []).append(warning)

        rows: list[dict[str, Any]] = []
        for pill in cv_output.get("pills") or []:
            instance_id = str(pill["instance_id"])
            rag = rag_by_instance.get(instance_id, {})
            candidates = rag.get("top_candidates") or []
            top_candidate = candidates[0] if candidates else None
            scoreline = pill.get("scoreline") or {}
            rows.append(
                {
                    "instance_id": instance_id,
                    "instance_token": pill.get("instance_token"),
                    "crop_path": pill.get("crop_path"),
                    "mask_path": pill.get("mask_path"),
                    "cv_status": pill.get("cv_status"),
                    "segmentation_confidence": (pill.get("segmentation") or {}).get("confidence"),
                    "shape_label": (pill.get("shape") or {}).get("label"),
                    "shape_confidence": (pill.get("shape") or {}).get("confidence"),
                    "color_primary": (pill.get("color") or {}).get("primary"),
                    "color_confidence": (pill.get("color") or {}).get("confidence"),
                    "imprint_raw": (pill.get("imprint") or {}).get("raw"),
                    "imprint_confidence": (pill.get("imprint") or {}).get("confidence"),
                    "scoreline_visible": scoreline.get("visible"),
                    "identification_status": rag.get("identification_status", "not_processed"),
                    "required_action": rag.get("required_action"),
                    "top_candidate_name": (top_candidate or {}).get("product_name"),
                    "top_candidate_score": (top_candidate or {}).get("final_score"),
                    "top1_top2_margin": ((top_candidate or {}).get("evidence") or {}).get("top1_top2_margin"),
                    "accepted_product": rag.get("accepted_product"),
                    "interaction_ids": [item.get("interaction_id") for item in interactions_by_instance.get(instance_id, [])],
                    "interaction_severities": [item.get("severity") for item in interactions_by_instance.get(instance_id, [])],
                    "duplicate_ingredients": [item.get("ingredient_name") for item in duplicates_by_instance.get(instance_id, [])],
                }
            )
        return rows

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write UTF-8 JSON without losing Vietnamese report content."""

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _write_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
        """Write a flat table while serializing nested evidence into JSON cells."""

        fields = [
            "instance_id",
            "instance_token",
            "crop_path",
            "mask_path",
            "cv_status",
            "segmentation_confidence",
            "shape_label",
            "shape_confidence",
            "color_primary",
            "color_confidence",
            "imprint_raw",
            "imprint_confidence",
            "scoreline_visible",
            "identification_status",
            "required_action",
            "top_candidate_name",
            "top_candidate_score",
            "top1_top2_margin",
            "accepted_product",
            "interaction_ids",
            "interaction_severities",
            "duplicate_ingredients",
        ]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flattened = dict(row)
                for key in (
                    "accepted_product",
                    "interaction_ids",
                    "interaction_severities",
                    "duplicate_ingredients",
                ):
                    flattened[key] = json.dumps(
                        flattened.get(key), ensure_ascii=False, default=str
                    )
                writer.writerow(flattened)
