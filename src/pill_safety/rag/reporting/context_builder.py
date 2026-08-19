from __future__ import annotations

from typing import Any


class ContextBuilderService:
    def __init__(self) -> None:
        pass

    def build_context(self, context_input: dict[str, Any]) -> dict[str, Any]:
        request_id = context_input.get("request_id")
        session_id = context_input.get("session_id")
        
        cv_output = context_input.get("cv_output") or {}
        rag_identification = context_input.get("rag_identification") or {}
        ddi_output = context_input.get("ddi_output") or {}

        # 1. Extract Unresolved Pills
        unresolved_pills = []
        pill_results = rag_identification.get("pill_results") or []
        for item in pill_results:
            status = item.get("identification_status")
            if status != "identified":
                reasons = item.get("decision_reasons")
                reason_str = ", ".join(reasons) if reasons else "unknown_reason"
                unresolved_pills.append({
                    "instance_id": item.get("instance_id"),
                    "identification_status": status,
                    "reason": reason_str,
                    "required_action": item.get("required_action") or "none"
                })

        # 2. Collect identified drugs, interactions, duplicate warnings from ddi_output
        identified_drugs = ddi_output.get("identified_drugs") or []
        interactions = ddi_output.get("interactions") or []
        duplicate_warnings = ddi_output.get("duplicate_ingredient_warnings") or []

        # 3. Formulate Scope Warnings
        ddi_scope_warnings = ddi_output.get("scope_warnings") or []
        generated_scope_warnings = []
        if len(unresolved_pills) > 0:
            generated_scope_warnings.append("only_identified_drugs_checked")
        generated_scope_warnings.append("no_interaction_found_does_not_mean_safe")

        scope_warnings = list(dict.fromkeys(list(ddi_scope_warnings) + generated_scope_warnings))

        # 4. Extract and deduplicate Sources
        sources_map = {}
        
        # Helper to register source
        def add_source(src_info: dict[str, Any] | None) -> None:
            if not src_info:
                return
            name = src_info.get("source_name")
            ref = src_info.get("source_reference") or ""
            if name:
                key = (name, ref)
                sources_map[key] = {
                    "source_name": name,
                    "source_reference": ref,
                    "last_updated": src_info.get("last_updated") or src_info.get("last_reviewed") or ""
                }

        for drug in identified_drugs:
            add_source(drug.get("source"))

        for inter in interactions:
            add_source(inter.get("source"))

        sources = sorted(list(sources_map.values()), key=lambda x: (x["source_name"], x["source_reference"]))

        return {
            "schema_version": "llm_context_v0",
            "request_id": request_id,
            "session_id": session_id,
            "task": "format_grounded_medication_safety_report",
            "identified_drugs": identified_drugs,
            "unresolved_pills": unresolved_pills,
            "interactions": interactions,
            "duplicate_ingredient_warnings": duplicate_warnings,
            "scope_warnings": scope_warnings,
            "sources": sources
        }
