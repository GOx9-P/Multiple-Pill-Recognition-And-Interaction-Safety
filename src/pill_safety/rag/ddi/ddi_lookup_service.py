from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from sqlalchemy.orm import Session

from pill_safety.database.models import DrugProduct
from pill_safety.database.services.drug_service import DrugService
from pill_safety.database.services.interaction_service import InteractionService


class DdiLookupService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.drug_service = DrugService(db)
        self.interaction_service = InteractionService(db)

    def lookup_ddi(self, ddi_request: dict[str, Any]) -> dict[str, Any]:
        request_id = ddi_request.get("request_id")
        session_id = ddi_request.get("session_id")
        identified_products = ddi_request.get("identified_products") or []

        identified_drugs = []
        # Maps ingredient_id (int) -> list of instance_id (str)
        ingredient_to_instances: dict[int, list[str]] = {}
        # Maps ingredient_id (int) -> name (str)
        ingredient_names: dict[int, str] = {}
        
        # Parse products and active ingredients
        for item in identified_products:
            if not isinstance(item, dict):
                continue
            instance_id = item.get("instance_id")
            product_id = item.get("product_id")
            if not instance_id or not product_id:
                continue

            drug = self._find_drug(product_id)
            if drug is None:
                # If drug not found, raise value error to ensure safety
                raise ValueError(f"Drug product not found for product_id: {product_id}")

            # Retrieve active ingredients
            active_ingredients_payload = []
            drug_detail = self.drug_service._detail(drug)
            for ing in drug_detail.get("active_ingredients", []):
                ing_id = ing.get("ingredient_id")
                name = ing.get("name")
                if ing_id is None or not name:
                    continue
                
                active_ingredients_payload.append({
                    "ingredient_id": f"ing_{ing_id}",
                    "name": name,
                    "strength": ing.get("strength") or "",
                    "rxcui": ing.get("rxcui") or ""
                })

                if ing_id not in ingredient_to_instances:
                    ingredient_to_instances[ing_id] = []
                ingredient_to_instances[ing_id].append(instance_id)
                ingredient_names[ing_id] = name

            identified_drugs.append({
                "instance_id": instance_id,
                "product_id": f"drug_{drug.drug_id}",
                "product_name": drug.name,
                "brand_name": drug.name,
                "generic_name": drug.generic_name or "",
                "dosage_form": drug.dosage_form or "",
                "route": drug.route or "",
                "ndc": drug.product_code,
                "market": drug.market,
                "active_ingredients": active_ingredients_payload,
                "source": {
                    "source_name": drug.source_name or "DailyMed",
                    "source_reference": drug.source_reference or "",
                    "last_updated": drug.published_date or ""
                }
            })

        # 1. Detect Duplicate Ingredients
        duplicate_warnings = []
        for ing_id, instances in ingredient_to_instances.items():
            if len(instances) >= 2:
                duplicate_warnings.append({
                    "ingredient_id": f"ing_{ing_id}",
                    "ingredient_name": ingredient_names[ing_id],
                    "source_instance_ids": sorted(instances),
                    "severity": "major",
                    "warning": "duplicate_ingredient"
                })

        # 2. Detect Drug-Drug Interactions
        interactions = []
        unique_ing_ids = sorted(list(ingredient_to_instances.keys()))
        pairs = list(combinations(unique_ing_ids, 2))
        
        for ing_a_id, ing_b_id in pairs:
            # get_by_ingredient_pair inside interaction_service handles sorting
            interaction_data = self.interaction_service.find_pair(ing_a_id, ing_b_id)
            if interaction_data is not None:
                # Compile source_instance_ids (union of both ingredient instances)
                instances_a = ingredient_to_instances.get(ing_a_id, [])
                instances_b = ingredient_to_instances.get(ing_b_id, [])
                source_instances = sorted(list(set(instances_a + instances_b)))

                interactions.append({
                    "interaction_id": f"ddi_{interaction_data['interaction_id']}",
                    "ingredient_a_id": f"ing_{ing_a_id}",
                    "ingredient_b_id": f"ing_{ing_b_id}",
                    "ingredient_a_name": ingredient_names[ing_a_id],
                    "ingredient_b_name": ingredient_names[ing_b_id],
                    "source_instance_ids": source_instances,
                    "severity": interaction_data["severity"],
                    "clinical_risk": interaction_data.get("clinical_risk") or "",
                    "mechanism": interaction_data.get("mechanism") or "",
                    "management": interaction_data.get("management") or "",
                    "source": {
                        "source_name": interaction_data.get("source") or "DDInter",
                        "source_reference": interaction_data["source_detail"].get("source_reference") or "",
                        "last_reviewed": interaction_data["source_detail"].get("last_reviewed") or ""
                    }
                })

        # 3. Calculate Overall Severity
        severity_ranks = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1, "none": 0}
        max_rank = 0
        overall_severity = "none"
        for inter in interactions:
            rank = severity_ranks.get(inter["severity"], 0)
            if rank > max_rank:
                max_rank = rank
                overall_severity = inter["severity"]

        if duplicate_warnings and max_rank < severity_ranks["major"]:
            overall_severity = "major"

        return {
            "schema_version": "ddi_output_v0",
            "request_id": request_id,
            "session_id": session_id,
            "identified_drugs": identified_drugs,
            "duplicate_ingredient_warnings": duplicate_warnings,
            "interactions": interactions,
            "overall_severity": overall_severity,
            "scope_warnings": [
                "only_identified_drugs_checked",
                "no_interaction_found_does_not_mean_safe"
            ]
        }

    def _find_drug(self, product_id: Any) -> DrugProduct | None:
        if not product_id:
            return None
        # Try parsing integer from "drug_{id}" or raw string/int
        drug_id = self._parse_drug_id(product_id)
        if drug_id is not None:
            drug = self.drug_service.repository.get_by_id(drug_id)
            if drug is not None:
                return drug
        
        # Fallback: search by product_code (NDC)
        return self.drug_service.repository.get_by_product_code(str(product_id))

    @staticmethod
    def _parse_drug_id(product_id: Any) -> int | None:
        if isinstance(product_id, int):
            return product_id
        if isinstance(product_id, str):
            match = re.match(r"^drug_(\d+)$", product_id.strip())
            if match:
                return int(match.group(1))
            try:
                return int(product_id.strip())
            except ValueError:
                pass
        return None
