from __future__ import annotations

from sqlalchemy.orm import Session

from pill_safety.database.repositories.interaction_repository import InteractionRepository


class InteractionService:
    def __init__(self, db: Session) -> None:
        self.repository = InteractionRepository(db)

    def find_pair(self, ingredient_a_id: int, ingredient_b_id: int) -> dict | None:
        interaction = self.repository.get_by_ingredient_pair(ingredient_a_id, ingredient_b_id)
        if interaction is None:
            return None
        return {
            "interaction_id": interaction.interaction_id,
            "ingredient_a_id": interaction.ingredient_a_id,
            "ingredient_b_id": interaction.ingredient_b_id,
            "severity": interaction.severity,
            "description": interaction.description,
            "recommendation": interaction.recommendation,
            "source": interaction.source,
            "clinical_risk": interaction.clinical_risk,
            "mechanism": interaction.mechanism,
            "management": interaction.management,
            "alternative": interaction.alternative,
            "source_detail": {
                "source_name": interaction.source_name,
                "source_reference": interaction.source_reference,
                "source_level": interaction.source_level,
                "evidence_text": interaction.evidence_text,
                "last_reviewed": interaction.last_reviewed,
            },
        }
