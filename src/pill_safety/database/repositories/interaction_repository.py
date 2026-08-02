from __future__ import annotations

from itertools import combinations

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from pill_safety.database.models import DrugInteraction


class InteractionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_ingredient_pair(self, ingredient_a_id: int, ingredient_b_id: int) -> DrugInteraction | None:
        first_id, second_id = sorted((ingredient_a_id, ingredient_b_id))
        statement = select(DrugInteraction).where(
            DrugInteraction.ingredient_a_id == first_id,
            DrugInteraction.ingredient_b_id == second_id,
        )
        return self.db.scalars(statement).first()

    def get_between_ingredients(self, ingredient_ids: list[int]) -> list[DrugInteraction]:
        unique_ids = sorted(set(ingredient_ids))
        pairs = list(combinations(unique_ids, 2))
        if not pairs:
            return []

        conditions = [
            and_(
                DrugInteraction.ingredient_a_id == first_id,
                DrugInteraction.ingredient_b_id == second_id,
            )
            for first_id, second_id in pairs
        ]
        statement = select(DrugInteraction).where(or_(*conditions)).order_by(DrugInteraction.severity)
        return list(self.db.scalars(statement).all())

