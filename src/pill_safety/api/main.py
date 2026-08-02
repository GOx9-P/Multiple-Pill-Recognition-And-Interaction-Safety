from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from pill_safety.database.services.drug_service import DrugService
from pill_safety.database.services.interaction_service import InteractionService
from pill_safety.database.session import get_db


app = FastAPI(title="Medication Safety API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Medication Safety API is running"}


@app.get("/health/database")
def database_health(db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"database": "connected"}


@app.get("/drugs")
def list_drugs(db: Annotated[Session, Depends(get_db)]) -> list[dict]:
    return DrugService(db).list_drugs()


@app.get("/drugs/search")
def search_drugs(
    db: Annotated[Session, Depends(get_db)],
    imprint: Annotated[str | None, Query()] = None,
    shape: Annotated[str | None, Query()] = None,
    color: Annotated[str | None, Query()] = None,
    dosage_form: Annotated[str | None, Query()] = None,
) -> list[dict]:
    return DrugService(db).search(
        imprint=imprint,
        shape=shape,
        color=color,
        dosage_form=dosage_form,
    )


@app.get("/drugs/{drug_id}")
def get_drug(drug_id: int, db: Annotated[Session, Depends(get_db)]) -> dict:
    drug = DrugService(db).get_drug(drug_id)
    if drug is None:
        raise HTTPException(status_code=404, detail="Drug not found")
    return drug


@app.get("/interactions/pair")
def get_interaction_pair(
    db: Annotated[Session, Depends(get_db)],
    ingredient_a_id: Annotated[int, Query()],
    ingredient_b_id: Annotated[int, Query()],
) -> dict:
    interaction = InteractionService(db).find_pair(ingredient_a_id, ingredient_b_id)
    if interaction is None:
        raise HTTPException(status_code=404, detail="No known interaction in current database")
    return interaction
