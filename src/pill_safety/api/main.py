from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from pill_safety.database.services.drug_service import DrugService
from pill_safety.database.services.interaction_service import InteractionService
from pill_safety.database.session import get_db
from pill_safety.rag.identification_service import IdentificationService
from pill_safety.rag.ddi.ddi_lookup_service import DdiLookupService
from pill_safety.rag.reporting.context_builder import ContextBuilderService
from pill_safety.rag.reporting.llm_report_generator import LlmReportGenerator
from pill_safety.schemas.rag import (
    RagIdentifyRequest,
    DdiRequest,
    ContextBuilderInput,
    RagReportRequest,
    ManualIdentifyRequest,
)


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


@app.post("/rag/identify")
def identify_pills(
    rag_request: RagIdentifyRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    try:
        return IdentificationService(db).identify(rag_request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/rag/ddi")
def lookup_ddi(
    ddi_request: DdiRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    try:
        return DdiLookupService(db).lookup_ddi(ddi_request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/rag/context")
def build_context(
    context_input: ContextBuilderInput,
) -> dict:
    try:
        return ContextBuilderService().build_context(context_input.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/rag/report")
def generate_report(
    report_request: RagReportRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    try:
        req_dict = report_request.model_dump()
        cv_output = req_dict.get("cv_output") or {}
        rag_id = req_dict.get("rag_identification")
        if not rag_id:
            rag_id = IdentificationService(db).identify(req_dict)

        ddi_out = req_dict.get("ddi_output")
        if not ddi_out:
            identified_products = []
            for pill_res in rag_id.get("pill_results", []):
                acc = pill_res.get("accepted_product")
                if acc and acc.get("drug_id"):
                    identified_products.append({
                        "instance_id": pill_res.get("instance_id"),
                        "product_id": f"drug_{acc['drug_id']}"
                    })
            if identified_products:
                ddi_out = DdiLookupService(db).lookup_ddi({
                    "request_id": report_request.request_id,
                    "session_id": report_request.session_id,
                    "identified_products": identified_products
                })
            else:
                ddi_out = {
                    "request_id": report_request.request_id,
                    "session_id": report_request.session_id,
                    "identified_drugs": [],
                    "duplicate_ingredient_warnings": [],
                    "interactions": [],
                    "overall_severity": "none",
                    "scope_warnings": []
                }

        context = ContextBuilderService().build_context({
            "request_id": report_request.request_id,
            "session_id": report_request.session_id,
            "cv_output": cv_output,
            "rag_identification": rag_id,
            "ddi_output": ddi_out
        })

        return LlmReportGenerator().generate_report(context)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/rag/manual-identify")
def manual_identify(
    manual_request: ManualIdentifyRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    try:
        id_service = IdentificationService(db)
        bound = id_service.manual_bind_unresolved_pill(
            instance_id=manual_request.instance_id,
            manual_drug_name=manual_request.manual_drug_name,
            product_id=manual_request.product_id,
        )
        return {
            "status": "success",
            "session_id": manual_request.session_id,
            "bound_pill": bound,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

