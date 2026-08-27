"""Drug Database Search & Directory View."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..adapters.pipeline_adapter import KNOWN_DRUG_DATABASE


def _fetch_all_drugs_from_db() -> list[dict[str, Any]]:
    """Query all drug products with appearances and ingredients from database."""
    try:
        from pill_safety.database.session import SessionLocal
        from pill_safety.database.models import DrugProduct, DrugAppearance, ProductIngredient, Ingredient
        from sqlalchemy import select

        with SessionLocal() as db:
            stmt = (
                select(DrugProduct, DrugAppearance)
                .outerjoin(DrugAppearance, DrugAppearance.drug_id == DrugProduct.drug_id)
                .where(DrugProduct.active.is_(True))
            )
            rows = db.execute(stmt).all()
            if not rows:
                return []

            results = []
            for prod, app in rows:
                # Get ingredients
                ing_stmt = (
                    select(Ingredient, ProductIngredient)
                    .join(ProductIngredient, ProductIngredient.ingredient_id == Ingredient.ingredient_id)
                    .where(ProductIngredient.drug_id == prod.drug_id)
                )
                ing_rows = db.execute(ing_stmt).all()
                ingredients_str = ", ".join([f"{ing.name} ({pi.strength or ''})" for ing, pi in ing_rows])
                
                results.append({
                    "imprint": (app.imprint if app and app.imprint else prod.name[:6]).upper(),
                    "product_name": prod.name,
                    "brand_name": prod.generic_name,
                    "generic_name": prod.generic_name,
                    "strength": ing_rows[0][1].strength if ing_rows and ing_rows[0][1].strength else "N/A",
                    "rxcui": prod.product_rxcui or "N/A",
                    "ndc": prod.product_code or "N/A",
                    "shape": app.shape if app and app.shape else "N/A",
                    "color": app.color if app and app.color else "N/A",
                    "ingredients": ingredients_str,
                })
            return results
    except Exception:
        return []


def render_drug_search_view() -> None:
    """Render the pharmaceutical database search and directory lookup interface."""
    st.markdown(
        """
        <div class="clinical-card">
            <div class="card-header-row">
                <h3 class="card-title">Medication database</h3>
            </div>
            <p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0;">
                Search medication names, imprints, appearance details, and active ingredients.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search Bar & Filter Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Search by medication name or imprint", placeholder="Examples: Plavix, 84A, Aspirin, TV5056, Omeprazole")
    with c2:
        shape_filter = st.selectbox("Shape", ["All", "ROUND", "OVAL", "CAPSULE", "OBLONG"])
    with c3:
        color_filter = st.selectbox("Color", ["All", "WHITE", "YELLOW", "ORANGE", "PINK", "BLUE"])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Fetch from DB or fallback
    db_drugs = _fetch_all_drugs_from_db()
    
    matches = []
    if db_drugs:
        for drug in db_drugs:
            name_match = (
                not query
                or query.lower() in drug["product_name"].lower()
                or query.lower() in (drug.get("brand_name") or "").lower()
                or query.lower() in drug["imprint"].lower()
                or query.lower() in (drug.get("ingredients") or "").lower()
            )
            shape_match = (shape_filter == "All" or shape_filter.lower() in (drug.get("shape") or "").lower())
            color_match = (color_filter == "All" or color_filter.lower() in (drug.get("color") or "").lower())
            if name_match and shape_match and color_match:
                matches.append((drug["imprint"], drug))
    else:
        for imprint, drug in KNOWN_DRUG_DATABASE.items():
            name_match = (
                not query
                or query.lower() in drug["product_name"].lower()
                or query.lower() in (drug.get("brand_name") or "").lower()
                or query.lower() in imprint.lower()
            )
            if name_match:
                matches.append((imprint, drug))

    st.markdown(f"**Results ({len(matches)} medications):**")

    if not matches:
        st.warning("No medications matched the current search.")
        return

    cols = st.columns(2)
    for idx, (imprint, drug) in enumerate(matches):
        col = cols[idx % 2]
        with col:
            st.markdown(
                f"""
                <div class="pill-card-item">
                    <div class="pill-card-top">
                        <span class="pill-instance-tag">IMPRINT: <code>{imprint}</code></span>
                        <span class="pill-badge accepted">RXNORM</span>
                    </div>
                    <h4 class="pill-drug-name">{drug['product_name']}</h4>
                    <div class="pill-meta-row"><span>Brand</span><span>{drug.get('brand_name') or 'N/A'}</span></div>
                    <div class="pill-meta-row"><span>Generic name</span><span>{drug.get('generic_name') or 'N/A'}</span></div>
                    <div class="pill-meta-row"><span>Strength</span><span>{drug.get('strength') or 'N/A'}</span></div>
                    <div class="pill-meta-row"><span>RxCUI</span><span><code>{drug.get('rxcui') or 'N/A'}</code></span></div>
                    <div class="pill-meta-row"><span>NDC</span><span><code>{drug.get('ndc') or 'N/A'}</code></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
