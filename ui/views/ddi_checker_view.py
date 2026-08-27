"""Direct Pairwise Drug-Drug Interaction Checker View."""

from __future__ import annotations

import streamlit as st

from ..adapters.pipeline_adapter import KNOWN_DDI_MATRIX
from ..adapters.view_models import InteractionPairViewModel
from ..components.interaction_cards import render_interaction_cards

AVAILABLE_INGREDIENTS = [
    "Clopidogrel",
    "Omeprazole",
    "Aspirin",
    "Lisinopril",
    "Amiodarone",
    "Warfarin",
    "Acetaminophen",
]


def render_ddi_checker_view() -> None:
    """Render the direct pairwise drug-drug interaction checker."""
    st.markdown(
        """
        <div class="clinical-card">
            <div class="card-header-row">
                <h3 class="card-title">Interaction checker</h3>
            </div>
            <p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0;">
                Select two ingredients to review known interaction warnings and recommended next steps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        drug_a = st.selectbox("First ingredient", AVAILABLE_INGREDIENTS, index=0)
    with c2:
        drug_b = st.selectbox("Second ingredient", AVAILABLE_INGREDIENTS, index=1)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    if st.button("Check interaction", type="primary", key="btn_check_pairwise_ddi"):
        pair_a = drug_a.strip().lower()
        pair_b = drug_b.strip().lower()

        if pair_a == pair_b:
            st.warning(f"You selected {drug_a} twice. Review the total dose to avoid accidental duplication.")
            return

        rule = KNOWN_DDI_MATRIX.get((pair_a, pair_b)) or KNOWN_DDI_MATRIX.get((pair_b, pair_a))

        if rule:
            inter_vm = InteractionPairViewModel(
                drug_a_name=drug_a,
                drug_b_name=drug_b,
                severity=rule["severity"],
                message=rule["message"],
                mechanism=rule.get("mechanism", ""),
                clinical_risk=rule.get("clinical_risk", ""),
                management=rule.get("management", ""),
                source=rule.get("source", "NLM Clinical Guidelines"),
            )
            render_interaction_cards([inter_vm], [])
        else:
            st.success(f"No known interaction warning was found for {drug_a} and {drug_b} in the current database.")
