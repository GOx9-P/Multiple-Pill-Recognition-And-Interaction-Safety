"""Pairwise Drug-Drug Interaction and Duplicate Ingredient Cards."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ..adapters.view_models import DuplicateIngredientViewModel, InteractionPairViewModel


def render_interaction_cards(
    interactions: list[InteractionPairViewModel],
    duplicates: list[DuplicateIngredientViewModel],
    on_interaction_selected: Callable[[int], None] | None = None,
    active_interaction_index: int | None = None,
) -> None:
    """Render pairwise DDI warnings and duplicate active ingredient cards."""
    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">Medication safety findings</h3>
            <span style="font-size: 0.8rem; color: var(--text-muted);">Review identified medications before use</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Render Duplicate Ingredient Overdose Warnings only if present
    if duplicates:
        st.markdown("**Duplicate ingredient warnings:**")
        for dup in duplicates:
            st.markdown(
                f"""
                <div class="ddi-card" style="border-left-color: var(--sev-duplicate);">
                    <div class="ddi-card-header">
                        <span class="ddi-pair-title">Duplicate ingredient: {dup.ingredient_name}</span>
                        <span class="pill-badge" style="background: var(--sev-duplicate-bg); color: var(--sev-duplicate-text); border: 1px solid var(--sev-duplicate-border);">DUPLICATE</span>
                    </div>
                    <div class="ddi-section">
                        <strong>Finding:</strong> {dup.warning}
                    </div>
                    <div class="ddi-section">
                        <strong>Related medications:</strong> <code>{', '.join(dup.source_instances)}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Render Pairwise Interactions
    if interactions:
        st.markdown(f"**Potential medication interactions ({len(interactions)}):**")
        for index, inter in enumerate(interactions):
            severity_class = inter.severity.lower()
            badge_color = "var(--sev-critical-text)" if severity_class == "critical" else "var(--sev-moderate-text)"
            badge_bg = "var(--sev-critical-bg)" if severity_class == "critical" else "var(--sev-moderate-bg)"
            badge_border = "var(--sev-critical-border)" if severity_class == "critical" else "var(--sev-moderate-border)"

            st.markdown(
                f"""
                <div class="ddi-card {severity_class}">
                    <div class="ddi-card-header">
                        <span class="ddi-pair-title">⚡ {inter.drug_a_name} + {inter.drug_b_name}</span>
                        <span class="pill-badge" style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border};">{inter.severity.upper()}</span>
                    </div>
                    <div class="ddi-section">
                        <strong>Finding:</strong> {inter.message}
                    </div>
                    {f'<div class="ddi-section"><strong>Mechanism:</strong> {inter.mechanism}</div>' if inter.mechanism else ''}
                    {f'<div class="ddi-section"><strong>Clinical risk:</strong> {inter.clinical_risk}</div>' if inter.clinical_risk else ''}
                    {f'<div class="ddi-section"><strong>Recommended action:</strong> {inter.management}</div>' if inter.management else ''}
                    <div class="ddi-section" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">
                        <em>Reference: {inter.source}</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if on_interaction_selected is not None and inter.source_instances:
                st.button(
                    "Show medications in photo",
                    key=f"show_interaction_photo_{index}",
                    on_click=on_interaction_selected,
                    args=(index,),
                )
    elif not duplicates:
        st.info("No harmful interaction pair was found among the identified medications in the current database.")
