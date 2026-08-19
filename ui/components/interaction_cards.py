"""Pairwise Drug-Drug Interaction and Duplicate Ingredient Cards."""

from __future__ import annotations

import streamlit as st

from ..adapters.view_models import DuplicateIngredientViewModel, InteractionPairViewModel


def render_interaction_cards(
    interactions: list[InteractionPairViewModel],
    duplicates: list[DuplicateIngredientViewModel],
) -> None:
    """Render pairwise DDI warnings and duplicate active ingredient cards."""
    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-primary);">⚠️ 2. Phân tích tương tác thuốc & Trùng lặp hoạt chất</h3>
            <span style="font-size: 0.8rem; color: var(--text-muted);">NLM / NIH DDI Standard Engine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Render Duplicate Ingredient Overdose Warnings only if present
    if duplicates:
        st.markdown("**🔄 Cảnh Báo Trùng Lặp Hoạt Chất (Nguy cơ quá liều tích lũy):**")
        for dup in duplicates:
            st.markdown(
                f"""
                <div class="ddi-card" style="border-left-color: var(--sev-duplicate);">
                    <div class="ddi-card-header">
                        <span class="ddi-pair-title">🔄 Trùng hoạt chất: {dup.ingredient_name}</span>
                        <span class="pill-badge" style="background: var(--sev-duplicate-bg); color: var(--sev-duplicate-text); border: 1px solid var(--sev-duplicate-border);">QUÁ LIỀU TÍCH LŨY</span>
                    </div>
                    <div class="ddi-section">
                        <strong>Cảnh báo:</strong> {dup.warning}
                    </div>
                    <div class="ddi-section">
                        <strong>Các viên phát hiện:</strong> <code>{', '.join(dup.source_instances)}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Render Pairwise Interactions
    if interactions:
        st.markdown(f"**⚡ Các Cặp Tương Tác Được Ghi Nhận ({len(interactions)} cặp):**")
        for inter in interactions:
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
                        <strong>Mô tả tương tác:</strong> {inter.message}
                    </div>
                    {f'<div class="ddi-section"><strong>Cơ chế dược lý:</strong> {inter.mechanism}</div>' if inter.mechanism else ''}
                    {f'<div class="ddi-section"><strong>Nguy cơ lâm sàng:</strong> {inter.clinical_risk}</div>' if inter.clinical_risk else ''}
                    {f'<div class="ddi-section"><strong>Khuyến cáo xử trí:</strong> {inter.management}</div>' if inter.management else ''}
                    <div class="ddi-section" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">
                        <em>Nguồn thẩm định: {inter.source}</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif not duplicates:
        st.info("✅ Không phát hiện cặp tương tác đối kháng bất lợi nào giữa các thuốc được nhận diện trong cơ sở dữ liệu.")
