"""Mobile Clinical Drug-Drug Interaction (DDI) Card Component."""

from __future__ import annotations

from typing import Any
import streamlit as st


def render_mobile_interaction_card(interaction: Any, index: int = 1) -> None:
    """Render a mobile-friendly drug interaction card with clinical action first."""
    if hasattr(interaction, "drug_a_name"):
        drug_a = interaction.drug_a_name
        drug_b = interaction.drug_b_name
        severity = interaction.severity.lower()
        effect = getattr(interaction, "clinical_risk", "") or getattr(interaction, "message", "")
        action = getattr(interaction, "management", "") or "Tránh dùng đồng thời."
        mechanism = getattr(interaction, "mechanism", "")
    elif isinstance(interaction, dict):
        drug_a = interaction.get("drug_a", "")
        drug_b = interaction.get("drug_b", "")
        severity = interaction.get("severity", "moderate").lower()
        effect = interaction.get("clinical_effect", "")
        action = interaction.get("recommendation", "")
        mechanism = interaction.get("mechanism", "")
    else:
        return

    sev_class = "critical" if severity == "critical" else ("moderate" if severity == "moderate" else "safe")
    sev_badge_text = "NGUY HIỂM CAO (CRITICAL)" if severity == "critical" else ("CẦN THEO DÕI (MODERATE)" if severity == "moderate" else severity.upper())

    # Build dense HTML without empty lines to prevent markdown codeblock interpretation
    html = (
        f'<div class="mobile-ddi-card {sev_class}">'
        f'<div class="mobile-ddi-header"><span class="mobile-ddi-badge {sev_class}">{sev_badge_text}</span></div>'
        f'<div class="mobile-ddi-pair">{drug_a} <span style="color: var(--text-muted);">×</span> {drug_b}</div>'
        f'<div class="mobile-ddi-section"><div class="mobile-ddi-label">⚠️ Hậu quả lâm sàng:</div><div class="mobile-ddi-text">{effect}</div></div>'
        f'<div class="mobile-ddi-section" style="background: var(--bg-surface-elevated); padding: 8px 10px; border-radius: var(--radius-sm); margin-top: 6px;">'
        f'<div class="mobile-ddi-label" style="color: var(--accent-brand);">💡 Khuyến nghị xử trí:</div><div class="mobile-ddi-text" style="font-weight: 600;">{action}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    if mechanism:
        with st.expander(f"🔬 Cơ chế dược lý ({drug_a} + {drug_b})", expanded=False):
            st.markdown(f"<div style='font-size: 0.825rem; line-height: 1.5; color: var(--text-secondary);'>{mechanism}</div>", unsafe_allow_html=True)
