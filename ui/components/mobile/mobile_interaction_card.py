"""Mobile Clinical Drug-Drug Interaction (DDI) Card Component."""

from __future__ import annotations

from html import escape
from typing import Any
import streamlit as st
from ui.mobile_ui_logic import get_consumer_interaction_action


def render_mobile_interaction_card(interaction: Any, index: int = 1) -> None:
    """Render a mobile-friendly drug interaction card with clinical action first."""
    if hasattr(interaction, "drug_a_name"):
        drug_a = interaction.drug_a_name
        drug_b = interaction.drug_b_name
        severity = interaction.severity.lower()
        effect = getattr(interaction, "clinical_risk", "") or getattr(interaction, "message", "")
        professional_guidance = getattr(interaction, "management", "")
        mechanism = getattr(interaction, "mechanism", "")
        source = getattr(interaction, "source", "")
    elif isinstance(interaction, dict):
        drug_a = interaction.get("drug_a", "")
        drug_b = interaction.get("drug_b", "")
        severity = interaction.get("severity", "moderate").lower()
        effect = interaction.get("clinical_effect", "")
        professional_guidance = interaction.get("recommendation", "")
        mechanism = interaction.get("mechanism", "")
        source = interaction.get("source", "")
    else:
        return

    sev_class = "critical" if severity in ("critical", "contraindicated") else "moderate"
    sev_badge_text = "Nguy hiểm cao" if sev_class == "critical" else "Cần theo dõi"
    action = get_consumer_interaction_action(severity)

    # Build dense HTML without empty lines to prevent markdown codeblock interpretation
    html = (
        f'<div class="mobile-ddi-card {sev_class}">'
        f'<div class="mobile-ddi-header"><span class="mobile-ddi-badge {sev_class}">{sev_badge_text}</span></div>'
        f'<div class="mobile-ddi-pair">{escape(drug_a)} <span aria-hidden="true">×</span> {escape(drug_b)}</div>'
        f'<div class="mobile-ddi-action"><strong>Bạn cần làm gì</strong>{escape(action)}</div>'
        f'<div class="mobile-ddi-section"><div class="mobile-ddi-label">Điều có thể xảy ra</div>'
        f'<div class="mobile-ddi-text">{escape(effect)}</div></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    if mechanism or professional_guidance or source:
        with st.expander(f"Cơ chế và nguồn: {drug_a} + {drug_b}", expanded=False):
            if professional_guidance:
                st.markdown("**Hướng dẫn chuyên môn:**")
                st.write(professional_guidance)
            if mechanism:
                st.markdown("**Cơ chế:**")
                st.write(mechanism)
            if source:
                st.caption(f"Nguồn: {source}")
