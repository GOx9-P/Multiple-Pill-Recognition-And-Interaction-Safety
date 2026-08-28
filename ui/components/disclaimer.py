"""Medical Legal Disclaimer Component."""

from __future__ import annotations

import streamlit as st


def render_disclaimer() -> None:
    """Render the persistent clinical and legal disclaimer."""
    st.markdown(
        """
        <div class="clinical-disclaimer">
            <span style="font-size: 1.25rem;">ℹ️</span>
            <div>
                <strong>Medical disclaimer:</strong>
                This application supports medication identification and safety review. It does <em>not replace</em> diagnosis,
                prescribing advice, or guidance from a qualified doctor or pharmacist. Do not start, stop, change, or combine
                medications based only on this result.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
