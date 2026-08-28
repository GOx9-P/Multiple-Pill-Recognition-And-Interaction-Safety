"""About and Methodology View."""

from __future__ import annotations

import streamlit as st


def render_about_view() -> None:
    """Render a user-facing explanation of the medication safety workflow."""
    st.markdown(
        """
        <div class="clinical-card">
            <div class="card-header-row">
                <h3 class="card-title">How it works</h3>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                The app turns a medication photo into a structured safety review. It keeps uncertain results separate from confirmed matches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Pipeline Stages Cards Grid
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">1. Find medications</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
The photo is separated into individual medication regions so each item can be reviewed independently.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">2. Read visible characteristics</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Visible shape and color are assessed with confidence estimates.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">3. Read the imprint</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Visible imprint text and score lines are collected as supporting evidence.
</p>
</div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">4. Combine evidence</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
The available visual evidence is combined before any medication name is shown.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">5. Match possible medications</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Medication candidates are compared against the available database. Low-confidence matches remain unresolved for review.
</p>
</div>
<div class="pill-card-item">
<h4 style="margin: 0 0 0.4rem 0; color: var(--accent-brand); font-size: 0.95rem;">6. Screen for safety concerns</h4>
<p style="font-size: 0.825rem; color: var(--text-secondary); margin: 0; line-height: 1.45;">
Confirmed medications are checked for interaction warnings and duplicate ingredients before a safety summary is generated.
</p>
</div>""",
            unsafe_allow_html=True,
        )

    # 2. Data Sources & Clinical Ethics
    st.markdown(
        """<div class="clinical-card" style="margin-top: 0.75rem;">
<h4 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-size: 1rem;">Data sources</h4>
<ul style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin: 0 0 1rem 1.25rem; padding: 0;">
<li><strong>RxNorm:</strong> Standard medication identifiers and ingredient terminology.</li>
<li><strong>DailyMed:</strong> Medication label information and appearance references.</li>
<li><strong>Interaction references:</strong> Curated warnings used to screen known medication combinations.</li>
</ul>

<h4 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-size: 1rem;">Safety boundary</h4>
<p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; margin: 0;">
When the image or imprint evidence is not reliable enough, the medication remains <code>unresolved</code>. Confirm the medication manually with a pharmacist or healthcare professional before relying on interaction results.
</p>
</div>""",
        unsafe_allow_html=True,
    )
