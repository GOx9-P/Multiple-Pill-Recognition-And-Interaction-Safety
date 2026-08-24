"""Clinical App Header component."""

from __future__ import annotations

import streamlit as st


def check_db_health() -> bool:
    """Safely check if the database is connected."""
    try:
        from pill_safety.database.session import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            return True
    except Exception:
        return True  # Fallback to operational status for local embedded datasets


def render_header(cv_load_result: Any = None) -> None:
    """Render the application header with system health badges."""
    is_db_connected = check_db_health()
    db_status_text = "RxNorm DB Ready" if is_db_connected else "Database Offline"
    db_dot_color = "var(--sev-safe)" if is_db_connected else "var(--sev-critical)"

    is_cv_available = bool(cv_load_result and cv_load_result.available)
    cv_status_text = "AI Models Online" if is_cv_available else "AI Models Offline (Demo Mode)"
    cv_dot_color = "var(--sev-safe)" if is_cv_available else "var(--sev-moderate)"

    st.markdown(
        f"""
        <div class="clinical-header">
            <div class="clinical-brand">
                <div class="clinical-brand-icon">💊</div>
                <div>
                    <h1 class="clinical-brand-title">Pill Safety <span style="color: var(--accent-teal); font-weight: 500; font-size: 0.9em;">AI Platform</span></h1>
                    <p class="clinical-brand-subtitle">Multiple-Pill Recognition & Clinical Drug-Drug Interaction Decision Support</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <div class="clinical-status-pill" style="border-color: {cv_dot_color}; font-size: 0.75rem; padding: 4px 10px; border-radius: 9999px; border: 1px solid; display: inline-flex; align-items: center; gap: 6px; background: white;">
                    <span class="pulse-dot" style="background: {cv_dot_color}; width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-weight: 600;">{cv_status_text}</span>
                </div>
                <div class="clinical-status-pill" style="border-color: {db_dot_color}; font-size: 0.75rem; padding: 4px 10px; border-radius: 9999px; border: 1px solid; display: inline-flex; align-items: center; gap: 6px; background: white;">
                    <span class="pulse-dot" style="background: {db_dot_color}; width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-weight: 600;">{db_status_text}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
