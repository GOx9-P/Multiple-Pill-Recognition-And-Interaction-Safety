"""Detected Pill Cards Component with Inline Manual Override."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ..adapters.view_models import PillViewModel


def render_pill_cards(
    pills: list[PillViewModel],
    selected_pill_id: str | None,
    on_pill_selected: Callable[[str], None],
    on_manual_override: Callable[[str, str], None],
) -> None:
    """Render structured cards for all detected pills with XAI evidence and human-in-the-loop overrides."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; font-size: 1rem; color: var(--text-primary);">💊 Danh sách viên thuốc ({len(pills)} viên)</h4>
            <span style="font-size: 0.75rem; color: var(--text-muted);">RAG Pharmaceutical Matcher</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pills:
        st.info("Chưa có viên thuốc nào được phát hiện trong ảnh.")
        return

    for idx, pill in enumerate(pills, start=1):
        is_selected = selected_pill_id is not None and pill.instance_id == selected_pill_id
        active_class = "active" if is_selected else ""

        # Status badge styling
        if pill.is_manual_override:
            badge_class = "manual"
            badge_text = "ĐÃ NHẬP TAY"
        elif pill.status == "accepted":
            badge_class = "accepted"
            badge_text = "ĐÃ XÁC THỰC"
        elif pill.status == "ambiguous":
            badge_class = "ambiguous"
            badge_text = "CHƯA RÕ RÀNG"
        else:
            badge_class = "unresolved"
            badge_text = "CHƯA XÁC ĐỊNH"

        drug_display_name = pill.drug_name or "Chưa thể định danh chắc chắn"
        brand_display = f" ({pill.brand_name})" if pill.brand_name else ""

        # Main Pill Card
        st.markdown(
            f"""
            <div class="pill-card-item {active_class}">
                <div class="pill-card-top">
                    <span class="pill-instance-tag">VIÊN #{idx} • {pill.instance_id}</span>
                    <span class="pill-badge {badge_class}">{badge_text}</span>
                </div>
                <h4 class="pill-drug-name">{drug_display_name}{brand_display}</h4>
                <div class="pill-meta-row"><span>Hình dáng (Shape):</span><span>{pill.shape} ({pill.shape_confidence*100:.0f}%)</span></div>
                <div class="pill-meta-row"><span>Màu sắc (Color):</span><span>{pill.color_primary} ({pill.color_confidence*100:.0f}%)</span></div>
                <div class="pill-meta-row"><span>Ký tự khắc (Imprint):</span><span><code>{pill.imprint_raw}</code></span></div>
                <div class="pill-meta-row"><span>Vạch chia liều:</span><span>{'Có' if pill.scoreline_visible else 'Không'}</span></div>
                {f'<div class="pill-meta-row"><span>Mã RxCUI / NDC:</span><span>{pill.rxcui or "N/A"} • {pill.ndc or "N/A"}</span></div>' if pill.rxcui else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Manual Override Form for Unresolved / Ambiguous Pills
        if pill.status in ("unresolved", "ambiguous") or pill.is_manual_override:
            with st.expander(f"✏️ Nhập bổ sung / Sửa đổi cho viên #{idx} ({pill.instance_id})", expanded=pill.status == "unresolved"):
                st.caption("Nếu ảnh mờ hoặc ký tự khắc bị che khuất, hãy nhập mã khắc hoặc tên thuốc để đối soát CSDL:")
                c1, c2 = st.columns([3, 1])
                with c1:
                    user_input = st.text_input(
                        "Nhập mã Imprint hoặc tên thuốc:",
                        value=pill.drug_name or (pill.imprint_raw if pill.imprint_raw != "—" else ""),
                        key=f"input_override_{pill.instance_id}_{idx}",
                        placeholder="Ví dụ: TV5056, 84A, Aspirin...",
                    )
                with c2:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("Xác nhận", key=f"btn_confirm_override_{pill.instance_id}_{idx}", type="primary"):
                        if user_input.strip():
                            on_manual_override(pill.instance_id, user_input.strip())
                            st.success("Đã ghi nhận và tính toán lại an toàn tương tác!")
                            st.rerun()

        # Expandable Explainable AI (XAI) Evidence Details
        with st.expander(f"📊 Bằng chứng đối soát AI (Evidence) — Viên #{idx}", expanded=False):
            if pill.top_candidates:
                st.markdown("**Ứng viên thuốc xếp hạng cao nhất trong CSDL:**")
                for cand in pill.top_candidates:
                    st.markdown(f"- **Hạng {cand.rank}:** {cand.product_name} — Điểm khớp: `{cand.final_score*100:.1f}%` (Imprint: {cand.imprint_score or 0:.2f}, Shape: {cand.shape_score or 0:.2f}, Color: {cand.color_score or 0:.2f})")
            else:
                st.markdown(f"- **Điểm hình dáng (Shape):** `{pill.shape_confidence or 0:.2f}` ({pill.shape})")
                st.markdown(f"- **Điểm màu sắc (Color):** `{pill.color_confidence or 0:.2f}` ({pill.color_primary})")
                st.markdown(f"- **Độ tin cậy OCR Imprint:** `{pill.imprint_confidence or 0:.2f}`")
                st.markdown(f"- **Ứng viên OCR:** `{', '.join(pill.imprint_candidates) if pill.imprint_candidates else '—'}`")
