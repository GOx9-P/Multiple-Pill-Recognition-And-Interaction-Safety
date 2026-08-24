from __future__ import annotations

from pathlib import Path

from ui.adapters.pipeline_adapter import evaluate_safety_and_report, parse_cv_output
from ui.demo_data import get_preset_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_empty_demo_scenario_represents_no_detected_pills() -> None:
    scenario = get_preset_scenario("empty")

    assert scenario["image_quality"]["status"] == "good"
    assert scenario["pills"] == []


def test_duplicate_ingredient_demo_produces_an_overdose_warning() -> None:
    pills, _ = parse_cv_output(get_preset_scenario("duplicate"))

    report = evaluate_safety_and_report(pills)

    assert len(report.duplicate_warnings) == 1
    assert report.duplicate_warnings[0].ingredient_name == "Acetaminophen"


def test_mobile_analysis_uses_pending_demo_cv_data_without_running_models() -> None:
    source = (
        PROJECT_ROOT / "ui" / "views" / "mobile" / "mobile_analyze_view.py"
    ).read_text(encoding="utf-8")

    assert "preset_data: dict[str, Any] | None = None" in source
    assert "st.session_state.raw_cv_data = preset_data" in source


def test_demo_picker_is_secondary_and_collapsed_by_default() -> None:
    source = (
        PROJECT_ROOT / "ui" / "views" / "mobile" / "mobile_analyze_view.py"
    ).read_text(encoding="utf-8")

    assert 'with st.expander("Dùng dữ liệu mẫu để xem nhanh", expanded=False)' in source
