from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ui.adapters import pipeline_adapter
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


def test_unmatched_real_cv_output_returns_ambiguous_pill_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_adapter,
        "_find_drug_in_database",
        lambda *_args, **_kwargs: None,
    )
    raw_cv_data = SimpleNamespace(
        image_quality=SimpleNamespace(
            status="usable",
            blur_score=0.05,
            glare_detected=False,
            lighting_warning=False,
        ),
        pills=[
            SimpleNamespace(
                instance_id="pill_001",
                shape=SimpleNamespace(label="round", confidence=0.91),
                color=SimpleNamespace(primary="white", secondary=None, confidence=0.88),
                imprint=SimpleNamespace(raw="ZX999", confidence=0.77),
                scoreline=SimpleNamespace(visible=False, confidence=0.0),
                bbox_xyxy=[10, 20, 80, 90],
                mask_path="outputs/masks/pill_001.png",
                crop_path="outputs/crops/pill_001.png",
            )
        ],
    )

    pills, quality = pipeline_adapter.parse_cv_output(raw_cv_data)

    assert quality.status == "usable"
    assert len(pills) == 1
    assert pills[0].status == "ambiguous"
    assert pills[0].drug_name is None
    assert pills[0].rxcui is None
    assert pills[0].ndc is None


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
