from __future__ import annotations

from ui.adapters.pipeline_adapter import evaluate_safety_and_report
from ui.adapters.view_models import InteractionPairViewModel, PillViewModel
from ui.mobile_ui_logic import (
    get_consumer_interaction_action,
    get_mobile_severity_content,
    get_pill_status_content,
    get_recognition_progress,
    sort_interactions_by_severity,
)


def _pill(instance_id: str, status: str = "accepted") -> PillViewModel:
    return PillViewModel(
        instance_id=instance_id,
        status=status,
        shape="Round",
        shape_confidence=0.9,
        color_primary="White",
        color_secondary=None,
        color_confidence=0.9,
        imprint_raw="?",
        imprint_confidence=0.2,
    )


def test_safe_copy_does_not_claim_the_regimen_is_absolutely_safe() -> None:
    content = get_mobile_severity_content("safe", has_unresolved=False)

    assert content.title == "Chưa phát hiện tương tác đã biết"
    assert "an toàn" not in content.description.lower()
    assert "dữ liệu hiện có" in content.description.lower()


def test_critical_copy_discloses_when_recognition_is_incomplete() -> None:
    content = get_mobile_severity_content("critical", has_unresolved=True)

    assert content.css_class == "critical"
    assert "chưa đầy đủ" in content.completeness_note.lower()
    assert "bác sĩ hoặc dược sĩ" in content.action.lower()


def test_pill_statuses_are_presented_in_plain_vietnamese() -> None:
    assert get_pill_status_content("accepted").label == "Đã nhận diện"
    assert get_pill_status_content("ambiguous").label == "Cần xác nhận"
    assert get_pill_status_content("unresolved").label == "Không nhận diện được"
    assert get_pill_status_content("accepted", is_manual=True).label == "Bạn đã xác nhận"


def test_recognition_progress_counts_only_resolved_pills() -> None:
    pills = [_pill("pill_1"), _pill("pill_2", "ambiguous"), _pill("pill_3", "unresolved")]

    progress = get_recognition_progress(pills)

    assert progress.resolved == 1
    assert progress.total == 3
    assert progress.has_unresolved is True
    assert progress.label == "Đã nhận diện 1/3 viên"


def test_interactions_are_sorted_from_highest_to_lowest_risk() -> None:
    interactions = [
        InteractionPairViewModel("A", "B", "minor", "minor"),
        InteractionPairViewModel("C", "D", "critical", "critical"),
        InteractionPairViewModel("E", "F", "moderate", "moderate"),
        InteractionPairViewModel("G", "H", "major", "major"),
    ]

    result = sort_interactions_by_severity(interactions)

    assert [item.severity for item in result] == ["critical", "major", "moderate", "minor"]


def test_consumer_interaction_action_never_tells_the_user_to_adjust_a_dose() -> None:
    action = get_consumer_interaction_action("critical")

    assert "bác sĩ hoặc dược sĩ" in action.lower()
    assert "giảm liều" not in action.lower()
    assert "tăng liều" not in action.lower()


def test_unknown_manual_override_remains_unresolved_and_cannot_produce_safe_result() -> None:
    pill = _pill("pill_unknown", "unresolved")

    report = evaluate_safety_and_report(
        pills=[pill],
        manual_overrides={"pill_unknown": "thuoc khong co trong du lieu"},
    )

    assert pill.status == "unresolved"
    assert pill.is_manual_override is False
    assert report.overall_severity == "unresolved"


def test_known_manual_override_is_accepted() -> None:
    pill = _pill("pill_known", "unresolved")

    report = evaluate_safety_and_report(
        pills=[pill],
        manual_overrides={"pill_known": "84A"},
    )

    assert pill.status == "accepted"
    assert pill.is_manual_override is True
    assert report.overall_severity == "safe"


def test_known_generic_name_manual_override_is_accepted() -> None:
    pill = _pill("pill_known_name", "unresolved")

    evaluate_safety_and_report(
        pills=[pill],
        manual_overrides={"pill_known_name": "Aspirin"},
    )

    assert pill.status == "accepted"
    assert pill.generic_name == "Aspirin"
