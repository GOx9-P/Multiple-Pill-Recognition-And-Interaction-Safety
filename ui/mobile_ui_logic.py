"""Pure presentation rules for the consumer-facing mobile medication flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ui.adapters.view_models import InteractionPairViewModel, PillViewModel


@dataclass(frozen=True)
class MobileSeverityContent:
    css_class: str
    eyebrow: str
    title: str
    description: str
    action: str
    symbol: str
    completeness_note: str = ""


@dataclass(frozen=True)
class PillStatusContent:
    css_class: str
    label: str
    symbol: str


@dataclass(frozen=True)
class RecognitionProgress:
    resolved: int
    total: int
    has_unresolved: bool
    label: str


_SEVERITY_CONTENT = {
    "critical": MobileSeverityContent(
        css_class="critical",
        eyebrow="Cần hành động ngay",
        title="Phát hiện tương tác nguy hiểm",
        description="Một cặp thuốc có thể gây hậu quả nghiêm trọng khi dùng cùng nhau.",
        action="Không tự ý dùng cùng lúc hoặc thay đổi thuốc. Hãy liên hệ bác sĩ hoặc dược sĩ.",
        symbol="!",
    ),
    "major": MobileSeverityContent(
        css_class="moderate",
        eyebrow="Cần trao đổi chuyên môn",
        title="Phát hiện tương tác đáng chú ý",
        description="Một số thuốc cần được đánh giá lại cách dùng hoặc liều dùng.",
        action="Dùng thuốc đúng chỉ định và hỏi bác sĩ hoặc dược sĩ trước khi thay đổi.",
        symbol="!",
    ),
    "moderate": MobileSeverityContent(
        css_class="moderate",
        eyebrow="Cần theo dõi",
        title="Có tương tác cần lưu ý",
        description="Một số thuốc có thể ảnh hưởng lẫn nhau hoặc trùng hoạt chất.",
        action="Dùng đúng đơn và hỏi bác sĩ hoặc dược sĩ về cách theo dõi phù hợp.",
        symbol="!",
    ),
    "unresolved": MobileSeverityContent(
        css_class="unresolved",
        eyebrow="Kết quả chưa đầy đủ",
        title="Cần xác nhận thêm thuốc",
        description="Ít nhất một viên chưa được nhận diện đủ chắc chắn.",
        action="Xác nhận các viên chưa rõ trước khi dựa vào kết quả tương tác.",
        symbol="?",
    ),
    "safe": MobileSeverityContent(
        css_class="safe",
        eyebrow="Kết quả hiện tại",
        title="Chưa phát hiện tương tác đã biết",
        description="Không tìm thấy tương tác giữa các thuốc đã nhận diện trong dữ liệu hiện có.",
        action="Tiếp tục dùng theo chỉ định và không tự thay đổi liều hoặc ngừng thuốc.",
        symbol="✓",
    ),
}


def get_mobile_severity_content(severity: str, has_unresolved: bool) -> MobileSeverityContent:
    """Return conservative, action-first copy for a safety result."""
    base = _SEVERITY_CONTENT.get(severity.lower(), _SEVERITY_CONTENT["unresolved"])
    if not has_unresolved or base.css_class == "unresolved":
        return base

    return MobileSeverityContent(
        css_class=base.css_class,
        eyebrow=base.eyebrow,
        title=base.title,
        description=base.description,
        action=base.action,
        symbol=base.symbol,
        completeness_note="Kết quả chưa đầy đủ vì còn viên thuốc cần xác nhận.",
    )


def get_pill_status_content(status: str, is_manual: bool = False) -> PillStatusContent:
    """Translate internal recognition states into plain Vietnamese."""
    if is_manual:
        return PillStatusContent("manual", "Bạn đã xác nhận", "✓")

    normalized = status.lower()
    if normalized == "accepted":
        return PillStatusContent("accepted", "Đã nhận diện", "✓")
    if normalized == "ambiguous":
        return PillStatusContent("ambiguous", "Cần xác nhận", "!")
    return PillStatusContent("unresolved", "Không nhận diện được", "?")


def get_recognition_progress(pills: Iterable[PillViewModel]) -> RecognitionProgress:
    """Summarize how many detected pills are safe to include in DDI evaluation."""
    pill_list = list(pills)
    resolved = sum(1 for pill in pill_list if pill.status == "accepted")
    total = len(pill_list)
    return RecognitionProgress(
        resolved=resolved,
        total=total,
        has_unresolved=resolved != total,
        label=f"Đã nhận diện {resolved}/{total} viên",
    )


def sort_interactions_by_severity(
    interactions: Iterable[InteractionPairViewModel],
) -> list[InteractionPairViewModel]:
    """Put the interactions requiring the fastest action first."""
    priority = {"critical": 0, "contraindicated": 0, "major": 1, "moderate": 2, "minor": 3, "safe": 4}
    return sorted(interactions, key=lambda item: priority.get(item.severity.lower(), 3))


def get_consumer_interaction_action(severity: str) -> str:
    """Return an actionable instruction that never invites self-adjusting medication."""
    if severity.lower() in ("critical", "contraindicated"):
        return "Không tự ý dùng các thuốc này cùng lúc. Hãy liên hệ bác sĩ hoặc dược sĩ sớm nhất có thể."
    return "Tiếp tục dùng đúng chỉ định và hỏi bác sĩ hoặc dược sĩ về cách theo dõi phù hợp."
