"""제외 규칙 확장 포인트.

현재는 **아무것도 제외하지 않는다** (학습데이터 구축 단계이므로 recall 최우선).

나중에 운영 단계에서 과검을 줄여야 할 때 이 파일만 수정하면 된다.
파이프라인은 ``should_exclude()`` 를 최종 병합 직전에 호출하므로,
여기에 조건을 추가하면 즉시 반영된다.

예시 (지금은 비활성):

    NEVER_MASK_PATTERNS = [
        re.compile(r"[\\d,]+\\s*원"),          # 금액
        re.compile(r"연\\s*[\\d.]+\\s*%"),      # 이자율
        re.compile(r"20\\d{2}[.\\-년]"),        # 약정일/만기일
    ]

    def should_exclude(region, boxes):
        if region.source is Source.RULE:
            return None                      # 체크섬 통과 항목은 절대 제외하지 않는다
        text = region.text or ""
        for pat in NEVER_MASK_PATTERNS:
            if pat.search(text):
                return "금액/이자율/날짜 필드"
        return None
"""

from __future__ import annotations

from .schema import OcrBox, PiiRegion

__all__ = ["should_exclude"]


def should_exclude(region: PiiRegion, boxes: list[OcrBox]) -> str | None:
    """영역을 결과에서 제외할지 판단한다.

    Args:
        region: 병합 직전의 탐지 영역.
        boxes: 페이지의 전체 OCR 박스 (문맥 판단용).

    Returns:
        제외할 사유 문자열, 또는 제외하지 않으면 ``None``.

    Note:
        기본 구현은 항상 ``None`` 을 반환한다. 즉 아무것도 제외하지 않는다.
        ``Source.RULE`` 항목은 체크섬을 통과한 확정 건이므로, 규칙을 추가할 때도
        제외 대상에서 빼두는 것을 권한다.
    """
    return None
