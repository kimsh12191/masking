"""규칙 레이어: 정규식 + 체크섬으로 정형 식별자를 확정한다.

LLM 이전에 실행되며, 여기서 확정된 박스는 프롬프트에 ``<CONFIRMED:타입>`` 으로
표시되어 LLM 이 중복 판단하지 않는다.

정책: **recall 우선**. 체크섬이 없는 항목(계좌 등)은 문맥 키워드로 넓게 잡고
``needs_review`` 로 넘긴다. 학습데이터 구축 단계이므로 과검은 허용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import OcrBox, OcrStatus
from .checksums import VALIDATORS, digits_only

# --------------------------------------------------------------------------
# 패턴 정의
# --------------------------------------------------------------------------

# 구분자는 하이픈/공백/점 모두 허용 (OCR 이 하이픈을 놓치는 경우가 흔하다)
_SEP = r"[\-–—.\s]?"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 주민등록번호 / 외국인등록번호 (앞 6 + 뒤 7)
    ("RRN", re.compile(rf"(?<!\d)(\d{{6}}){_SEP}(\d{{7}})(?!\d)")),
    # 법인등록번호 6-7 (주민번호와 형태가 같아 체크섬으로 구분)
    # 사업자등록번호 3-2-5
    ("BIZ_NO", re.compile(rf"(?<!\d)(\d{{3}}){_SEP}(\d{{2}}){_SEP}(\d{{5}})(?!\d)")),
    # 카드번호 4-4-4-4 (15자리 AMEX 포함)
    (
        "CARD_NO",
        re.compile(rf"(?<!\d)(\d{{4}}){_SEP}(\d{{4}}){_SEP}(\d{{4}}){_SEP}(\d{{3,4}})(?!\d)"),
    ),
    # 휴대전화
    ("PHONE", re.compile(rf"(?<!\d)(01[016789]){_SEP}(\d{{3,4}}){_SEP}(\d{{4}})(?!\d)")),
    # 유선전화 (지역번호 02 또는 3자리)
    ("PHONE", re.compile(rf"(?<!\d)(0\d{{1,2}}){_SEP}(\d{{3,4}})[\-–—](\d{{4}})(?!\d)")),
    # 이메일
    (
        "EMAIL",
        re.compile(r"[A-Za-z0-9._%+\-]+\s?@\s?[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ),
    # 여권번호 (M12345678 형태 / 구형 숫자 9자리는 오탐이 많아 제외)
    ("PASSPORT", re.compile(r"(?<![A-Za-z0-9])([MSRODmsrod])(\d{8})(?![A-Za-z0-9])")),
    # 운전면허번호 11-12-345678-01 또는 지역명 포함형
    (
        "DRIVER_LICENSE",
        re.compile(rf"(?<!\d)(\d{{2}}){_SEP}(\d{{2}}){_SEP}(\d{{6}}){_SEP}(\d{{2}})(?!\d)"),
    ),
]

#: 계좌번호는 표준 체크섬이 없다. 문맥 키워드가 같은 행/근처에 있을 때만 잡는다.
ACCOUNT_PATTERN = re.compile(rf"(?<!\d)(\d{{2,6}}){_SEP}(\d{{2,6}}){_SEP}(\d{{2,7}})(?!\d)")

ACCOUNT_KEYWORDS = (
    "계좌", "예금", "입금", "출금", "송금", "자동이체", "이체", "예금주",
    "가상계좌", "환불계좌", "수령계좌", "은행",
)

#: 항목명(라벨) 텍스트. 값이 아니라 필드명이므로 개인정보가 아니다.
FIELD_LABEL_WORDS = (
    "성명", "이름", "주민등록번호", "주민번호", "생년월일", "주소", "연락처",
    "전화번호", "휴대전화", "이메일", "전자우편", "직장명", "소속", "직위",
    "사업자등록번호", "법인등록번호", "계좌번호", "카드번호", "여권번호",
    "운전면허번호", "예금주", "서명", "날인", "귀하", "신청인", "대표자",
)


@dataclass
class RuleHit:
    """규칙 레이어 탐지 결과."""

    box_index: int
    label: str
    text: str
    span: tuple[int, int]          # 박스 텍스트 내 문자 오프셋
    checksum_ok: bool
    needs_review: bool = False


# --------------------------------------------------------------------------
# 판별 헬퍼
# --------------------------------------------------------------------------


def is_field_label(text: str) -> bool:
    """항목명(필드 라벨)인지 판별. 값이 붙어 있으면 라벨로 보지 않는다."""
    stripped = re.sub(r"[\s:：()\[\]]", "", text)
    if not stripped:
        return True
    return any(stripped == word for word in FIELD_LABEL_WORDS)


def _classify_13digit(raw: str) -> tuple[str, bool] | None:
    """13자리 숫자를 RRN / FOREIGN_ID / CORP_NO 중 하나로 분류한다."""
    d = digits_only(raw)
    if len(d) != 13:
        return None

    gender = d[6]
    # 외국인등록번호: 성별코드 5~8
    if gender in "5678" and VALIDATORS["FOREIGN_ID"](d):
        return ("FOREIGN_ID", VALIDATORS["RRN"](d))
    if VALIDATORS["RRN"](d):
        return ("RRN", True)
    if VALIDATORS["CORP_NO"](d):
        return ("CORP_NO", True)
    # 체크섬 실패. OCR 오독 가능성이 높으므로 recall 우선으로 RRN 후보로 남긴다.
    return ("RRN", False)


def _row_context(boxes: list[OcrBox], target: OcrBox, window: int = 2) -> str:
    """같은 행 및 인접 행의 텍스트를 모아 문맥 문자열로 반환한다."""
    parts: list[str] = []
    for b in boxes:
        if abs(b.row - target.row) <= window:
            parts.append(b.text)
    return " ".join(parts)


# --------------------------------------------------------------------------
# 메인 진입점
# --------------------------------------------------------------------------


def detect(boxes: list[OcrBox]) -> list[RuleHit]:
    """OCR 박스 목록에서 정형 식별자를 탐지한다.

    Args:
        boxes: ``row`` 가 채워진 (읽기순서 정렬된) OCR 박스 목록.

    Returns:
        박스 인덱스별 탐지 결과. 한 박스에서 여러 건이 나올 수 있다.
    """
    hits: list[RuleHit] = []

    for box in boxes:
        if box.status is OcrStatus.FAILED or not box.text.strip():
            continue

        text = box.text
        claimed: list[tuple[int, int]] = []

        for label, pattern in PATTERNS:
            for m in pattern.finditer(text):
                span = m.span()
                # 이미 다른(더 앞선) 패턴이 점유한 구간은 건너뛴다
                if any(span[0] < c[1] and c[0] < span[1] for c in claimed):
                    continue

                raw = m.group(0)
                resolved_label = label
                checksum_ok = True

                if label == "RRN":
                    verdict = _classify_13digit(raw)
                    if verdict is None:
                        continue
                    resolved_label, checksum_ok = verdict
                elif label in VALIDATORS:
                    checksum_ok = VALIDATORS[label](raw)
                    if not checksum_ok and label in ("BIZ_NO", "CARD_NO", "DRIVER_LICENSE"):
                        # 체크섬 실패 + 자유형 숫자열은 오탐이 많다.
                        # 문맥 키워드가 있으면 후보로 유지, 없으면 버린다.
                        ctx = _row_context(boxes, box)
                        keyword = {
                            "BIZ_NO": ("사업자", "등록번호"),
                            "CARD_NO": ("카드", "카드번호"),
                            "DRIVER_LICENSE": ("면허",),
                        }[label]
                        if not any(k in ctx for k in keyword):
                            continue

                claimed.append(span)
                hits.append(
                    RuleHit(
                        box_index=box.index,
                        label=resolved_label,
                        text=raw,
                        span=span,
                        checksum_ok=checksum_ok,
                        needs_review=not checksum_ok,
                    )
                )

        # 계좌번호: 문맥 키워드 필요
        ctx = _row_context(boxes, box)
        if any(k in ctx for k in ACCOUNT_KEYWORDS):
            for m in ACCOUNT_PATTERN.finditer(text):
                span = m.span()
                if any(span[0] < c[1] and c[0] < span[1] for c in claimed):
                    continue
                d = digits_only(m.group(0))
                if not 10 <= len(d) <= 16:
                    continue
                claimed.append(span)
                hits.append(
                    RuleHit(
                        box_index=box.index,
                        label="ACCOUNT_NO",
                        text=m.group(0),
                        span=span,
                        checksum_ok=False,
                        needs_review=True,  # 체크섬이 없으므로 항상 검토 대상
                    )
                )

    return hits
