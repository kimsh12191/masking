"""규칙 레이어: 정규식 + 체크섬으로 정형 식별자를 확정한다.

LLM 이전에 실행되며, 여기서 확정된 박스는 프롬프트에 ``<CONFIRMED:타입>`` 으로
표시되어 LLM 이 중복 판단하지 않는다.

정책: **recall 우선**. 체크섬이 없는 항목(계좌 등)은 문맥 키워드로 넓게 잡고
``needs_review`` 로 넘긴다. 학습데이터 구축 단계이므로 과검은 허용한다.

회피 표기 대응: 박스마다 **원문과 정규형을 각각 훑는다**
(``normalize.canonicalize``). ``구1공8공4``, ``0lO-1234-5678``,
``hong(at)hana(dot)com`` 같은 표기를 각 정규식에 변형으로 덧붙이면 조합 폭발이
나므로, 정규화 레이어에서 한 번 접는다.

정규형 스캔은 **추가 경로이지 대체 경로가 아니다.** 접기가 정상적인 값을
망가뜨릴 수 있다 — 여권번호 ``S12345678`` 은 ``S`` 가 ``5`` 로 접혀 패턴이
깨진다. 그래서 원문 스캔이 먼저 돌고 우선권을 갖는다.

``RuleHit.span`` 은 어느 경로로 잡혔든 **원문 오프셋**이다 — 마스킹은 원본에
적용되기 때문이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..normalize import Canonical, canonicalize, identity
from ..schema import OcrBox, OcrStatus
from .checksums import VALIDATORS, digits_only

# --------------------------------------------------------------------------
# 패턴 정의
# --------------------------------------------------------------------------

# 구분자는 하이픈/공백/점 모두 허용 (OCR 이 하이픈을 놓치는 경우가 흔하다)
_SEP = r"[\-–—.\s]?"

# IPv4 옥텟 (0~255). 범위를 정규식에서 제한해 "300.1.2.3" 같은 것을 걸러낸다.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
_IPV4 = rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\d.])"

# IPv6. **전체형(8그룹) 또는 '::' 압축형만** 인정한다.
# 그룹 수 하한을 낮추면 "12:34:56" 같은 시각 표기를 IP 로 잡는다.
_H = r"[0-9A-Fa-f]{1,4}"
_IPV6 = (
    r"(?<![0-9A-Za-z:.])(?:"
    rf"{_H}(?::{_H}){{7}}"                        # 전체형 8그룹
    rf"|{_H}(?::{_H})*::(?:{_H}(?::{_H})*)?"      # 앞쪽 있는 압축형 (fe80::1, fe80::)
    rf"|::{_H}(?::{_H})*"                         # 앞쪽 없는 압축형 (::1)
    r")(?![0-9A-Za-z:.])"
)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # IP 를 맨 앞에 둔다. 구조 검증(옥텟 범위 / 그룹 수)을 통과한 IP 는
    # 뒤쪽 숫자 패턴이 일부를 가져가기 전에 구간을 선점해야 한다.
    ("IP", re.compile(_IPV6)),
    ("IP", re.compile(_IPV4)),
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
#:
#: ``is_field_label()`` 은 여기에 **완전히 일치**하는 박스만 라벨로 본다.
#: ``propagate`` 모듈은 같은 목록을 값 앞의 **접두 라벨 제거**에도 쓴다
#: (예: ``"담당자: 조민석"`` -> 전파 씨앗은 ``"조민석"``).
FIELD_LABEL_WORDS = (
    "성명", "이름", "주민등록번호", "주민번호", "생년월일", "주소", "연락처",
    "전화번호", "휴대전화", "이메일", "전자우편", "직장명", "소속", "직위",
    "사업자등록번호", "법인등록번호", "계좌번호", "카드번호", "여권번호",
    "운전면허번호", "예금주", "서명", "날인", "귀하", "신청인", "대표자",
    # 접두 라벨로 자주 붙는 것들
    "담당자", "담당", "본인", "고객명", "고객", "한글성명", "영문성명",
    "직책", "부서", "회사명", "상호", "법인명", "자택주소", "현주소",
    "주민등록상주소", "이메일주소", "휴대폰", "휴대폰번호", "전화",
)


@dataclass
class RuleHit:
    """규칙 레이어 탐지 결과.

    Attributes:
        box_index: 박스 번호.
        label: 확정된 라벨.
        text: **원문** 그대로의 매칭 문자열. 회피 표기라면 회피 표기가 담긴다
            (감사 추적용 — 문서에 실제로 뭐가 적혀 있었는지 남아야 한다).
        canonical: 정규화된 매칭 문자열. 체크섬 검증에 쓴 값이다.
        span: **원문** 기준 문자 오프셋. 마스킹은 원본에 적용된다.
        checksum_ok: 체크섬 통과 여부.
        needs_review: 사람 검토 필요 여부.
        normalized: 회피 표기를 접어서 잡았는가. 검수 우선순위 판단에 쓴다.
    """

    box_index: int
    label: str
    text: str
    span: tuple[int, int]
    checksum_ok: bool
    needs_review: bool = False
    canonical: str = ""
    normalized: bool = False


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


def _row_context(
    boxes: list[OcrBox],
    target: OcrBox,
    window: int = 2,
    canon: dict[int, str] | None = None,
) -> str:
    """같은 행 및 인접 행의 텍스트를 모아 문맥 문자열로 반환한다.

    Args:
        canon: 박스별 정규형 캐시. 주면 정규형으로 문맥을 만든다 (전각으로 적힌
            "계좌" 같은 것도 키워드에 걸리게 한다).
    """
    parts: list[str] = []
    for b in boxes:
        if abs(b.row - target.row) <= window:
            parts.append(canon.get(b.index, b.text) if canon else b.text)
    return " ".join(parts)


# --------------------------------------------------------------------------
# 메인 진입점
# --------------------------------------------------------------------------


def _scan_box(
    box: OcrBox,
    boxes: list[OcrBox],
    canon: Canonical,
    canon_text: dict[int, str],
) -> list[RuleHit]:
    """박스 하나를 주어진 텍스트 표현 위에서 훑는다.

    Args:
        box: 대상 박스.
        boxes: 문맥 판단용 전체 박스.
        canon: 스캔할 텍스트 표현. 원문(``normalize.identity``)일 수도 있고
            정규형(``normalize.canonicalize``)일 수도 있다.
        canon_text: 박스별 정규형 캐시 (문맥 키워드 매칭용).

    Returns:
        ``span`` 이 **원문 오프셋**으로 되돌려진 탐지 결과.
    """
    text = canon.text
    hits: list[RuleHit] = []
    claimed: list[tuple[int, int]] = []

    def record(
        label: str, match: re.Match[str], checksum_ok: bool, needs_review: bool
    ) -> None:
        orig = canon.span(*match.span())
        raw = match.group(0)
        original = box.text[orig[0] : orig[1]]
        hits.append(
            RuleHit(
                box_index=box.index,
                label=label,
                text=original or raw,
                span=orig,
                checksum_ok=checksum_ok,
                needs_review=needs_review,
                canonical=raw,
                normalized=original != raw,
            )
        )

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
                    ctx = _row_context(boxes, box, canon=canon_text)
                    keyword = {
                        "BIZ_NO": ("사업자", "등록번호"),
                        "CARD_NO": ("카드", "카드번호"),
                        "DRIVER_LICENSE": ("면허",),
                    }[label]
                    if not any(k in ctx for k in keyword):
                        continue

            claimed.append(span)
            record(resolved_label, m, checksum_ok, not checksum_ok)

    # 계좌번호: 문맥 키워드 필요
    ctx = _row_context(boxes, box, canon=canon_text)
    if any(k in ctx for k in ACCOUNT_KEYWORDS):
        for m in ACCOUNT_PATTERN.finditer(text):
            span = m.span()
            if any(span[0] < c[1] and c[0] < span[1] for c in claimed):
                continue
            d = digits_only(m.group(0))
            if not 10 <= len(d) <= 16:
                continue
            claimed.append(span)
            # 체크섬이 없으므로 항상 검토 대상
            record("ACCOUNT_NO", m, checksum_ok=False, needs_review=True)

    return hits


def _resolve_overlaps(candidates: list[RuleHit]) -> list[RuleHit]:
    """구간이 겹치는 후보 중 **근거가 강한 쪽**을 남긴다.

    "원문 경로가 항상 이긴다" 로 하면 안 된다. 원문에서 나온 약한 후보가
    정규형에서 나온 강한 후보를 밀어내기 때문이다. 실제로 겪은 사례:

        "9O1231-1234563"
          원문   -> ACCOUNT_NO "1231-1234563" (체크섬 없음, 문맥 키워드로 잡힘)
          정규형 -> RRN        "901231-1234563" (체크섬 통과)

    주민등록번호가 계좌번호 후보에 가려져 사라진다. 우선순위는 경로가 아니라
    **근거의 강도**여야 한다.

    순위:
        1. 체크섬 통과가 먼저
        2. 같으면 원문 경로가 먼저 (접기가 정상 값을 망가뜨렸을 수 있다)
        3. 같으면 긴 구간이 먼저 (값을 더 많이 덮는다)
    """
    ranked = sorted(
        candidates,
        key=lambda h: (
            not h.checksum_ok,
            h.normalized,
            -(h.span[1] - h.span[0]),
            h.span[0],
        ),
    )
    kept: list[RuleHit] = []
    for hit in ranked:
        if any(hit.span[0] < k.span[1] and k.span[0] < hit.span[1] for k in kept):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda h: (h.span, h.label))


def detect(boxes: list[OcrBox]) -> list[RuleHit]:
    """OCR 박스 목록에서 정형 식별자를 탐지한다.

    박스마다 **두 번** 훑고 겹치는 결과를 정리한다.

    1. **원문 그대로**
    2. **정규형** — 회피 표기(``공1공8공4``, ``0lO-1234-5678``,
       ``hong(at)hana(dot)com``)를 회수하는 경로

    정규화를 **대체** 경로로 쓰면 안 된다. 접기가 정상적인 값을 망가뜨릴 수
    있다 — 여권번호 ``S12345678`` 은 ``S`` 가 ``5`` 로 접혀 ``512345678`` 이
    되고 PASSPORT 패턴이 깨진다. 그래서 두 경로를 모두 돌리고,
    ``_resolve_overlaps`` 가 근거가 강한 쪽을 남긴다.

    Args:
        boxes: ``row`` 가 채워진 (읽기순서 정렬된) OCR 박스 목록.

    Returns:
        박스 인덱스별 탐지 결과. 한 박스에서 여러 건이 나올 수 있다.
        ``span`` 은 항상 원문 오프셋이다.
    """
    hits: list[RuleHit] = []

    # 박스별 정규형을 한 번만 계산한다 (_row_context 가 매 박스마다 전체를
    # 훑으므로 캐시하지 않으면 정규화가 O(n²) 번 돈다).
    canon_map = {
        box.index: canonicalize(box.text)
        for box in boxes
        if box.status is not OcrStatus.FAILED and box.text.strip()
    }
    canon_text = {idx: c.text for idx, c in canon_map.items()}

    for box in boxes:
        canon = canon_map.get(box.index)
        if canon is None:
            continue

        candidates = _scan_box(box, boxes, identity(box.text), canon_text)
        if canon.changed:
            candidates += _scan_box(box, boxes, canon, canon_text)

        hits.extend(_resolve_overlaps(candidates))

    return hits
