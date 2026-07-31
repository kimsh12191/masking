"""③ 검증 및 최종 정리 — 체크섬·자리수·항목명 필터.

이 모듈은 **탐지하지 않는다.** VLM 이 이미 판단한 것을 검사만 한다.
이전 구조에서는 같은 지식(정규식 + 체크섬)이 **탐지기**로 쓰였고, 그게 문제의
근원이었다.

    정규식이 탐지기였을 때:  12자리 숫자열 -> "체크섬 통과한 운전면허번호,
                            confidence 1.00, 검토 불필요"
                            (``DRIVER_LICENSE`` 는 체크섬이 아예 없는데도
                             통과로 취급됐고, 문맥 키워드 검사도 건너뛰었다.
                             하이픈을 흘린 주민등록번호가 여기로 빨려 들어가
                             확정 라벨을 받고 두 LLM pass 에서 제외됐다.)

    체크섬이 검증기일 때:    VLM 이 "RRN: 901112-2846261" 이라고 했다
                            -> 자리수를 센다 (13 ✓)
                            -> 체크섬을 돈다 (통과 -> verified)
                            같은 값을 DRIVER_LICENSE 라고 했다면
                            -> 13자리인데 면허번호는 12자리다 -> 경고 + 교정

**검증기는 틀린 답을 만들 수 없다.** 최악의 경우 "모르겠다" 를 낼 뿐이다.
탐지기였을 때는 틀린 답을 확신을 담아 만들어냈다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .rules.checksums import VALIDATORS, digits_only, validate_rrn
from .schema import Agreement, BBox, PiiRegion, Source

log = logging.getLogger(__name__)

#: 라벨별 기대 숫자 자리수. 옛 정규식이 알던 형식 지식을 여기로 옮겼다.
#:
#: 자리수는 **탐지 조건이 아니라 검사 항목**이다. 어긋나면 버리지 않고
#: ``needs_review`` 를 세운다 — VLM 이 맞고 OCR 이 한 자리 흘렸을 수도 있다.
DIGIT_LENGTHS: dict[str, tuple[int, ...]] = {
    "RRN": (13,),
    "FOREIGN_ID": (13,),
    "CORP_NO": (13,),
    "DRIVER_LICENSE": (12,),
    "BIZ_NO": (10,),
    "CARD_NO": (15, 16),
}

#: 13자리 6-7 형태를 가질 수 있는 라벨. 이 안에서의 혼동은 흔하고 경미하다
#: (셋 다 마스킹 대상이다). 이 밖으로 나가는 혼동이 위험하다.
_THIRTEEN_FAMILY: frozenset[str] = frozenset({"RRN", "FOREIGN_ID", "CORP_NO"})

#: 항목명(라벨) 텍스트. 값이 아니라 필드명이므로 개인정보가 아니다.
#:
#: **완전히 일치**하는 것만 라벨로 본다. "성명 홍길동" 은 값이 붙어 있으므로
#: 라벨이 아니다.
FIELD_LABEL_WORDS: frozenset[str] = frozenset(
    {
        "성명", "이름", "주민등록번호", "주민번호", "생년월일", "출생연월일",
        "주소", "연락처", "전화번호", "휴대전화", "이메일", "전자우편",
        "직장명", "소속", "직위", "사업자등록번호", "법인등록번호", "계좌번호",
        "카드번호", "여권번호", "운전면허번호", "예금주", "서명", "날인",
        "귀하", "신청인", "대표자", "담당자", "담당", "본인", "고객명", "고객",
        "한글성명", "영문성명", "직책", "부서", "회사명", "상호", "법인명",
        "자택주소", "현주소", "주민등록상주소", "이메일주소", "휴대폰",
        "휴대폰번호", "전화", "구분", "성별", "관계", "세대주", "가족사항",
        "등록기준지", "본", "비고",
    }
)

_LABEL_STRIP = re.compile(r"[\s:：()\[\]]")


@dataclass
class VerifyConfig:
    """검증 설정.

    Attributes:
        retype_on_checksum: 체크섬으로 종류가 **증명된** 경우 type 을 고칠지.
            13자리이고 주민등록번호 체크섬을 통과했는데 VLM 이 다른 종류라고
            했다면, 그건 주민등록번호다. 체크섬 통과는 우연히 일어나지 않는다
            (11분의 1이 아니라, 자리수까지 맞아야 하므로 훨씬 낮다).
            끄면 교정 대신 ``needs_review`` 만 세운다.
        drop_field_labels: 항목명만 있는 영역을 버릴지. 끄면 남기고 플래그만 센다.
        dedup_iou: 같은 라벨끼리 이 비율 이상 겹치면 중복으로 본다.
    """

    retype_on_checksum: bool = True
    drop_field_labels: bool = True
    dedup_iou: float = 0.6


# --------------------------------------------------------------------------
# 항목명 필터
# --------------------------------------------------------------------------


def is_field_label(text: str | None) -> bool:
    """항목명(필드 라벨)인지 판별. 값이 붙어 있으면 라벨로 보지 않는다."""
    if not text:
        return False
    stripped = _LABEL_STRIP.sub("", text)
    return bool(stripped) and stripped in FIELD_LABEL_WORDS


def _is_label_only(region: PiiRegion) -> bool:
    """값 없이 항목명만 잡힌 영역인가.

    ``vlm_text`` 를 먼저 본다 — 이건 모델의 판단 자체가 틀린 경우다
    ("성 명" 이라는 인쇄된 헤더를 NAME 으로 보고). ``text`` 는 크롭 OCR 이
    읽은 값이므로, 그것도 항목명뿐이면 역시 값이 아니다.
    """
    return is_field_label(region.vlm_text) or is_field_label(region.text)


# --------------------------------------------------------------------------
# 체크섬 · 자리수 검증
# --------------------------------------------------------------------------


def _candidate_texts(region: PiiRegion) -> list[str]:
    """검증에 쓸 문자열 후보.

    두 엔진의 값을 **모두** 시도한다. 한쪽이 한 자리를 흘렸어도 다른 쪽이
    온전하면 검증이 통과한다. 어느 쪽이든 통과하면 그 값이 실재한다는 뜻이다.
    """
    out: list[str] = []
    for text in (region.text, region.vlm_text):
        if text and text.strip() and text not in out:
            out.append(text)
    return out


def _check_digits(region: PiiRegion, texts: list[str]) -> str | None:
    """자리수를 검사한다. 문제가 있으면 사유 문구, 없으면 ``None``.

    ``DIGIT_LENGTHS`` 에 없는 라벨(이름·주소·전화 등)은 검사 대상이 아니다.
    """
    expected = DIGIT_LENGTHS.get(region.type)
    if not expected or not texts:
        return None

    lengths = {len(digits_only(t)) for t in texts}
    if lengths & set(expected):
        return None

    got = "/".join(str(n) for n in sorted(lengths))
    want = "/".join(str(n) for n in expected)
    hint = ""
    for label, allowed in DIGIT_LENGTHS.items():
        if label != region.type and lengths & set(allowed):
            hint = f" — 자리수는 {label} 에 맞는다"
            break
    return f"자리수 불일치: {region.type} 은 {want}자리인데 {got}자리다{hint}"


def _run_checksum(region: PiiRegion, texts: list[str]) -> tuple[str | None, str]:
    """체크섬을 돈다.

    Returns:
        ``(결과, 사유)``. 결과는 ``"ok"`` / ``"failed"`` / ``None``
        (그 라벨에 체크섬이 없음).
    """
    validator = VALIDATORS.get(region.type)
    if validator is None or not texts:
        return None, ""
    for text in texts:
        if validator(text):
            return "ok", "체크섬 통과"
    return "failed", "체크섬 미통과 (OCR 오독 또는 형식 오류 가능)"


def _retype(region: PiiRegion, texts: list[str]) -> str | None:
    """체크섬으로 종류가 증명되면 라벨을 고친다. 고쳤으면 사유 문구를 반환.

    주민등록번호 체크섬만 이 자격이 있다 — 13자리 + 가중치 검증을 우연히
    통과하기는 어렵다. 사업자·법인번호 체크섬은 자리수가 짧아 우연 통과 확률이
    높으므로 교정 근거로 쓰지 않는다.
    """
    if region.type in _THIRTEEN_FAMILY:
        return None
    if not any(len(digits_only(t)) == 13 and validate_rrn(t) for t in texts):
        return None
    old = region.type
    region.type = "RRN"
    return f"type 교정: {old} -> RRN (13자리 + 주민등록번호 체크섬 통과)"


def verify_regions(
    regions: list[PiiRegion],
    config: VerifyConfig | None = None,
    warnings: list[str] | None = None,
) -> list[PiiRegion]:
    """영역 목록을 검증하고 플래그를 세운다. **제자리에서 수정한다.**

    Args:
        regions: 좌표까지 확정된 영역 목록.
        config: 검증 설정.
        warnings: 경고를 append 할 목록.

    Returns:
        같은 목록 (체이닝 편의).
    """
    cfg = config or VerifyConfig()
    warn = warnings if warnings is not None else []

    for region in regions:
        texts = _candidate_texts(region)
        notes: list[str] = []

        if cfg.retype_on_checksum and (note := _retype(region, texts)):
            notes.append(note)
            warn.append(f"{note} (값 '{(region.text or region.vlm_text or '')[:20]}')")

        digit_problem = _check_digits(region, texts)
        if digit_problem:
            notes.append(digit_problem)
            region.needs_review = True

        result, note = _run_checksum(region, texts)
        region.checksum = result
        region.verified = result == "ok"
        if note:
            notes.append(note)
        if result == "failed":
            region.needs_review = True

        # 체크섬 통과 + 두 엔진 완전일치면 사람이 볼 이유가 없다.
        # 검토 큐를 이 조건으로 비워야 정작 위험한 건이 눈에 띈다.
        if region.verified and region.agreement is Agreement.EXACT:
            region.confidence = 1.0
            region.needs_review = False

        if notes:
            region.reason = (
                f"{region.reason}; {'; '.join(notes)}" if region.reason else "; ".join(notes)
            )

    return regions


# --------------------------------------------------------------------------
# 최종 정리
# --------------------------------------------------------------------------


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _rank(region: PiiRegion) -> tuple[int, int, float]:
    """중복 중 남길 우선순위. 큰 것이 이긴다.

    좌표가 확정된 쪽이 무조건 먼저다 — 같은 값에 대해 정확한 좌표와 근사
    좌표가 둘 다 있으면 정확한 쪽만 남겨야 한다. 그 다음이 체크섬, 그 다음이 conf.
    """
    return (
        1 if region.source is Source.OCR_REFINED else 0,
        1 if region.verified else 0,
        region.confidence,
    )


def finalize(
    regions: list[PiiRegion],
    config: VerifyConfig | None = None,
    warnings: list[str] | None = None,
) -> list[PiiRegion]:
    """중복 제거, 항목명 영역 제외, ID 부여.

    ID 는 읽기 순서(위->아래, 왼->오른)로 결정론적으로 부여한다. 같은 입력을
    다시 처리하면 같은 ID 가 나오므로 사람이 붙인 어노테이션을 재연결할 수 있다.

    Args:
        regions: 검증까지 끝난 영역 목록.
        config: 설정.
        warnings: 경고를 append 할 목록.

    Returns:
        정리된 **새** 목록.
    """
    cfg = config or VerifyConfig()
    warn = warnings if warnings is not None else []

    kept: list[PiiRegion] = []
    for region in regions:
        if _is_label_only(region):
            if cfg.drop_field_labels:
                warn.append(
                    f"항목명만 있는 영역 제외: {region.type} "
                    f"'{region.vlm_text or region.text}' @ {region.bbox}"
                )
                continue
            region.needs_review = True

        # 같은 라벨 + 크게 겹치는 기존 영역이 있으면 나은 쪽만 남긴다
        duplicate = next(
            (
                k
                for k in kept
                if k.type == region.type and _iou(k.bbox, region.bbox) >= cfg.dedup_iou
            ),
            None,
        )
        if duplicate is None:
            kept.append(region)
        elif _rank(region) > _rank(duplicate):
            kept[kept.index(duplicate)] = region

    kept.sort(key=lambda r: (r.bbox[1], r.bbox[0], r.type))
    for n, region in enumerate(kept, start=1):
        region.id = f"r{n:03d}"
    return kept
