"""값 전파: 한 번 확정된 값을 문서 전체에서 회수한다.

같은 개인정보 값은 한 문서에 여러 번 등장한다. 신청서 상단의 성명, 중단의
동의란 서명, 하단의 확인란, 첨부 위임장 — 같은 "홍길동" 이다. 그런데 탐지
경로는 위치마다 다르게 동작한다.

  * 규칙 레이어는 **박스 하나씩만** 본다. 쪼개진 값은 못 잡는다.
  * pass-1 은 필드 라벨 근처를 잘 잡지만, 라벨 없이 홀로 있는 값은 놓친다.
  * pass-2 는 이미지를 보지만 max_tokens 안에서 상위 몇 개만 보고한다.

즉 **같은 값인데 어떤 위치는 잡히고 어떤 위치는 안 잡히는** 상태가 정상적으로
발생한다. 한 곳에서 개인정보로 확정됐다면 나머지 위치도 같은 개인정보다.
이 모듈은 확정된 값을 씨앗으로 삼아 OCR 박스 전체를 훑어 남은 위치를 회수한다.

씨앗 만들기 — 탐지 영역의 ``text`` 를 그대로 쓰면 안 된다:

  * 규칙 레이어 영역은 ``text`` 가 매칭된 값 자체다. 그대로 쓴다.
  * LLM 영역은 ``text`` 가 **박스 전체 텍스트**다. 두 가지를 떼어내야 한다.
    - **다른 영역이 담당하는 구간**: ``"홍길동 901231-1234563"`` 의 NAME 씨앗은
      주민번호를 뺀 ``"홍길동"`` 이어야 한다. 안 빼면 다른 칸의 ``"홍길동"`` 과
      매칭되지 않는다.
    - **앞에 붙은 필드 라벨**: ``"담당자: 조민석"`` -> ``"조민석"``.

매칭은 세 단계로 하고 앞이 성공하면 멈춘다:

  1. 완전일치 포함 (구분자·공백 무시)
  2. **OCR 혼동문자 접기** (``O``↔``0``, ``l``↔``1``) 후 완전일치.
     숫자 비율이 높은 값에만 적용한다.
  3. 유사도 매칭. 한글 오독처럼 혼동표로 못 잡는 것을 받는다.

정밀도 보호 장치:

  * 라벨별 **최소 길이**. 짧은 문자열은 문서 곳곳에 우연히 들어 있다.
  * 필드 라벨만 있는 박스는 **씨앗도 대상도 되지 않는다**.
  * ``ORG``/``TITLE`` 은 기본 제외. "하나은행", "과장" 같은 값이 문서 전체에
    깔려 있어 전파하면 서식 전체가 마스킹된다.
  * 완전일치가 아닌 경로(접기·유사도)로 붙은 것은 ``needs_review`` 로 표시한다.

전파 범위는 **페이지 1장**이다. 다중 페이지 문서에서 페이지를 넘나드는 전파는
``PageResult`` 를 모아 처리하는 상위 단계의 일이다 (현재 미구현).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .merge import DEFAULT_PAD, LOW_CONF_THRESHOLD, ClaimLedger, as_ledger
from .ocr.layout import pad_bbox
from .rules.detectors import FIELD_LABEL_WORDS, is_field_label
from .schema import OcrBox, OcrStatus, PiiRegion, Source

__all__ = [
    "DEFAULT_PROPAGATE_LABELS",
    "PROPAGATE_MIN_LEN",
    "PropagateConfig",
    "fold_confusables",
    "normalize",
    "propagate_regions",
    "seed_value",
]

#: 정규화 시 제거할 문자. 구분자·공백·괄호는 OCR 마다 달라지므로 무시한다.
_STRIP_RE = re.compile(r"[\s\-–—_./\\,·:;()\[\]{}'\"|]+")

#: 값 앞에 붙은 필드 라벨을 떼는 패턴 (긴 라벨을 먼저 시도해야 한다).
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(w) for w in sorted(FIELD_LABEL_WORDS, key=len, reverse=True))
    + r")\s*[:：\-]?\s*"
)

#: 라벨별 최소 길이 (정규화 후). 짧은 값은 문서 곳곳에 우연히 들어 있다.
PROPAGATE_MIN_LEN: dict[str, int] = {
    "NAME": 2,
    "ADDRESS": 8,
    "BIRTH": 6,
    "RRN": 10,
    "FOREIGN_ID": 10,
    "PASSPORT": 7,
    "DRIVER_LICENSE": 8,
    "BIZ_NO": 8,
    "CORP_NO": 10,
    "CARD_NO": 10,
    "PHONE": 7,
    "EMAIL": 6,
    "IP": 7,
    "ACCOUNT_NO": 8,
    "OTHER": 5,
    # 기본 전파 대상은 아니지만 opt-in 할 수 있으므로 길이는 정의해 둔다
    "ORG": 4,
    "TITLE": 4,
    "SIGNATURE": 99,  # 텍스트가 없다. 사실상 전파 불가.
}

#: 라벨 목록을 지정하지 않았을 때 전파할 라벨.
#:
#: ``ORG``/``TITLE`` 은 제외한다 — "하나은행", "과장" 같은 값이 서식 전체에
#: 깔려 있어 전파하면 문서 전체가 마스킹된다. ``SIGNATURE`` 는 텍스트가 없다.
DEFAULT_PROPAGATE_LABELS: tuple[str, ...] = tuple(
    label
    for label in PROPAGATE_MIN_LEN
    if label not in ("ORG", "TITLE", "SIGNATURE")
)

#: OCR 이 실제로 혼동하는 글자 -> 숫자 정규형.
#:
#: 일반 유사도 임계값으로는 이걸 잡을 수 없다. 11자리 전화번호에 오독이
#: **한 글자만** 있어도 ``SequenceMatcher`` 비율은 0.909 로 떨어지므로,
#: 임계값을 0.9 근처에 두면 아무것도 안 걸리고 0.8 로 내리면 전혀 다른 번호가
#: 걸린다. 혼동쌍을 명시적으로 접는 편이 훨씬 정확하다.
_CONFUSABLES = str.maketrans(
    {
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "l": "1", "I": "1", "i": "1", "|": "1",
        "Z": "2", "z": "2",
        "S": "5", "s": "5",
        "b": "6", "G": "6",
        "T": "7",
        "B": "8",
        "g": "9", "q": "9",
    }
)

#: 혼동문자 접기를 적용할 숫자 비율 하한.
#:
#: 식별자(번호)에만 적용한다. 이름·주소에 적용하면 "Bob" 과 "8ob" 이 같아진다.
_FOLD_DIGIT_RATIO = 0.6

#: 유사도 매칭을 시도할 최소 길이. 짧은 문자열의 유사도는 노이즈다.
_SIM_MIN_LEN = 5

#: 유사도 매칭 대상 박스 텍스트 길이 상한 (연산량 방어).
_SIM_MAX_TARGET = 200


@dataclass
class PropagateConfig:
    """값 전파 설정.

    Attributes:
        enabled: 전파 수행 여부.
        fold_confusables: OCR 혼동문자(``O``↔``0``, ``l``↔``1``)를 접어서
            비교할지. 숫자 비율이 높은 값에만 적용된다.
        min_similarity: 유사도 매칭 임계값 (0.0~1.0). ``1.0`` 이면 유사도
            매칭을 끈다. 이름·주소처럼 혼동문자 접기로 안 되는 값에 쓴다.
            **0.9 이상으로 올리면 사실상 완전일치만 남는다** — 11자 값에서
            한 글자만 틀려도 비율이 0.909 다.
        max_seeds: 씨앗 개수 상한. 박스가 많은 페이지에서 연산량을 제한한다.
            잘리면 warning 을 남긴다 (조용히 줄이지 않는다).
        propagate_labels: 전파할 라벨. ``None`` 이면
            ``DEFAULT_PROPAGATE_LABELS``.
    """

    enabled: bool = True
    fold_confusables: bool = True
    min_similarity: float = 0.85
    max_seeds: int = 64
    propagate_labels: tuple[str, ...] | None = None


# --------------------------------------------------------------------------
# 정규화
# --------------------------------------------------------------------------


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """정규화 문자열과 ``정규화 위치 -> 원문 위치`` 매핑을 함께 만든다.

    매핑이 있어야 매칭 결과를 원문 오프셋(``char_span``)으로 돌려놓을 수 있고,
    다운스트림이 부분 마스킹을 구현할 수 있다.
    """
    chars: list[str] = []
    positions: list[int] = []
    for i, ch in enumerate(text):
        if _STRIP_RE.match(ch):
            continue
        chars.append(ch)
        positions.append(i)
    return "".join(chars), positions


def normalize(text: str) -> str:
    """구분자·공백을 제거한 비교용 문자열."""
    return _normalize_with_map(text)[0]


def seed_value(text: str) -> str:
    """탐지 텍스트에서 전파 씨앗으로 쓸 **값 부분**만 남긴다.

    ``"담당자: 조민석"`` -> ``"조민석"``. 필드 라벨을 남겨두면 "담당자" 라는
    글자만 있는 박스까지 전파되어 서식 라벨이 전부 마스킹된다.
    """
    prev = None
    out = text.strip()
    while out and out != prev:
        prev = out
        out = _LABEL_PREFIX_RE.sub("", out, count=1).strip()
    return out


# --------------------------------------------------------------------------
# 매칭
# --------------------------------------------------------------------------


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def fold_confusables(text: str) -> str:
    """OCR 혼동문자를 숫자 정규형으로 접는다.

    숫자 비율이 낮은 값(이름·주소·영문)에는 적용하지 않는다 — "Bob" 을 접으면
    "806" 이 되어 전혀 다른 값과 같아진다.

    비율은 **접기 전 원문**으로 판정해야 한다. 접은 결과로 판정하면 글자가
    숫자로 바뀌면서 비율이 올라가 모든 영문 단어가 통과한다.
    """
    if _digit_ratio(text) < _FOLD_DIGIT_RATIO:
        return text
    return text.translate(_CONFUSABLES)


def _find_exact(needle: str, haystack: str) -> tuple[int, int] | None:
    pos = haystack.find(needle)
    if pos < 0:
        return None
    return (pos, pos + len(needle))


def _find_similar(
    needle: str, haystack: str, threshold: float
) -> tuple[int, int, float] | None:
    """``haystack`` 안에서 ``needle`` 과 가장 비슷한 구간을 찾는다.

    OCR 오독을 흡수하기 위한 것이다. 길이가 같은 창을 훑으며 최고 유사도를
    고른다. 박스 텍스트는 짧으므로 이 정도 연산으로 충분하다.
    """
    if threshold >= 1.0 or len(needle) < _SIM_MIN_LEN:
        return None
    if len(haystack) > _SIM_MAX_TARGET or len(haystack) < len(needle):
        return None

    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)
    best: tuple[int, int, float] | None = None
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        matcher.set_seq1(haystack[start : start + width])
        ratio = matcher.ratio()
        if ratio >= threshold and (best is None or ratio > best[2]):
            best = (start, start + width, ratio)
    return best


# --------------------------------------------------------------------------
# 씨앗 수집
# --------------------------------------------------------------------------


@dataclass
class _Seed:
    label: str
    value: str          # 정규화된 값
    confidence: float
    origin: str         # 로그용 원문


def _spans_by_box(regions: list[PiiRegion]) -> dict[int, list[tuple[int, int]]]:
    """박스별로 **다른 영역이 이미 담당하는** 문자 구간을 모은다."""
    out: dict[int, list[tuple[int, int]]] = {}
    for region in regions:
        if region.char_span is None or len(region.member_index) != 1:
            continue
        out.setdefault(region.member_index[0], []).append(region.char_span)
    return out


def _subtract_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` 에서 주어진 구간을 지운다.

    한 박스에 여러 종류가 섞여 있을 때 필요하다. ``"홍길동 901231-1234563"`` 의
    NAME 영역은 박스 전체 텍스트를 ``text`` 로 갖는데, 그걸 그대로 씨앗으로
    쓰면 다른 곳의 ``"홍길동"`` 과 매칭되지 않는다. 주민번호 구간은 이미 RRN
    영역이 담당하므로 빼야 이름만 남는다.
    """
    if not spans:
        return text
    keep = [True] * len(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(text), end)):
            keep[i] = False
    return "".join(ch for ch, k in zip(text, keep, strict=True) if k)


def _collect_seeds(
    regions: list[PiiRegion], boxes: list[OcrBox], labels: set[str]
) -> list[_Seed]:
    """탐지 결과에서 전파 씨앗을 만든다. 같은 (라벨, 값) 은 한 번만."""
    seen: set[tuple[str, str]] = set()
    seeds: list[_Seed] = []
    other_spans = _spans_by_box(regions)

    for region in regions:
        if region.type not in labels:
            continue
        raw = (region.text or "").strip()
        if not raw:
            continue

        # 규칙 레이어 탐지는 ``text`` 가 이미 매칭된 값 자체다. LLM 탐지는 박스
        # 전체 텍스트이므로 (a) 다른 영역이 담당하는 구간과 (b) 앞에 붙은 필드
        # 라벨을 떼어내야 값만 남는다.
        if region.char_span is None and len(region.member_index) == 1:
            idx = region.member_index[0]
            spans = other_spans.get(idx, [])
            if spans and 0 <= idx < len(boxes) and boxes[idx].text == raw:
                raw = _subtract_spans(raw, spans).strip()
                if not raw:
                    continue

        value = seed_value(raw)
        norm = normalize(value)
        if len(norm) < PROPAGATE_MIN_LEN.get(region.type, 99):
            continue
        if is_field_label(value):
            continue

        key = (region.type, norm)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(
            _Seed(
                label=region.type,
                value=norm,
                confidence=region.confidence,
                origin=value,
            )
        )

    # 긴 값을 먼저 쓴다 — 긴 값이 더 특이하고, 짧은 값에 먹히지 않는다.
    seeds.sort(key=lambda s: (-len(s.value), s.label, s.value))
    return seeds


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def propagate_regions(
    regions: list[PiiRegion],
    boxes: list[OcrBox],
    page_w: int,
    page_h: int,
    claimed: ClaimLedger | set[int] | None = None,
    config: PropagateConfig | None = None,
    warnings: list[str] | None = None,
) -> list[PiiRegion]:
    """확정된 값과 같은/비슷한 텍스트를 가진 박스를 추가 영역으로 회수한다.

    Args:
        regions: 규칙 + pass-1 + pass-2 로 이미 탐지된 영역 (씨앗 공급원).
        boxes: 페이지의 전체 OCR 박스.
        page_w: 페이지 폭.
        page_h: 페이지 높이.
        claimed: 중복 차단 원장. ``(박스, 라벨)`` 이 이미 있으면 건너뛴다.
        config: 전파 설정.
        warnings: 경고를 append 할 리스트.

    Returns:
        **추가로** 만들어진 영역 목록 (입력 ``regions`` 는 포함하지 않는다).
        ``claimed`` 원장은 in-place 로 갱신된다.
    """
    cfg = config or PropagateConfig()
    warn = warnings if warnings is not None else []
    if not cfg.enabled:
        return []

    labels = set(cfg.propagate_labels or DEFAULT_PROPAGATE_LABELS)
    ledger = as_ledger(claimed)

    # 이미 탐지된 영역은 원장에 올린다. 이걸 빼면 씨앗이 **자기 자신의 박스**에
    # 다시 전파되어 같은 영역이 두 번 만들어진다 (호출부가 원장을 채워 넘기지
    # 않는 경우에도 안전해야 한다).
    for region in regions:
        for idx in region.member_index:
            ledger.add(idx, region.type)

    seeds = _collect_seeds(regions, boxes, labels)
    if len(seeds) > cfg.max_seeds:
        warn.append(
            f"값 전파: 씨앗 {len(seeds)}개 중 상위 {cfg.max_seeds}개만 사용 "
            f"(긴 값 우선). 나머지는 전파하지 않았다."
        )
        seeds = seeds[: cfg.max_seeds]
    if not seeds:
        return []

    # 박스별 정규화 텍스트를 한 번만 계산한다
    norm_cache: dict[int, tuple[str, list[int]]] = {}
    for box in boxes:
        if box.status is OcrStatus.FAILED or not box.text.strip():
            continue
        if is_field_label(box.text):
            continue  # 필드 라벨 박스는 값이 아니다
        norm_cache[box.index] = _normalize_with_map(box.text)

    out: list[PiiRegion] = []

    for seed in seeds:
        for idx, (norm_text, positions) in norm_cache.items():
            if ledger.has(idx, seed.label):
                continue

            # ① 완전일치 -> ② 혼동문자 접기 -> ③ 유사도. 앞이 성공하면 멈춘다.
            match_kind = "exact"
            span_norm = _find_exact(seed.value, norm_text)

            if span_norm is None and cfg.fold_confusables:
                folded_seed = fold_confusables(seed.value)
                folded_text = fold_confusables(norm_text)
                # 접기가 실제로 뭔가를 바꿨을 때만 의미가 있다. 길이는 보존되므로
                # 오프셋을 그대로 원문에 되돌릴 수 있다.
                if folded_seed != seed.value or folded_text != norm_text:
                    span_norm = _find_exact(folded_seed, folded_text)
                    if span_norm is not None:
                        match_kind = "folded"

            similarity = 1.0
            if span_norm is None:
                found = _find_similar(seed.value, norm_text, cfg.min_similarity)
                if found is None:
                    continue
                span_norm = (found[0], found[1])
                similarity = found[2]
                match_kind = "similar"

            box = boxes[idx]
            char_span = (
                positions[span_norm[0]],
                positions[span_norm[1] - 1] + 1,
            )
            exact = match_kind == "exact"
            # 전파는 원본 탐지보다 한 단계 약한 근거다. 완전일치는 거의 확실하고,
            # 접힌 것/유사한 것은 사람이 확인해야 한다.
            conf = min(seed.confidence, 0.95 if exact else 0.7)

            out.append(
                PiiRegion(
                    id="",  # finalize 에서 부여
                    type=seed.label,
                    bbox=pad_bbox(box.bbox, DEFAULT_PAD, page_w, page_h),
                    source=Source.PROPAGATED,
                    confidence=conf,
                    text=box.text,
                    member_boxes=[box.bbox],
                    member_index=[box.index],
                    char_span=char_span,
                    ocr_status=box.status,
                    needs_review=(not exact) or box.status is not OcrStatus.OK,
                    low_confidence=conf < LOW_CONF_THRESHOLD,
                    reason=(
                        f"다른 위치에서 확정된 {seed.label} 값 '{seed.origin}' 과 "
                        + {
                            "exact": "동일",
                            "folded": "동일 (OCR 혼동문자 보정 후)",
                            "similar": f"유사(={similarity:.2f})",
                        }[match_kind]
                    ),
                )
            )
            ledger.add(idx, seed.label)

    if isinstance(claimed, set):
        claimed.update(ledger.indices())

    return out
