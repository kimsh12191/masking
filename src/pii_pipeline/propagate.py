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

비교는 **회피 표기를 접은 형태**로 한다 (``normalize.canonicalize``).
``"공1공-1234-5678"`` 이 ``"010-1234-5678"`` 과 같은 값으로 잡히고, 전각·
비가시문자·호모글리프도 같은 값이 된다. 그 위에서 두 단계로 매칭한다.

  1. 완전일치 포함 (구분자·공백 무시)
  2. 유사도 매칭. 한글 오독처럼 정규화로 못 접는 것을 받는다.
     **``SIMILARITY_LABELS`` 에 있는 라벨만** 이 단계를 쓴다 — 번호류는 한 글자
     달라지면 다른 값이므로 유사도로 이어붙이면 안 된다.

정밀도 보호 장치:

  * 라벨별 **최소 길이**. 짧은 문자열은 문서 곳곳에 우연히 들어 있다.
  * 필드 라벨만 있는 박스는 **씨앗도 대상도 되지 않는다**.
  * 원문 그대로 같지 않은 것(정규화를 거친 것, 유사도로 붙은 것)은
    ``needs_review`` 로 표시하고 conf 를 낮춘다.

전파 범위는 **페이지 1장**이다. 다중 페이지 문서에서 페이지를 넘나드는 전파는
``PageResult`` 를 모아 처리하는 상위 단계의 일이다 (현재 미구현).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .merge import DEFAULT_PAD, LOW_CONF_THRESHOLD, ClaimLedger, as_ledger
from .normalize import canonicalize
from .ocr.layout import pad_bbox
from .rules.detectors import FIELD_LABEL_WORDS, is_field_label
from .schema import OcrBox, OcrStatus, PiiRegion, Source

__all__ = [
    "DEFAULT_PROPAGATE_LABELS",
    "PROPAGATE_MIN_LEN",
    "SIMILARITY_LABELS",
    "PropagateConfig",
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
    "ORG": 3,
    "TITLE": 2,
    "SIGNATURE": 99,  # 텍스트가 없다. 사실상 전파 불가.
}

#: 라벨 목록을 지정하지 않았을 때 전파할 라벨.
#:
#: ``ORG``/``TITLE`` 도 포함한다 — 마스킹 범위를 넓게 잡는 방침이므로 기관명·
#: 직위도 등장하는 모든 위치를 덮어야 한다. 서식 전체에 깔린 은행명이 전부
#: 잡히는 것은 의도한 동작이다. 좁히려면 ``propagate_labels`` 로 지정하라.
#: ``SIGNATURE`` 만 제외한다 — 텍스트가 없어 전파할 값 자체가 없다.
DEFAULT_PROPAGATE_LABELS: tuple[str, ...] = tuple(
    label for label in PROPAGATE_MIN_LEN if label != "SIGNATURE"
)

#: 유사도 매칭을 **허용하는** 라벨.
#:
#: 번호류에는 유사도를 쓰지 않는다. 숫자는 문자열로서 중복성이 없어서, 한 글자
#: 다른 번호는 "오독된 같은 번호" 가 아니라 **그냥 다른 번호**다. 실제로 겪은
#: 오탐: 여권번호 ``S12345678`` 이 전화번호 ``010-1234-5678`` 과 유사도 0.89 로
#: 붙었다 (정규화 후 ``S12345678`` vs ``012345678``).
#:
#: 번호류의 회피 표기는 ``normalize.canonicalize`` 가 결정론적으로 접으므로
#: 유사도가 필요 없다. 유사도는 한글 OCR 오독처럼 **글자에 중복성이 있어
#: 사람이 봐도 같은 값이라고 판단할 수 있는** 경우에만 쓴다.
SIMILARITY_LABELS: frozenset[str] = frozenset({"NAME", "ADDRESS", "ORG", "TITLE"})

#: 유사도 매칭을 시도할 최소 길이. 짧은 문자열의 유사도는 노이즈다.
_SIM_MIN_LEN = 5

#: 유사도 매칭 대상 박스 텍스트 길이 상한 (연산량 방어).
_SIM_MAX_TARGET = 200


@dataclass
class PropagateConfig:
    """값 전파 설정.

    Attributes:
        enabled: 전파 수행 여부.
        min_similarity: 유사도 매칭 임계값 (0.0~1.0). ``1.0`` 이면 유사도
            매칭을 끈다. ``SIMILARITY_LABELS`` 라벨에만 적용된다.
            **0.9 이상으로 올리면 사실상 완전일치만 남는다** — 11자 값에서
            한 글자만 틀려도 비율이 0.909 다.
        max_seeds: 씨앗 개수 상한. 박스가 많은 페이지에서 연산량을 제한한다.
            잘리면 warning 을 남긴다 (조용히 줄이지 않는다).
        propagate_labels: 전파할 라벨. ``None`` 이면
            ``DEFAULT_PROPAGATE_LABELS``.
    """

    enabled: bool = True
    min_similarity: float = 0.85
    max_seeds: int = 64
    propagate_labels: tuple[str, ...] | None = None


# --------------------------------------------------------------------------
# 정규화
# --------------------------------------------------------------------------


@dataclass
class _Norm:
    """비교용 정규 문자열 + 원문 오프셋 매핑.

    Attributes:
        text: 비교에 쓰는 문자열.
        starts: ``text[i]`` 가 유래한 원문 시작 오프셋.
        ends: ``text[i]`` 가 유래한 원문 끝 오프셋 (배타적).
        changed: 원문을 접어야 이 형태가 나왔는가. ``True`` 면 전파 결과를
            ``needs_review`` 로 표시한다 — 회피 표기를 접어서 붙인 것이므로
            사람이 확인해야 한다.
    """

    text: str
    starts: list[int]
    ends: list[int]
    changed: bool

    def span(self, start: int, end: int) -> tuple[int, int]:
        return (self.starts[start], self.ends[end - 1])


def _normalize_with_map(text: str) -> _Norm:
    """회피 표기를 접고 구분자를 제거한 비교용 형태를 만든다.

    ``normalize.canonicalize`` 로 회피 표기(한글 수사, 호모글리프, 전각,
    비가시문자)를 먼저 접고, 그 위에서 구분자·공백을 제거한다. 이러면
    ``"공1공-1234-5678"`` 이 ``"010-1234-5678"`` 과 같은 값으로 매칭된다.

    매핑이 있어야 매칭 결과를 원문 오프셋(``char_span``)으로 돌려놓을 수 있고,
    다운스트림이 부분 마스킹을 구현할 수 있다.
    """
    canon = canonicalize(text)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, ch in enumerate(canon.text):
        if _STRIP_RE.match(ch):
            continue
        chars.append(ch)
        starts.append(canon.starts[i])
        ends.append(canon.ends[i])
    return _Norm("".join(chars), starts, ends, canon.changed)


def normalize(text: str) -> str:
    """회피 표기를 접고 구분자·공백을 제거한 비교용 문자열."""
    return _normalize_with_map(text).text


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
    normalized: bool = False  # 회피 표기를 접어야 이 값이 나왔는가


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
        norm = _normalize_with_map(value)
        if len(norm.text) < PROPAGATE_MIN_LEN.get(region.type, 99):
            continue
        if is_field_label(value):
            continue

        key = (region.type, norm.text)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(
            _Seed(
                label=region.type,
                value=norm.text,
                confidence=region.confidence,
                origin=value,
                normalized=norm.changed,
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
    norm_cache: dict[int, _Norm] = {}
    for box in boxes:
        if box.status is OcrStatus.FAILED or not box.text.strip():
            continue
        if is_field_label(box.text):
            continue  # 필드 라벨 박스는 값이 아니다
        norm_cache[box.index] = _normalize_with_map(box.text)

    out: list[PiiRegion] = []

    for seed in seeds:
        for idx, norm in norm_cache.items():
            if ledger.has(idx, seed.label):
                continue

            # ① 완전일치 -> ② 유사도. 회피 표기 접기는 이미 정규화에서 끝났다.
            # 유사도는 글자에 중복성이 있는 라벨에만 쓴다 (번호류는 한 글자만
            # 달라도 다른 값이다).
            span_norm = _find_exact(seed.value, norm.text)
            similarity = 1.0
            if span_norm is None:
                if seed.label not in SIMILARITY_LABELS:
                    continue
                found = _find_similar(seed.value, norm.text, cfg.min_similarity)
                if found is None:
                    continue
                span_norm = (found[0], found[1])
                similarity = found[2]

            # 회피 표기를 접어서 붙은 것이면 사람이 확인해야 한다. 원문 그대로
            # 같았던 것과 같은 신뢰도로 취급할 수 없다.
            folded = seed.normalized or norm.changed
            if similarity < 1.0:
                kind = f"유사(={similarity:.2f})"
            elif folded:
                kind = "동일 (회피 표기 정규화 후)"
            else:
                kind = "동일"
            solid = similarity >= 1.0 and not folded

            box = boxes[idx]
            # 전파는 원본 탐지보다 한 단계 약한 근거다. 완전일치는 거의 확실하고,
            # 접힌 것/유사한 것은 사람이 확인해야 한다.
            conf = min(seed.confidence, 0.95 if solid else 0.7)

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
                    char_span=norm.span(*span_norm),
                    ocr_status=box.status,
                    needs_review=(not solid) or box.status is not OcrStatus.OK,
                    low_confidence=conf < LOW_CONF_THRESHOLD,
                    reason=(
                        f"다른 위치에서 확정된 {seed.label} 값 "
                        f"'{seed.origin}' 과 {kind}"
                    ),
                )
            )
            ledger.add(idx, seed.label)

    if isinstance(claimed, set):
        claimed.update(ledger.indices())

    return out
