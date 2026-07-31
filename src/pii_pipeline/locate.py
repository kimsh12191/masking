"""② 좌표 확정 단계 — VLM 이 지목한 영역을 크롭해 OCR 로 측량한다.

    VlmFinding (값은 정확, 좌표는 대략)
      └─ 크롭 (여유 패딩 + 업샘플)
      └─ 배치 OCR
      └─ 크롭 안에서 VLM 이 읽은 값과 같은 텍스트 줄 찾기
      └─ PiiRegion (좌표 정확)

핵심은 마지막 단계다. **VLM 의 좌표를 신뢰하지 않고, VLM 의 텍스트를 신뢰한다.**
크롭 안에서 그 텍스트를 찾아낸 OCR 박스의 좌표를 쓴다. 그러면 좌표는 픽셀
단위로 정확해지고, 동시에 두 엔진이 같은 값을 읽었다는 **교차검증**이 공짜로
따라온다 (``Agreement``).

찾지 못하는 경우가 셋 있고, 각각 다르게 처리한다.

===============================  ==============================================
크롭 OCR 이 아무것도 못 읽었다     손글씨·도장. VLM 좌표를 쓰고 ``VLM_COARSE``
OCR 은 읽었는데 값이 다르다        한쪽 오독. VLM 좌표 + ``reason`` 에 OCR 원문
읽을 글자가 애초에 없다 (서명)     정상. VLM 좌표를 쓰고 검토 플래그 없음
===============================  ==============================================

두 번째 경우를 ``reason`` 에 남기는 것이 중요하다. "VLM 은 901112-2846261,
OCR 은 9O1112-284626 을 읽었다" 가 로그에 남으면 어느 엔진을 손봐야 하는지
바로 보인다. 이전 구조에서는 이 정보가 그냥 사라졌다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .normalize import canonicalize
from .ocr.layout import denorm_bbox, pad_bbox, union_bbox
from .ocr.paddle_runner import PaddleOcrRunner
from .schema import (
    TEXTLESS_LABELS,
    Agreement,
    BBox,
    OcrBox,
    OcrStatus,
    PiiRegion,
    Source,
    VlmFinding,
)

log = logging.getLogger(__name__)

#: 비교용 정규화에서 제거할 구분자.
_STRIP = frozenset(" \t\n\r-–—_./\\,·:;()[]{}'\"|")

#: 유사도 매칭을 **허용하는** 라벨.
#:
#: 번호류에는 쓰지 않는다. 숫자는 문자열로서 중복성이 없어서 한 글자 다른 번호는
#: "오독된 같은 번호" 가 아니라 **그냥 다른 번호**다. 번호를 유사도로 이어붙이면
#: 옆 칸의 다른 사람 주민번호에 붙는다.
SIMILARITY_LABELS: frozenset[str] = frozenset({"NAME", "ADDRESS", "ORG", "TITLE"})

#: 유사도 매칭을 시도할 최소 길이.
#:
#: **2 다.** 한국 이름은 대부분 3글자이고, 3글자에서 한 자만 오독되면
#: ``SequenceMatcher`` 비율이 0.667 이다. 하한을 4 로 두면 이름이 전부 유사
#: 매칭 대상에서 빠져 스캔 문서의 이름이 모두 ``vlm_coarse`` 로 떨어진다.
#:
#: 짧은 문자열의 유사도가 노이즈라는 것은 사실이지만, 그 걱정은 **페이지 전체를
#: 훑을 때**의 이야기다. 여기서 탐색 범위는 VLM 이 이미 지목한 크롭 안의 박스
#: 몇 개뿐이므로, 엉뚱한 사람의 이름과 만날 확률이 애초에 낮다. 그래서 기준을
#: 낮출 수 있다 — 대신 결과는 ``Agreement.SIMILAR`` + ``needs_review`` 로
#: 남고 두 엔진이 각각 뭘 읽었는지가 ``reason`` 에 적힌다.
_SIM_MIN_LEN = 2

#: 최종 박스 여유 패딩 (px). 경계 글자 잘림 방지.
DEFAULT_PAD = 2

#: VLM 좌표를 그대로 쓸 때의 여유 패딩 (px). 좌표가 부정확하므로 넉넉히.
COARSE_PAD = 12

#: 이 값 미만이면 low_confidence 플래그. 폐기하지는 않는다.
LOW_CONF_THRESHOLD = 0.5


@dataclass
class LocateConfig:
    """좌표 확정 설정.

    Attributes:
        pad_ratio: 크롭 여유 (VLM bbox 크기 대비 비율). VLM 좌표는 어긋나므로
            넉넉히 잡는다. 좁으면 값이 크롭 밖으로 나가 못 찾는다.
        min_pad_px: 크롭 여유의 최솟값 (px). 작은 bbox 에서 비율만으로는 부족하다.
        upscale: 크롭을 OCR 에 넣기 전 확대 배율. **작은 글씨에 대한 유일한
            대응책이다.** 원본에 없는 정보를 만들지는 못하지만, rec 모델은
            입력 글자 높이에 민감하므로 실측으로 효과가 있다. 1.0 이면 끈다.
        max_crop_side: 업샘플 후 크롭 긴 변 상한 (px). 넘으면 배율을 줄인다.
        retry_pad_ratio: 1차에서 값을 못 찾았을 때 크롭을 다시 뜰 여유 비율.
            VLM bbox 가 어긋난 경우를 한 번 더 시도한다.
        similarity: 유사 매칭 임계값. ``SIMILARITY_LABELS`` 에만 적용된다.
            1.0 이면 유사 매칭을 끈다.
            **0.65 인 것은 3글자 이름 때문이다.** "홍길동" 을 OCR 이 "홍길둥"
            으로 읽으면 비율이 0.667 이다. 0.75 로 두면 한국 이름의 한 글자
            오독이 전부 매칭 실패가 되어 ``vlm_coarse`` 로 떨어진다.
        max_join: 값이 여러 OCR 박스로 쪼개졌을 때 이어붙일 최대 박스 수.
    """

    pad_ratio: float = 0.35
    min_pad_px: int = 12
    upscale: float = 2.0
    max_crop_side: int = 1600
    retry_pad_ratio: float = 1.2
    similarity: float = 0.65
    max_join: int = 4


# --------------------------------------------------------------------------
# 비교용 정규화 (원문 오프셋 매핑 포함)
# --------------------------------------------------------------------------


@dataclass
class _Norm:
    """비교용 정규 문자열 + 원문 오프셋 매핑.

    ``char_span`` 은 반드시 **원문 기준**이어야 한다 — 다운스트림이 부분
    마스킹을 구현할 때 원본 텍스트에 적용하기 때문이다.
    """

    text: str
    starts: list[int]
    ends: list[int]

    def span(self, start: int, end: int) -> tuple[int, int]:
        """정규 문자열 구간 ``[start, end)`` 를 원문 오프셋으로 되돌린다."""
        return (self.starts[start], self.ends[end - 1])


def _norm(text: str) -> _Norm:
    """회피 표기를 접고 구분자·공백을 제거한 비교용 형태를 만든다.

    ``"공1공-1234-5678"`` 이 ``"01012345678"`` 이 되어 ``"010 1234 5678"`` 과
    같은 값으로 매칭된다.
    """
    canon = canonicalize(text)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, ch in enumerate(canon.text):
        if ch in _STRIP:
            continue
        chars.append(ch.upper())
        starts.append(canon.starts[i])
        ends.append(canon.ends[i])
    return _Norm("".join(chars), starts, ends)


def _similar(needle: str, haystack: str, threshold: float) -> tuple[int, int, float] | None:
    """``haystack`` 안에서 ``needle`` 과 가장 비슷한 같은 길이 구간을 찾는다."""
    if threshold >= 1.0 or len(needle) < _SIM_MIN_LEN or len(haystack) < len(needle):
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
# 크롭
# --------------------------------------------------------------------------


def crop_rect(
    finding: VlmFinding, page_w: int, page_h: int, pad_ratio: float, min_pad: int
) -> BBox:
    """VLM 의 대략 좌표에서 크롭할 페이지 사각형을 만든다.

    패딩은 **비율과 절대값 중 큰 쪽**을 쓴다. 비율만 쓰면 작은 bbox 에서 패딩이
    거의 0 이 되고, 절대값만 쓰면 큰 주소 블록에서 부족하다.
    """
    x1, y1, x2, y2 = denorm_bbox(finding.bbox_norm, page_w, page_h)
    pad_x = max(min_pad, int((x2 - x1) * pad_ratio))
    pad_y = max(min_pad, int((y2 - y1) * pad_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(page_w, x2 + pad_x),
        min(page_h, y2 + pad_y),
    )


def _cut(image: Any, rect: BBox, cfg: LocateConfig) -> tuple[Any, float]:
    """이미지를 잘라 확대한다.

    Returns:
        ``(크롭 배열, 실제 적용된 배율)``. 배율을 함께 돌려주는 것은 좌표를
        되돌리는 데 필요하기 때문이다. 상한 때문에 요청 배율이 깎일 수 있다.
    """
    x1, y1, x2, y2 = rect
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return crop, 1.0

    scale = max(1.0, cfg.upscale)
    long_side = max(crop.shape[0], crop.shape[1]) * scale
    if long_side > cfg.max_crop_side:
        scale = max(1.0, cfg.max_crop_side / max(crop.shape[0], crop.shape[1]))
    if scale <= 1.0:
        return crop, 1.0

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - 환경 의존
        return crop, 1.0

    resized = cv2.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, scale


def _to_page(boxes: list[OcrBox], rect: BBox, scale: float, crop_id: int) -> list[OcrBox]:
    """크롭 좌표의 박스들을 페이지 좌표로 옮긴다 (새 객체를 만든다)."""
    ox, oy = rect[0], rect[1]
    out: list[OcrBox] = []
    for box in boxes:
        bx1, by1, bx2, by2 = box.bbox
        out.append(
            OcrBox(
                index=-1,  # 나중에 페이지 전역 번호를 부여한다
                bbox=(
                    ox + int(bx1 / scale),
                    oy + int(by1 / scale),
                    ox + int(round(bx2 / scale)),
                    oy + int(round(by2 / scale)),
                ),
                text=box.text,
                status=box.status,
                rec_conf=box.rec_conf,
                crop_id=crop_id,
            )
        )
    return out


# --------------------------------------------------------------------------
# 크롭 안에서 값 찾기
# --------------------------------------------------------------------------


@dataclass
class _Match:
    """크롭 안에서 VLM 의 값을 찾아낸 결과."""

    boxes: list[OcrBox]
    agreement: Agreement
    char_span: tuple[int, int] | None
    similarity: float


def find_value(
    seed_text: str, boxes: list[OcrBox], label: str, cfg: LocateConfig
) -> _Match | None:
    """크롭 OCR 결과에서 VLM 이 읽은 값과 같은 박스(들)를 찾는다.

    탐색 순서에 이유가 있다.

    1. **박스 1개 안에서 완전일치** — 가장 좁고 정확한 답. 부분 마스킹을 위한
       ``char_span`` 도 이 경우에만 의미가 있다.
    2. **인접 박스 이어붙여 완전일치** — 값이 쪼개진 경우 (``"010-1234"`` 와
       ``"5678"`` 이 별개 박스). 읽기 순서로 붙인다.
    3. **유사 매칭** — 한글 오독을 흡수한다. ``SIMILARITY_LABELS`` 만.

    좁은 답을 먼저 찾는 것이 중요하다. 이어붙이기를 먼저 시도하면 값 하나에
    옆 칸까지 딸려 들어와 박스가 셀 두 개를 덮는다.

    Args:
        seed_text: VLM 이 읽은 값.
        boxes: 크롭 OCR 박스 (읽기 순서).
        label: 라벨. 유사 매칭 허용 여부를 결정한다.
        cfg: 설정.

    Returns:
        찾았으면 ``_Match``, 못 찾았으면 ``None``.
    """
    seed = _norm(seed_text).text
    if not seed:
        return None

    usable = [b for b in boxes if b.status is not OcrStatus.FAILED and b.text.strip()]
    if not usable:
        return None

    norms = [_norm(b.text) for b in usable]

    # ① 박스 1개 완전일치
    for box, norm in zip(usable, norms, strict=True):
        pos = norm.text.find(seed)
        if pos >= 0:
            return _Match(
                boxes=[box],
                agreement=Agreement.EXACT,
                char_span=norm.span(pos, pos + len(seed)),
                similarity=1.0,
            )

    # ② 인접 박스 이어붙이기
    for width in range(2, min(cfg.max_join, len(usable)) + 1):
        for i in range(len(usable) - width + 1):
            joined = "".join(n.text for n in norms[i : i + width])
            if seed in joined:
                return _Match(
                    boxes=list(usable[i : i + width]),
                    agreement=Agreement.EXACT,
                    char_span=None,  # 여러 박스에 걸쳐 원문 오프셋이 하나가 아니다
                    similarity=1.0,
                )

    # ③ 유사 매칭 (글자에 중복성이 있는 라벨만)
    if label not in SIMILARITY_LABELS:
        return None

    best: _Match | None = None
    for box, norm in zip(usable, norms, strict=True):
        found = _similar(seed, norm.text, cfg.similarity)
        if found and (best is None or found[2] > best.similarity):
            best = _Match(
                boxes=[box],
                agreement=Agreement.SIMILAR,
                char_span=norm.span(found[0], found[1]),
                similarity=found[2],
            )
    return best


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def locate(
    findings: list[VlmFinding],
    image: Any,
    ocr: PaddleOcrRunner,
    config: LocateConfig | None = None,
    warnings: list[str] | None = None,
) -> tuple[list[PiiRegion], list[OcrBox]]:
    """VLM 탐지 목록의 좌표를 크롭 OCR 로 확정한다.

    OCR 은 **두 번의 배치**로 돈다. 1차는 전부, 2차는 1차에서 값을 못 찾은
    것만 크롭을 넓혀 다시 시도한다. VLM bbox 가 조금 어긋나 값이 크롭 경계
    밖으로 나간 경우를 회수하기 위한 것이고, 재시도는 한 번만 한다 —
    두 번 넓혀서 안 잡히면 크롭 문제가 아니다.

    Args:
        findings: VLM 탐지 목록.
        image: 전처리된 BGR numpy 배열. 좌표계의 기준이다.
        ocr: OCR 실행기.
        config: 설정.
        warnings: 경고를 append 할 목록.

    Returns:
        ``(영역 목록, 크롭 OCR 박스 전체)``. 영역은 ``findings`` 와 **1:1** 이다
        (좌표를 못 잡아도 ``VLM_COARSE`` 로 남는다 — 개인정보를 조용히 버리지
        않는다). 박스 목록은 페이지 좌표로 환산되어 있고 오버레이/디버깅용이다.
    """
    cfg = config or LocateConfig()
    warn = warnings if warnings is not None else []
    if not findings:
        return [], []

    page_h, page_w = image.shape[:2]

    # ── 1차 배치 ──────────────────────────────────────────────
    matches: list[_Match | None] = [None] * len(findings)
    per_finding: list[list[OcrBox]] = [[] for _ in findings]
    _ocr_batch(findings, image, ocr, cfg, cfg.pad_ratio, range(len(findings)),
               page_w, page_h, matches, per_finding)

    # ── 2차 배치 (1차 실패분만 크롭을 넓혀 재시도) ────────────
    retry = [
        i
        for i, m in enumerate(matches)
        if m is None and findings[i].type not in TEXTLESS_LABELS and findings[i].text.strip()
    ]
    if retry:
        log.debug("좌표 확정 재시도 %d건 (크롭 확대)", len(retry))
        _ocr_batch(findings, image, ocr, cfg, cfg.retry_pad_ratio, retry,
                   page_w, page_h, matches, per_finding)

    # ── 박스에 페이지 전역 번호 부여 ──────────────────────────
    all_boxes: list[OcrBox] = []
    for boxes in per_finding:
        for box in boxes:
            box.index = len(all_boxes)
            all_boxes.append(box)

    # ── 영역 생성 ─────────────────────────────────────────────
    regions: list[PiiRegion] = []
    for finding, match, boxes in zip(findings, matches, per_finding, strict=True):
        if match is not None:
            regions.append(_refined_region(finding, match, cfg, page_w, page_h))
        else:
            regions.append(_coarse_region(finding, boxes, cfg, page_w, page_h))

    n_coarse = sum(1 for r in regions if r.coarse)
    if n_coarse:
        warn.append(
            f"좌표 확정 실패 {n_coarse}/{len(regions)}건 — VLM 좌표를 그대로 사용 "
            f"(vlm_coarse). 해당 영역은 사람 검토가 필요하다."
        )
    return regions, all_boxes


def _ocr_batch(
    findings: list[VlmFinding],
    image: Any,
    ocr: PaddleOcrRunner,
    cfg: LocateConfig,
    pad_ratio: float,
    targets: Any,
    page_w: int,
    page_h: int,
    matches: list[_Match | None],
    per_finding: list[list[OcrBox]],
) -> None:
    """``targets`` 인덱스의 탐지들에 대해 크롭 -> 배치 OCR -> 매칭을 수행한다.

    ``matches`` / ``per_finding`` 을 제자리에서 갱신한다. 이미 매칭된 항목의
    결과는 덮어쓰지 않는다 (2차에서 더 나쁜 매칭이 나와도 1차를 지킨다).
    """
    idx_list = list(targets)
    if not idx_list:
        return

    rects: list[BBox] = []
    crops: list[Any] = []
    scales: list[float] = []
    for i in idx_list:
        rect = crop_rect(findings[i], page_w, page_h, pad_ratio, cfg.min_pad_px)
        crop, scale = _cut(image, rect, cfg)
        rects.append(rect)
        crops.append(crop)
        scales.append(scale)

    results = ocr.run_many(crops)

    for i, rect, scale, boxes in zip(idx_list, rects, scales, results, strict=True):
        page_boxes = _to_page(boxes, rect, scale, crop_id=i)
        per_finding[i] = page_boxes  # 재시도 시 더 넓은 크롭 결과로 교체한다
        if matches[i] is not None:
            continue
        matches[i] = find_value(findings[i].text, page_boxes, findings[i].type, cfg)


def _refined_region(
    finding: VlmFinding, match: _Match, cfg: LocateConfig, page_w: int, page_h: int
) -> PiiRegion:
    """크롭 OCR 이 좌표를 확정한 경우."""
    member = [b.bbox for b in match.boxes]
    ocr_text = " ".join(b.text for b in match.boxes if b.text)
    worst = _worst_status(match.boxes)

    # 두 엔진이 같은 값을 읽었다면 VLM 자기확신도보다 높게 볼 근거가 있다.
    # 다만 1.0 으로 올리지는 않는다 — 체크섬 통과만이 그 자격이 있다 (verify.py).
    conf = min(0.95, max(finding.conf, 0.9)) if match.agreement is Agreement.EXACT else finding.conf

    reason = (
        "크롭 OCR 과 완전일치"
        if match.agreement is Agreement.EXACT
        else f"크롭 OCR 과 유사 일치 (={match.similarity:.2f}): OCR '{ocr_text[:40]}'"
    )
    return PiiRegion(
        id="",  # finalize 에서 부여
        type=finding.type,
        bbox=pad_bbox(union_bbox(member), DEFAULT_PAD, page_w, page_h),
        source=Source.OCR_REFINED,
        confidence=conf,
        text=ocr_text or None,
        vlm_text=finding.text or None,
        field=finding.field or None,
        member_boxes=member,
        member_index=[b.index for b in match.boxes],
        char_span=match.char_span,
        ocr_status=worst,
        coarse=False,
        needs_review=(match.agreement is not Agreement.EXACT) or worst is not OcrStatus.OK,
        low_confidence=conf < LOW_CONF_THRESHOLD,
        agreement=match.agreement,
        reason=reason,
    )


def _coarse_region(
    finding: VlmFinding,
    boxes: list[OcrBox],
    cfg: LocateConfig,
    page_w: int,
    page_h: int,
) -> PiiRegion:
    """크롭 OCR 이 값을 찾지 못한 경우. VLM 좌표를 그대로 쓴다.

    세 경우를 ``reason`` 으로 구분한다. 이 구분이 있어야 "OCR 을 손봐야 하나
    VLM 을 손봐야 하나" 를 로그만 보고 판단할 수 있다.
    """
    textless = finding.type in TEXTLESS_LABELS or not finding.text.strip()
    read = " ".join(b.text for b in boxes if b.text.strip())

    if textless:
        reason = "읽을 텍스트가 없는 항목 (서명·인영). VLM 좌표 사용"
    elif not read:
        reason = "크롭 OCR 이 아무 글자도 읽지 못했다 (손글씨/도장 추정). VLM 좌표 사용"
    else:
        reason = (
            f"크롭 OCR 이 VLM 의 값을 찾지 못했다. "
            f"VLM '{finding.text[:30]}' vs OCR '{read[:40]}'"
        )

    bbox = denorm_bbox(finding.bbox_norm, page_w, page_h)
    return PiiRegion(
        id="",
        type=finding.type,
        bbox=pad_bbox(bbox, COARSE_PAD, page_w, page_h),
        source=Source.VLM_COARSE,
        confidence=finding.conf,
        text=None,
        vlm_text=finding.text or None,
        field=finding.field or None,
        member_boxes=[bbox],
        member_index=[],
        ocr_status=OcrStatus.FAILED,
        coarse=True,
        # 서명·인영은 원래 읽을 글자가 없다. 이걸 검토 대상에 넣으면 검토
        # 큐가 서명으로 가득 차서 정작 봐야 할 불일치 건이 묻힌다.
        needs_review=not textless,
        low_confidence=finding.conf < LOW_CONF_THRESHOLD,
        agreement=Agreement.NONE,
        reason=reason,
    )


def _worst_status(boxes: list[OcrBox]) -> OcrStatus:
    order = {OcrStatus.OK: 0, OcrStatus.LOW_CONF: 1, OcrStatus.FAILED: 2}
    worst = OcrStatus.OK
    for box in boxes:
        if order[box.status] > order[worst]:
            worst = box.status
    return worst
