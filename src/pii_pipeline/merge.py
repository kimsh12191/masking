"""병합 및 검증.

LLM 출력을 그대로 신뢰하지 않는다. 다음을 모두 검사한다.

===========================  ==================================================
검사                          처리
===========================  ==================================================
범위 초과 idx                 폐기 + warning
같은 박스 + 같은 라벨 재출력  규칙 레이어 우선, LLM 것 폐기
같은 박스 + 다른 라벨         **둘 다 채택** (한 박스에 여러 종류가 섞인 경우)
같은 박스 집합 + 같은 라벨    conf 높은 쪽 채택
그룹인데 공간적으로 멀다      행 간격으로 그룹 분해
conf 낮음                     **폐기하지 않고** low_confidence 플래그
===========================  ==================================================

``text`` 는 LLM 출력이 아니라 **OCR 원문에서 재조립**한다. 이래야 텍스트 환각이
구조적으로 불가능하다.

중복 차단은 **박스 단위가 아니라 (박스, 라벨) 단위**다 (``ClaimLedger``).
한 박스에 여러 종류가 섞이는 일이 흔하기 때문이다 — ``"홍길동 901231-1234567"``
을 박스 단위로 차단하면 규칙 레이어가 RRN 을 확정한 순간 이름이 사라진다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .exclusions import should_exclude
from .ocr.layout import denorm_bbox, pad_bbox, spatially_split, union_bbox
from .rules.detectors import RuleHit
from .schema import BBox, OcrBox, OcrStatus, PiiRegion, Source

#: 최종 박스에 적용할 여유 패딩 (px). 경계 글자 잘림 방지.
DEFAULT_PAD = 2

#: VLM grounding 좌표는 부정확하므로 더 넉넉하게 패딩한다.
COARSE_PAD = 12

#: 이 값 미만이면 low_confidence 플래그. 폐기하지는 않는다.
LOW_CONF_THRESHOLD = 0.5

#: 모든 라벨을 차단하는 와일드카드. 평문 ``set[int]`` 로 넘어온 입력에 쓴다.
_ANY_LABEL = "*"


# --------------------------------------------------------------------------
# 중복 차단 원장
# --------------------------------------------------------------------------


class ClaimLedger:
    """어떤 박스가 **어떤 라벨로** 이미 확정됐는지 추적한다.

    박스 단위로 차단하면 한 박스에 섞인 두 번째 개인정보가 사라진다.
    ``"홍길동 901231-1234567"`` 한 박스에서 규칙 레이어가 RRN 을 확정하면,
    박스 단위 차단에서는 이름이 영원히 회수되지 않는다. 부분 마스킹
    (``char_span`` 기반)을 구현한 다운스트림에서는 그대로 유출로 이어진다.

    ``set[int]`` 를 그대로 넘기면 하위호환을 위해 **모든 라벨 차단**으로 본다.
    """

    def __init__(
        self, initial: Mapping[int, str] | Iterable[int] | None = None
    ) -> None:
        self._by_box: dict[int, set[str]] = {}
        if initial is None:
            return
        if isinstance(initial, Mapping):
            for idx, label in initial.items():
                # confirmed_map 은 "PHONE+EMAIL" 처럼 합쳐 놓는다
                for part in str(label).split("+"):
                    self.add(idx, part)
        else:
            for idx in initial:
                self.add(idx, _ANY_LABEL)

    def has(self, idx: int, label: str) -> bool:
        """``idx`` 박스가 ``label`` 로 이미 확정됐는가."""
        labels = self._by_box.get(idx)
        if not labels:
            return False
        return _ANY_LABEL in labels or label in labels

    def add(self, idx: int, label: str) -> None:
        self._by_box.setdefault(idx, set()).add(label)

    def add_all(self, indices: Iterable[int], label: str) -> None:
        for idx in indices:
            self.add(idx, label)

    def labels(self, idx: int) -> set[str]:
        return set(self._by_box.get(idx, ()))

    def indices(self) -> set[int]:
        """무엇이든 확정된 박스 번호. 프롬프트의 "이미 탐지된 번호" 용."""
        return set(self._by_box)

    def __contains__(self, idx: object) -> bool:
        return idx in self._by_box

    def __iter__(self) -> Iterable[int]:
        return iter(sorted(self._by_box))

    def __len__(self) -> int:
        return len(self._by_box)


def as_ledger(claimed: ClaimLedger | set[int] | None) -> ClaimLedger:
    """평문 ``set`` 도 받아들인다 (기존 호출부/테스트 하위호환)."""
    if isinstance(claimed, ClaimLedger):
        return claimed
    return ClaimLedger(claimed)


def claims_from_hits(hits: list[RuleHit]) -> ClaimLedger:
    """규칙 레이어 결과로 원장을 만든다. 라벨별로 차단된다."""
    ledger = ClaimLedger()
    for hit in hits:
        ledger.add(hit.box_index, hit.label)
    return ledger


def _assemble_text(indices: Iterable[int], boxes: list[OcrBox]) -> str | None:
    """OCR 원문에서 텍스트를 재조립한다 (LLM 출력을 쓰지 않는다)."""
    parts = [
        boxes[i].text
        for i in indices
        if 0 <= i < len(boxes) and boxes[i].status is not OcrStatus.FAILED and boxes[i].text
    ]
    return " ".join(parts) if parts else None


def _worst_status(indices: Iterable[int], boxes: list[OcrBox]) -> OcrStatus:
    order = {OcrStatus.OK: 0, OcrStatus.LOW_CONF: 1, OcrStatus.FAILED: 2}
    worst = OcrStatus.OK
    for i in indices:
        if 0 <= i < len(boxes) and order[boxes[i].status] > order[worst]:
            worst = boxes[i].status
    return worst


# --------------------------------------------------------------------------
# ③ 규칙 레이어 -> PiiRegion
# --------------------------------------------------------------------------


def regions_from_rules(
    hits: list[RuleHit], boxes: list[OcrBox], page_w: int, page_h: int
) -> list[PiiRegion]:
    """규칙 레이어 탐지 결과를 영역으로 변환한다."""
    regions: list[PiiRegion] = []
    for hit in hits:
        if not 0 <= hit.box_index < len(boxes):
            continue
        box = boxes[hit.box_index]
        regions.append(
            PiiRegion(
                id="",  # finalize 에서 부여
                type=hit.label,
                bbox=pad_bbox(box.bbox, DEFAULT_PAD, page_w, page_h),
                source=Source.RULE,
                confidence=1.0 if hit.checksum_ok else 0.6,
                text=hit.text,
                member_boxes=[box.bbox],
                member_index=[box.index],
                char_span=hit.span,
                ocr_status=box.status,
                needs_review=hit.needs_review,
                low_confidence=not hit.checksum_ok,
                reason="체크섬 통과" if hit.checksum_ok else "패턴 일치, 체크섬 미통과",
            )
        )
    return regions


def confirmed_map(hits: list[RuleHit]) -> dict[int, str]:
    """프롬프트의 ``<CONFIRMED:타입>`` 태그용 매핑을 만든다."""
    out: dict[int, str] = {}
    for hit in hits:
        if hit.box_index in out:
            if hit.label not in out[hit.box_index]:
                out[hit.box_index] = f"{out[hit.box_index]}+{hit.label}"
        else:
            out[hit.box_index] = hit.label
    return out


# --------------------------------------------------------------------------
# ④ pass-1 -> PiiRegion
# --------------------------------------------------------------------------


def regions_from_pass1(
    payload: dict[str, Any],
    boxes: list[OcrBox],
    claimed: ClaimLedger | set[int],
    page_w: int,
    page_h: int,
    warnings: list[str],
) -> list[PiiRegion]:
    """텍스트 pass 출력을 영역으로 변환하고 검증한다.

    Args:
        claimed: 중복 차단 원장. ``ClaimLedger`` 면 **(박스, 라벨) 단위**로
            차단하므로 이미 다른 라벨로 확정된 박스에도 새 라벨을 붙일 수 있다.
            평문 ``set[int]`` 를 주면 박스 단위로 전부 차단한다.
    """
    ledger = as_ledger(claimed)
    regions: list[PiiRegion] = []

    for item in payload.get("regions", []) or []:
        raw_idx = item.get("idx") or []
        label = item.get("type")
        conf = float(item.get("conf", 0.0))
        if not label:
            continue

        out_of_range = [i for i in raw_idx if not 0 <= i < len(boxes)]
        if out_of_range:
            warnings.append(f"pass1: 범위를 벗어난 박스 번호 폐기 {out_of_range} (type={label})")

        usable = [
            i for i in raw_idx if 0 <= i < len(boxes) and not ledger.has(i, label)
        ]
        if not usable:
            continue

        for group in spatially_split(usable, boxes):
            regions.append(
                _build_llm_region(
                    group, label, conf, boxes, page_w, page_h, Source.LLM_PASS1, None
                )
            )
            ledger.add_all(group, label)

    if isinstance(claimed, set):
        # 평문 set 을 넘긴 호출부는 in-place 갱신을 기대한다
        claimed.update(ledger.indices())

    return regions


# --------------------------------------------------------------------------
# ④' pass-2 -> PiiRegion
# --------------------------------------------------------------------------


def regions_from_pass2(
    payload: dict[str, Any],
    boxes: list[OcrBox],
    claimed: ClaimLedger | set[int],
    page_w: int,
    page_h: int,
    warnings: list[str],
) -> list[PiiRegion]:
    """이미지 검수 pass 출력을 영역으로 변환한다.

    ``idx`` 가 있으면 OCR 좌표를 쓴다 (정확). 없고 ``bbox_norm`` 만 있으면
    VLM grounding 좌표를 쓰되 ``coarse``/``needs_review`` 로 태깅한다.

    Args:
        claimed: 중복 차단 원장. ``regions_from_pass1`` 과 같은 규칙을 따른다.
    """
    ledger = as_ledger(claimed)
    regions: list[PiiRegion] = []

    for item in payload.get("missed", []) or []:
        label = item.get("type")
        if not label:
            continue
        conf = float(item.get("conf", 0.0))
        reason = item.get("reason")
        raw_idx = item.get("idx") or []
        bbox_norm = item.get("bbox_norm")

        out_of_range = [i for i in raw_idx if not 0 <= i < len(boxes)]
        if out_of_range:
            warnings.append(f"pass2: 범위를 벗어난 박스 번호 폐기 {out_of_range} (type={label})")

        usable = [
            i for i in raw_idx if 0 <= i < len(boxes) and not ledger.has(i, label)
        ]

        if usable:
            for group in spatially_split(usable, boxes):
                regions.append(
                    _build_llm_region(
                        group, label, conf, boxes, page_w, page_h, Source.VLM_PASS2, reason
                    )
                )
                ledger.add_all(group, label)
            continue

        if raw_idx and not usable:
            # 전부 같은 라벨로 이미 탐지된 박스였다. 앵커링 무시 사례이므로 조용히 넘긴다.
            continue

        if bbox_norm and len(bbox_norm) == 4:
            bbox = denorm_bbox([float(v) for v in bbox_norm], page_w, page_h)
            regions.append(
                PiiRegion(
                    id="",
                    type=label,
                    bbox=pad_bbox(bbox, COARSE_PAD, page_w, page_h),
                    source=Source.VLM_GROUNDING,
                    confidence=conf,
                    text=None,
                    member_boxes=[bbox],
                    member_index=[],
                    ocr_status=OcrStatus.FAILED,
                    coarse=True,
                    needs_review=True,  # 좌표 부정확 -> 반드시 사람이 확인
                    low_confidence=conf < LOW_CONF_THRESHOLD,
                    reason=reason,
                )
            )
        else:
            warnings.append(f"pass2: idx 도 bbox_norm 도 없는 항목 폐기 (type={label})")

    if isinstance(claimed, set):
        claimed.update(ledger.indices())

    return regions


def _build_llm_region(
    group: list[int],
    label: str,
    conf: float,
    boxes: list[OcrBox],
    page_w: int,
    page_h: int,
    source: Source,
    reason: str | None,
) -> PiiRegion:
    member: list[BBox] = [boxes[i].bbox for i in group]
    status = _worst_status(group, boxes)
    return PiiRegion(
        id="",
        type=label,
        bbox=pad_bbox(union_bbox(member), DEFAULT_PAD, page_w, page_h),
        source=source,
        confidence=conf,
        text=_assemble_text(group, boxes),
        member_boxes=member,
        member_index=list(group),
        ocr_status=status,
        needs_review=(status is not OcrStatus.OK) or conf < LOW_CONF_THRESHOLD,
        low_confidence=conf < LOW_CONF_THRESHOLD,
        reason=reason,
    )


# --------------------------------------------------------------------------
# 최종 정리
# --------------------------------------------------------------------------


def finalize(
    regions: list[PiiRegion], boxes: list[OcrBox], warnings: list[str]
) -> list[PiiRegion]:
    """중복 제거, 제외 규칙 적용, ID 부여.

    같은 박스 집합이 두 번 나오면 conf 높은 쪽만 남긴다.
    ID 는 읽기 순서 기반으로 결정론적으로 부여한다 (재실행 시 동일 ID → 어노테이션 재연결 가능).
    """
    kept: list[PiiRegion] = []

    for region in regions:
        excluded = should_exclude(region, boxes)
        if excluded:
            warnings.append(f"제외 규칙 적용: {region.type} @ {region.bbox} ({excluded})")
            continue
        kept.append(region)

    # 같은 member_index 집합 + 같은 라벨 → conf 높은 쪽만
    best: dict[tuple[str, tuple[int, ...]], PiiRegion] = {}
    coarse_only: list[PiiRegion] = []
    for region in kept:
        if not region.member_index:
            coarse_only.append(region)
            continue
        key = (region.type, tuple(sorted(region.member_index)))
        prior = best.get(key)
        if prior is None or region.confidence > prior.confidence:
            best[key] = region

    merged = list(best.values()) + coarse_only
    merged.sort(key=lambda r: (r.bbox[1], r.bbox[0], r.type))

    for n, region in enumerate(merged, start=1):
        region.id = f"r{n:03d}"
    return merged
