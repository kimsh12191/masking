"""병합 및 검증.

LLM 출력을 그대로 신뢰하지 않는다. 다음을 모두 검사한다.

===========================  ==================================================
검사                          처리
===========================  ==================================================
범위 초과 idx                 폐기 + warning
이미 확정된 박스 재출력       규칙 레이어 우선, LLM 것 폐기
한 박스가 두 라벨에           conf 높은 쪽 채택
그룹인데 공간적으로 멀다      행 간격으로 그룹 분해
conf 낮음                     **폐기하지 않고** low_confidence 플래그
===========================  ==================================================

``text`` 는 LLM 출력이 아니라 **OCR 원문에서 재조립**한다. 이래야 텍스트 환각이
구조적으로 불가능하다.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    claimed: set[int],
    page_w: int,
    page_h: int,
    warnings: list[str],
) -> list[PiiRegion]:
    """텍스트 pass 출력을 영역으로 변환하고 검증한다."""
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

        usable = [i for i in raw_idx if 0 <= i < len(boxes) and i not in claimed]
        if not usable:
            continue

        for group in spatially_split(usable, boxes):
            regions.append(
                _build_llm_region(
                    group, label, conf, boxes, page_w, page_h, Source.LLM_PASS1, None
                )
            )
            claimed.update(group)

    return regions


# --------------------------------------------------------------------------
# ④' pass-2 -> PiiRegion
# --------------------------------------------------------------------------


def regions_from_pass2(
    payload: dict[str, Any],
    boxes: list[OcrBox],
    claimed: set[int],
    page_w: int,
    page_h: int,
    warnings: list[str],
) -> list[PiiRegion]:
    """이미지 검수 pass 출력을 영역으로 변환한다.

    ``idx`` 가 있으면 OCR 좌표를 쓴다 (정확). 없고 ``bbox_norm`` 만 있으면
    VLM grounding 좌표를 쓰되 ``coarse``/``needs_review`` 로 태깅한다.
    """
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

        usable = [i for i in raw_idx if 0 <= i < len(boxes) and i not in claimed]

        if usable:
            for group in spatially_split(usable, boxes):
                regions.append(
                    _build_llm_region(
                        group, label, conf, boxes, page_w, page_h, Source.VLM_PASS2, reason
                    )
                )
                claimed.update(group)
            continue

        if raw_idx and not usable:
            # 전부 이미 탐지된 박스였다. 앵커링 무시 사례이므로 조용히 넘긴다.
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
