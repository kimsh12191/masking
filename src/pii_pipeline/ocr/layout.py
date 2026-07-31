"""읽기 순서 정렬 및 좌표 유틸.

읽기 순서가 꼬이면 LLM 이 라벨-값 쌍을 잘못 묶는다. 다단 컬럼 금융 서식에서
단순 y 정렬은 실패하므로, 행 클러스터링 후 행 내부를 x 로 정렬한다.
"""

from __future__ import annotations

import math
from statistics import median

from ..schema import BBox, OcrBox


def quad_to_bbox(quad: list[list[float]] | list[tuple[float, float]]) -> BBox:
    """PaddleOCR 의 4점 사각형을 축정렬 외접 사각형으로 변환한다.

    최솟값은 내림, 최댓값은 올림한다. ``round()`` 를 쓰면 은행가 반올림 때문에
    경계 글자가 1px 잘릴 수 있다 — 마스킹 대상 좌표에서는 넉넉한 쪽이 안전하다.
    """
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def union_bbox(boxes: list[BBox]) -> BBox:
    """여러 박스의 합집합 사각형."""
    if not boxes:
        raise ValueError("union_bbox() 에 빈 목록이 전달되었습니다")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def pad_bbox(bbox: BBox, pad: int, page_w: int, page_h: int) -> BBox:
    """박스를 여유롭게 확장한다 (경계 글자 잘림 방지)."""
    return (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(page_w, bbox[2] + pad),
        min(page_h, bbox[3] + pad),
    )


def median_height(boxes: list[OcrBox]) -> float:
    heights = [b.height for b in boxes if b.height > 0]
    return float(median(heights)) if heights else 1.0


def assign_reading_order(boxes: list[OcrBox], y_tol_ratio: float = 0.6) -> list[OcrBox]:
    """행 클러스터링 후 읽기 순서대로 ``index`` 와 ``row`` 를 재부여한다.

    같은 행 판정은 세로 중심 좌표 차이가 ``중위 글자높이 * y_tol_ratio`` 이내인지로
    한다. 폰트 크기가 섞인 서식에서도 안정적으로 동작한다.

    Args:
        boxes: OCR 박스 목록 (순서 무관).
        y_tol_ratio: 같은 행으로 볼 세로 허용오차 비율.

    Returns:
        읽기 순서로 정렬되고 ``index``/``row`` 가 채워진 **새** 목록.
    """
    if not boxes:
        return []

    tol = median_height(boxes) * y_tol_ratio
    ordered = sorted(boxes, key=lambda b: (b.cy, b.bbox[0]))

    rows: list[list[OcrBox]] = [[ordered[0]]]
    row_ref = ordered[0].cy

    for box in ordered[1:]:
        if abs(box.cy - row_ref) <= tol:
            rows[-1].append(box)
        else:
            rows.append([box])
            row_ref = box.cy

    result: list[OcrBox] = []
    idx = 0
    for row_no, row in enumerate(rows):
        for box in sorted(row, key=lambda b: b.bbox[0]):
            box.index = idx
            box.row = row_no
            result.append(box)
            idx += 1
    return result


def spatially_split(
    indices: list[int],
    boxes: list[OcrBox],
    max_vgap_ratio: float = 1.5,
    max_hgap_ratio: float = 4.0,
) -> list[list[int]]:
    """LLM 이 묶은 인덱스 그룹을 공간적 근접성으로 재검증한다.

    주소처럼 여러 박스에 걸친 항목은 물리적으로 붙어 있어야 한다. 모델이 서식의
    양 끝에 있는 무관한 박스를 잘못 묶는 경우가 있으므로 여기서 분해한다.

    판정은 **행 번호 차이가 아니라 실제 픽셀 간격**으로 한다. 행 번호는 서식이
    드문드문한 문서에서 물리적 거리를 반영하지 못한다 (행 2칸 차이가 600px 일 수 있다).

    Args:
        indices: LLM 이 반환한 박스 인덱스 목록.
        boxes: 읽기순서 정렬된 전체 박스 목록 (``boxes[i].index == i`` 가정).
        max_vgap_ratio: 다른 행일 때 허용할 세로 공백 (중위 글자높이 배수).
        max_hgap_ratio: 같은 행일 때 허용할 가로 공백 (중위 글자높이 배수).

    Returns:
        분해된 인덱스 그룹 목록. 유효한 인덱스가 없으면 빈 목록.
    """
    valid = sorted(i for i in indices if 0 <= i < len(boxes))
    if not valid:
        return []

    unit = median_height(boxes)
    groups: list[list[int]] = [[valid[0]]]

    for prev, cur in zip(valid, valid[1:], strict=False):  # 인접 쌍 순회
        pb, cb = boxes[prev], boxes[cur]
        if pb.row == cb.row:
            too_far = (cb.bbox[0] - pb.bbox[2]) > unit * max_hgap_ratio
        else:
            too_far = (cb.bbox[1] - pb.bbox[3]) > unit * max_vgap_ratio

        if too_far:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    return groups


def denorm_bbox(bbox_norm: list[float], page_w: int, page_h: int) -> BBox:
    """정규화 좌표 [x1,y1,x2,y2] 를 픽셀 좌표로 변환한다."""
    x1, y1, x2, y2 = bbox_norm
    px = (
        int(round(min(x1, x2) * page_w)),
        int(round(min(y1, y2) * page_h)),
        int(round(max(x1, x2) * page_w)),
        int(round(max(y1, y2) * page_h)),
    )
    # 폭/높이가 0 이 되는 퇴화 케이스 방어
    return (px[0], px[1], max(px[2], px[0] + 1), max(px[3], px[1] + 1))
