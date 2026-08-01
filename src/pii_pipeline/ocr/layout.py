"""좌표 유틸 및 크롭 내 읽기 순서 정렬.

이전 구조에서는 이 모듈이 **페이지 전체**의 읽기 순서를 만들어야 했다. 다단
컬럼 서식에서 라벨과 값을 짝지어 LLM 에 텍스트로 넘기려면 순서가 정확해야
했고, 순서가 꼬이면 판단이 그대로 틀렸다.

지금은 그 부담이 없다. LLM 은 이미지를 직접 보므로 읽기 순서를 코드가 알려줄
필요가 없고, 여기서 정렬하는 것은 **크롭 하나 안의 몇 줄**뿐이다. 값이 여러
줄/여러 박스로 쪼개졌을 때 이어붙일 순서만 맞으면 된다.
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


def denorm_bbox(bbox_norm: tuple[float, ...] | list[float], w: int, h: int) -> BBox:
    """정규화 좌표 ``[x1,y1,x2,y2]`` 를 픽셀 좌표로 변환한다.

    폭/높이가 0 이 되는 퇴화 케이스는 1px 로 보정한다 — 면적 0 박스는
    크롭에서 빈 배열이 되어 OCR 이 예외를 던진다.
    """
    x1, y1, x2, y2 = bbox_norm
    px = (
        int(round(min(x1, x2) * w)),
        int(round(min(y1, y2) * h)),
        int(round(max(x1, x2) * w)),
        int(round(max(y1, y2) * h)),
    )
    return (px[0], px[1], max(px[2], px[0] + 1), max(px[3], px[1] + 1))


def norm_bbox(bbox: BBox, w: int, h: int) -> tuple[float, float, float, float]:
    """픽셀 좌표를 정규화 좌표로 변환한다 (``denorm_bbox`` 의 역)."""
    if w <= 0 or h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)


def median_height(boxes: list[OcrBox]) -> float:
    heights = [b.height for b in boxes if b.height > 0]
    return float(median(heights)) if heights else 1.0


def sort_reading_order(boxes: list[OcrBox], y_tol_ratio: float = 0.6) -> list[OcrBox]:
    """행 클러스터링 후 읽기 순서로 정렬하고 ``index`` 를 0부터 다시 부여한다.

    같은 행 판정은 세로 중심 좌표 차이가 ``중위 글자높이 * y_tol_ratio`` 이내인지로
    한다. 폰트 크기가 섞여 있어도 안정적이다.

    Args:
        boxes: OCR 박스 목록 (순서 무관). **입력 객체의 ``index`` 가 변경된다.**
        y_tol_ratio: 같은 행으로 볼 세로 허용오차 비율.

    Returns:
        정렬된 목록. 빈 입력이면 빈 목록.
    """
    if not boxes:
        return []

    tol = median_height(boxes) * y_tol_ratio
    ordered = sorted(boxes, key=lambda b: ((b.bbox[1] + b.bbox[3]) / 2.0, b.bbox[0]))

    rows: list[list[OcrBox]] = [[ordered[0]]]
    row_ref = (ordered[0].bbox[1] + ordered[0].bbox[3]) / 2.0

    for box in ordered[1:]:
        cy = (box.bbox[1] + box.bbox[3]) / 2.0
        if abs(cy - row_ref) <= tol:
            rows[-1].append(box)
        else:
            rows.append([box])
            row_ref = cy

    result: list[OcrBox] = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b.bbox[0]))
    for i, box in enumerate(result):
        box.index = i
    return result
