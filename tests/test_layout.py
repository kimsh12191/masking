"""좌표 유틸 및 크롭 내 읽기 순서 정렬 테스트."""

from __future__ import annotations

import pytest

from pii_pipeline.ocr.layout import (
    denorm_bbox,
    median_height,
    norm_bbox,
    pad_bbox,
    quad_to_bbox,
    sort_reading_order,
    union_bbox,
)
from pii_pipeline.schema import OcrBox


def box(x1: int, y1: int, x2: int, y2: int, text: str = "t") -> OcrBox:
    return OcrBox(index=-1, bbox=(x1, y1, x2, y2), text=text)


class TestQuadToBbox:
    def test_axis_aligned(self) -> None:
        assert quad_to_bbox([[10, 20], [110, 20], [110, 50], [10, 50]]) == (10, 20, 110, 50)

    def test_rotated_quad_takes_extremes(self) -> None:
        # 기울어진 사각형은 외접 사각형으로
        assert quad_to_bbox([[12, 18], [108, 24], [110, 52], [14, 46]]) == (12, 18, 110, 52)

    def test_floors_min_and_ceils_max(self) -> None:
        """경계 글자가 잘리지 않도록 바깥쪽으로 확장한다."""
        assert quad_to_bbox([[9.6, 19.4], [110.2, 19.4], [110.2, 50.5], [9.6, 50.5]]) == (
            9, 19, 111, 51,
        )


class TestUnionBbox:
    def test_merges(self) -> None:
        assert union_bbox([(10, 10, 50, 30), (60, 12, 100, 32)]) == (10, 10, 100, 32)

    def test_single(self) -> None:
        assert union_bbox([(10, 10, 50, 30)]) == (10, 10, 50, 30)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="빈 목록"):
            union_bbox([])


class TestPadBbox:
    def test_expands(self) -> None:
        assert pad_bbox((50, 50, 100, 80), 5, 200, 200) == (45, 45, 105, 85)

    def test_clamps_to_page(self) -> None:
        assert pad_bbox((2, 2, 198, 198), 10, 200, 200) == (0, 0, 200, 200)


class TestDenormBbox:
    def test_roundtrip(self) -> None:
        assert denorm_bbox((0.1, 0.2, 0.5, 0.4), 1000, 500) == (100, 100, 500, 200)

    def test_sorts_reversed_coords(self) -> None:
        """x2<x1 로 뒤집혀 와도 정렬해서 살린다."""
        assert denorm_bbox((0.5, 0.4, 0.1, 0.2), 1000, 500) == (100, 100, 500, 200)

    def test_degenerate_gets_one_pixel(self) -> None:
        """면적 0 은 크롭에서 빈 배열이 되어 OCR 이 죽는다."""
        assert denorm_bbox((0.5, 0.5, 0.5, 0.5), 100, 100) == (50, 50, 51, 51)


class TestNormBbox:
    def test_inverse_of_denorm(self) -> None:
        assert norm_bbox((100, 100, 500, 200), 1000, 500) == (0.1, 0.2, 0.5, 0.4)

    def test_zero_page_is_safe(self) -> None:
        assert norm_bbox((1, 2, 3, 4), 0, 0) == (0.0, 0.0, 0.0, 0.0)


class TestMedianHeight:
    def test_ignores_zero_height(self) -> None:
        assert median_height([box(0, 0, 10, 0), box(0, 0, 10, 20), box(0, 0, 10, 40)]) == 30.0

    def test_empty_returns_one(self) -> None:
        assert median_height([]) == 1.0


class TestSortReadingOrder:
    def test_orders_rows_then_columns(self) -> None:
        boxes = [
            box(400, 100, 500, 130, "b"),
            box(100, 100, 200, 130, "a"),
            box(100, 200, 200, 230, "c"),
        ]
        assert [b.text for b in sort_reading_order(boxes)] == ["a", "b", "c"]

    def test_reassigns_index_from_zero(self) -> None:
        boxes = [box(400, 100, 500, 130, "b"), box(100, 100, 200, 130, "a")]
        assert [b.index for b in sort_reading_order(boxes)] == [0, 1]

    def test_slight_vertical_jitter_stays_one_row(self) -> None:
        """스캔 기울기로 몇 px 어긋난 것은 같은 행이다."""
        boxes = [
            box(400, 104, 500, 134, "b"),
            box(100, 100, 200, 130, "a"),
            box(700, 97, 800, 127, "c"),
        ]
        assert [b.text for b in sort_reading_order(boxes)] == ["a", "b", "c"]

    def test_empty(self) -> None:
        assert sort_reading_order([]) == []
