"""읽기 순서 정렬 및 좌표 유틸 테스트."""

from __future__ import annotations

import pytest

from pii_pipeline.ocr.layout import (
    assign_reading_order,
    denorm_bbox,
    pad_bbox,
    quad_to_bbox,
    spatially_split,
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

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            union_bbox([])


class TestPadBbox:
    def test_expands(self) -> None:
        assert pad_bbox((10, 10, 50, 30), 3, 200, 200) == (7, 7, 53, 33)

    def test_clamps_to_page(self) -> None:
        assert pad_bbox((1, 1, 199, 199), 5, 200, 200) == (0, 0, 200, 200)


class TestAssignReadingOrder:
    def test_empty(self) -> None:
        assert assign_reading_order([]) == []

    def test_single_row_sorted_by_x(self) -> None:
        boxes = [box(300, 100, 400, 130, "C"), box(100, 102, 200, 132, "A"),
                 box(200, 101, 290, 131, "B")]
        out = assign_reading_order(boxes)
        assert [b.text for b in out] == ["A", "B", "C"]
        assert [b.index for b in out] == [0, 1, 2]
        assert {b.row for b in out} == {0}

    def test_two_column_form_does_not_interleave_rows(self) -> None:
        """다단 서식: 같은 행의 라벨-값이 붙어 있어야 한다."""
        boxes = [
            box(100, 100, 180, 130, "성명"),
            box(300, 102, 420, 132, "홍길동"),
            box(100, 160, 260, 190, "주민등록번호"),
            box(300, 161, 460, 191, "901231-1234563"),
        ]
        out = assign_reading_order(boxes)
        assert [b.text for b in out] == ["성명", "홍길동", "주민등록번호", "901231-1234563"]
        assert [b.row for b in out] == [0, 0, 1, 1]

    def test_slight_vertical_jitter_stays_same_row(self) -> None:
        # 30px 높이 기준 허용오차 18px 안쪽
        boxes = [box(100, 100, 200, 130, "A"), box(300, 108, 400, 138, "B")]
        out = assign_reading_order(boxes)
        assert [b.row for b in out] == [0, 0]

    def test_large_vertical_gap_splits_row(self) -> None:
        boxes = [box(100, 100, 200, 130, "A"), box(300, 200, 400, 230, "B")]
        out = assign_reading_order(boxes)
        assert [b.row for b in out] == [0, 1]

    def test_reassigns_index_contiguously(self) -> None:
        boxes = [box(0, 0, 10, 10) for _ in range(5)]
        out = assign_reading_order(boxes)
        assert [b.index for b in out] == [0, 1, 2, 3, 4]


class TestSpatiallySplit:
    def _address_boxes(self) -> list[OcrBox]:
        boxes = [
            box(100, 100, 260, 130, "서울특별시 강남구"),
            box(270, 100, 430, 130, "테헤란로 123"),
            box(100, 140, 300, 170, "○○빌딩 5층"),
            box(100, 800, 300, 830, "무관한 박스"),
        ]
        return assign_reading_order(boxes)

    def test_adjacent_group_kept_together(self) -> None:
        boxes = self._address_boxes()
        assert spatially_split([0, 1, 2], boxes) == [[0, 1, 2]]

    def test_distant_box_is_split_off(self) -> None:
        """멀리 떨어진 박스를 LLM 이 잘못 묶은 경우 분해되어야 한다."""
        boxes = self._address_boxes()
        assert spatially_split([0, 1, 3], boxes) == [[0, 1], [3]]

    def test_out_of_range_dropped(self) -> None:
        boxes = self._address_boxes()
        assert spatially_split([0, 99, -1], boxes) == [[0]]

    def test_all_invalid_returns_empty(self) -> None:
        boxes = self._address_boxes()
        assert spatially_split([50, 60], boxes) == []

    def test_unsorted_input_is_normalized(self) -> None:
        boxes = self._address_boxes()
        assert spatially_split([2, 0, 1], boxes) == [[0, 1, 2]]

    def test_same_row_but_far_apart_horizontally_is_split(self) -> None:
        """같은 행이어도 서식 양 끝에 있는 박스는 한 항목이 아니다."""
        boxes = assign_reading_order(
            [
                box(100, 100, 200, 130, "왼쪽끝"),
                box(1600, 100, 1800, 130, "오른쪽끝"),
            ]
        )
        assert spatially_split([0, 1], boxes) == [[0], [1]]

    def test_row_index_gap_alone_does_not_split(self) -> None:
        """행 번호는 2칸 차이지만 물리적으로는 붙어 있으므로 유지되어야 한다."""
        boxes = assign_reading_order(
            [
                box(100, 100, 300, 130, "A"),
                box(100, 135, 300, 165, "B"),
                box(100, 170, 300, 200, "C"),
            ]
        )
        assert [b.row for b in boxes] == [0, 1, 2]
        assert spatially_split([0, 1, 2], boxes) == [[0, 1, 2]]


class TestDenormBbox:
    def test_scales_to_pixels(self) -> None:
        assert denorm_bbox([0.1, 0.2, 0.5, 0.4], 1000, 2000) == (100, 400, 500, 800)

    def test_normalizes_inverted_coords(self) -> None:
        assert denorm_bbox([0.5, 0.4, 0.1, 0.2], 1000, 2000) == (100, 400, 500, 800)

    def test_degenerate_box_gets_minimum_size(self) -> None:
        out = denorm_bbox([0.5, 0.5, 0.5, 0.5], 1000, 1000)
        assert out[2] > out[0] and out[3] > out[1]
