"""학습 데이터 생성 테스트.

이 모듈의 계약은 한 줄이다.

    **학습 라벨을 추론 디코딩에 통과시키면 원래 박스가 나와야 한다.**

이게 깨지면 모델에게 틀린 자리를 가르치게 되는데, 학습 결과만 보고는 원인을
찾을 수 없다 ("좌표가 안 좋아지네" 로만 보인다). 그래서 왕복 검산을 테스트로
못박아 둔다 — 전처리나 타일 분할이 바뀌면 여기서 먼저 터져야 한다.
"""

from __future__ import annotations

import random

import pytest

from pii_pipeline.detect import tile_rects
from pii_pipeline.schema import OcrBox, OcrStatus
from pii_pipeline.train.dataset import (
    GroundingConfig,
    build_tile_sample,
    quantization_limit,
    scale_regions,
    roundtrip,
    to_permille,
)

PAGE_W, PAGE_H = 1760, 2464
RECTS = tile_rects(PAGE_W, PAGE_H, 3, 0.08, 32)


def box(
    text: str, x1: int, y1: int, x2: int, y2: int, status=OcrStatus.OK, conf=0.95
) -> OcrBox:
    return OcrBox(index=-1, bbox=(x1, y1, x2, y2), text=text, status=status, rec_conf=conf)


def tile_of(y: int) -> tuple[int, tuple[float, float, float, float]]:
    """``y`` 를 온전히 담는 첫 타일."""
    for i, r in enumerate(RECTS):
        if r[1] * PAGE_H <= y - 30 and y + 30 <= r[3] * PAGE_H:
            return i, r
    raise AssertionError(f"y={y} 를 담는 타일이 없다")


# --------------------------------------------------------------------------
# 좌표 왕복 — 이 파일의 존재 이유
# --------------------------------------------------------------------------


class TestRoundtrip:
    @pytest.mark.parametrize("tile", range(3))
    @pytest.mark.parametrize("bbox", [
        (100, 60, 400, 100),      # 타일 위쪽
        (880, 300, 1300, 350),    # 가운데
        (1500, 700, 1740, 780),   # 오른쪽 끝
        (0, 10, 60, 50),          # 왼쪽 위 모서리
    ])
    def test_label_survives_the_inference_decoder(self, tile: int, bbox) -> None:
        """라벨 -> 추론 디코딩 -> 원래 박스. 오차는 per-mille 양자화뿐이어야 한다."""
        rect = RECTS[tile]
        # 타일 안쪽으로 옮긴다 (타일 밖 좌표는 클램프되므로 왕복이 성립하지 않는다)
        oy = int(rect[1] * PAGE_H)
        shifted = (bbox[0], bbox[1] + oy, bbox[2], bbox[3] + oy)

        permille = to_permille(shifted, rect, PAGE_W, PAGE_H)
        back = roundtrip(permille, rect, PAGE_W, PAGE_H)

        limit = quantization_limit(rect, PAGE_W, PAGE_H) + 1.0
        for got, want in zip(back, shifted, strict=True):
            assert abs(got - want) <= limit

    def test_permille_stays_in_range(self) -> None:
        """스키마가 0~1000 정수를 요구한다. 벗어나면 학습과 추론 규약이 달라진다."""
        rect = RECTS[1]
        oy = int(rect[1] * PAGE_H)
        for bbox in [(0, oy, 10, oy + 10), (1750, oy + 800, 1760, oy + 860)]:
            for v in to_permille(bbox, rect, PAGE_W, PAGE_H):
                assert 0 <= v <= 1000
                assert isinstance(v, int)

    def test_quantization_limit_tracks_tile_width(self) -> None:
        """허용치를 상수로 박으면 타일 크기가 바뀔 때 조용히 틀린다."""
        wide = (0.0, 0.0, 1.0, 0.25)
        narrow = (0.0, 0.0, 0.25, 0.25)
        assert quantization_limit(wide, PAGE_W, PAGE_H) > quantization_limit(
            narrow, PAGE_W, PAGE_H
        )


# --------------------------------------------------------------------------
# 무엇을 정답으로 삼는가
# --------------------------------------------------------------------------


class TestLabelSelection:
    def test_confident_box_teaches_text_and_position(self) -> None:
        i, rect = tile_of(200)
        s = build_tile_sample([box("강동혁", 300, 180, 460, 220)], i, rect, PAGE_W, PAGE_H)
        assert s is not None
        assert s.items[0]["text"] == "강동혁"
        assert s.n_textless == 0

    def test_failed_box_teaches_position_only(self) -> None:
        """손글씨·도장. det 는 성공했으므로 **좌표 라벨로는 멀쩡하다.**"""
        i, rect = tile_of(200)
        handwriting = box("", 300, 180, 460, 220, OcrStatus.FAILED, 0.1)
        s = build_tile_sample([handwriting], i, rect, PAGE_W, PAGE_H)
        assert s is not None
        assert s.items[0]["text"] == ""
        assert s.n_textless == 1

    def test_low_confidence_text_is_not_taught(self) -> None:
        """오독을 정답으로 주면 지금 잘하는 전사 능력이 망가진다."""
        i, rect = tile_of(200)
        blurry = box("흐릿한값", 300, 180, 460, 220, OcrStatus.LOW_CONF, 0.5)
        s = build_tile_sample([blurry], i, rect, PAGE_W, PAGE_H)
        assert s is not None
        assert s.items[0]["text"] == ""  # 좌표는 쓰고 글자는 버린다

    def test_textless_can_be_turned_off(self) -> None:
        i, rect = tile_of(200)
        handwriting = box("", 300, 180, 460, 220, OcrStatus.FAILED, 0.1)
        cfg = GroundingConfig(keep_textless=False)
        assert build_tile_sample([handwriting], i, rect, PAGE_W, PAGE_H, cfg) is None


# --------------------------------------------------------------------------
# 타일 경계
# --------------------------------------------------------------------------


class TestTileBoundary:
    def test_boxes_outside_the_tile_are_excluded(self) -> None:
        far = box("다른타일", 300, 2300, 460, 2340)
        s = build_tile_sample([far], 0, RECTS[0], PAGE_W, PAGE_H)
        assert s is None

    def test_straddling_boxes_are_dropped_not_clipped(self) -> None:
        """반쪽만 가르치면 모델이 잘린 박스를 배운다. 겹친 옆 타일이 온전히 잡는다."""
        edge_y = int(RECTS[0][3] * PAGE_H)
        straddling = box("경계", 300, edge_y - 20, 460, edge_y + 20)
        assert build_tile_sample([straddling], 0, RECTS[0], PAGE_W, PAGE_H) is None

        # 같은 줄이 다음 타일에는 온전히 들어간다 (overlap 이 줄 높이보다 크다)
        s = build_tile_sample([straddling], 1, RECTS[1], PAGE_W, PAGE_H)
        assert s is not None and s.items[0]["text"] == "경계"

    def test_tiny_boxes_are_noise(self) -> None:
        i, rect = tile_of(200)
        speck = box(".", 300, 180, 303, 183)
        assert build_tile_sample([speck], i, rect, PAGE_W, PAGE_H) is None


# --------------------------------------------------------------------------
# 샘플 구성
# --------------------------------------------------------------------------


class TestSampleShape:
    def test_items_are_in_reading_order(self) -> None:
        i, rect = tile_of(300)
        oy = int(rect[1] * PAGE_H)
        boxes = [
            box("셋", 100, oy + 200, 300, oy + 240),
            box("하나", 100, oy + 100, 300, oy + 140),
            box("둘", 400, oy + 100, 600, oy + 140),
        ]
        s = build_tile_sample(boxes, i, rect, PAGE_W, PAGE_H)
        assert s is not None
        assert [it["text"] for it in s.items] == ["하나", "둘", "셋"]

    def test_respects_the_schema_item_cap(self) -> None:
        """추론에서 낼 수 없는 길이를 학습시키면 안 된다 (VLM_SCHEMA.maxItems)."""
        i, rect = tile_of(300)
        oy = int(rect[1] * PAGE_H)
        boxes = [box(f"값{n}", 100, oy + 50 + n * 20, 300, oy + 66 + n * 20) for n in range(40)]
        cfg = GroundingConfig(max_items=32)
        s = build_tile_sample(boxes, i, rect, PAGE_W, PAGE_H, cfg)
        assert s is not None and len(s.items) == 32

    def test_empty_tile_yields_no_sample(self) -> None:
        assert build_tile_sample([], 0, RECTS[0], PAGE_W, PAGE_H) is None

    def test_target_json_matches_the_inference_schema(self) -> None:
        """학습 타깃의 모양이 추론 출력과 같아야 한다."""
        import json

        i, rect = tile_of(200)
        s = build_tile_sample([box("강동혁", 300, 180, 460, 220)], i, rect, PAGE_W, PAGE_H)
        assert s is not None
        payload = json.loads(s.target_json())
        assert set(payload) == {"findings"}
        assert set(payload["findings"][0]) == {"text", "bbox_2d"}
        assert len(payload["findings"][0]["bbox_2d"]) == 4


class TestScaleRegions:
    """입력 크기는 고정, **글자 크기만** 달라지게 하는 증강.

    캔버스를 고정해도 문서마다 폰트 크기가 다르다. 추론 타일 하나의 스케일만
    학습하면 그보다 작거나 큰 글씨에서 좌표가 흔들린다.
    """

    PAGE_W, PAGE_H = 1760, 2464
    TILE = (0.0, 0.0, 1.0, 960 / 2464)

    def regions(self, scales: list[float], seed: int = 0) -> list[tuple]:
        return scale_regions(
            self.TILE, self.PAGE_W, self.PAGE_H, scales, random.Random(seed)
        )

    def test_aspect_ratio_matches_the_tile(self) -> None:
        """종횡비가 다르면 타일 크기로 늘릴 때 글자가 찌그러진다."""
        tile_ar = (self.TILE[2] - self.TILE[0]) * self.PAGE_W / (
            (self.TILE[3] - self.TILE[1]) * self.PAGE_H
        )
        for r in self.regions([1.5, 2.0, 2.5]):
            ar = (r[2] - r[0]) * self.PAGE_W / ((r[3] - r[1]) * self.PAGE_H)
            assert ar == pytest.approx(tile_ar, rel=1e-6)

    def test_higher_scale_means_a_smaller_region(self) -> None:
        """작은 영역을 타일 크기로 키우니 글자가 그만큼 커 보인다."""
        r15, r20 = self.regions([1.5, 2.0])
        assert (r20[3] - r20[1]) < (r15[3] - r15[1])

    def test_regions_stay_inside_the_page(self) -> None:
        for r in self.regions([1.5, 2.0, 3.0], seed=7):
            assert 0.0 <= r[0] < r[2] <= 1.0
            assert 0.0 <= r[1] < r[3] <= 1.0

    def test_scale_one_or_less_is_skipped(self) -> None:
        """타일이 이미 페이지 폭 전체다. 더 넓은 영역은 없다."""
        assert self.regions([1.0, 0.5]) == []

    def test_deterministic_for_a_given_seed(self) -> None:
        """같은 시드면 같은 데이터셋이어야 재현이 된다."""
        assert self.regions([1.5, 2.0], seed=3) == self.regions([1.5, 2.0], seed=3)

    def test_labels_survive_the_roundtrip_in_an_augmented_region(self) -> None:
        """증강 영역에서도 좌표가 추론 경로로 정확히 복원되어야 한다."""
        boxes = [box("강동혁", 300, 200, 480, 250), box("서울시", 300, 400, 520, 450)]
        for rect in self.regions([2.0], seed=1):
            sample = build_tile_sample(boxes, -1, rect, self.PAGE_W, self.PAGE_H)
            if sample is None:
                continue
            limit = quantization_limit(rect, self.PAGE_W, self.PAGE_H) + 1.0
            assert sample.max_error_px <= limit


class TestLocateTask:
    """값을 주고 위치만 묻는 주 과제 — 질문은 중복 제거, 답은 모든 출현."""

    def sample(self, boxes: list) -> object:
        return build_tile_sample(boxes, 0, (0.0, 0.0, 1.0, 1.0), PAGE_W, PAGE_H)

    def test_query_drops_duplicates(self) -> None:
        """질문에 두 번 적으면 '두 번 물었으니 두 개' 를 배운다. 추론에서는
        값이 몇 번 나오는지 아무도 모르므로 쓸 수 없는 규칙이다."""
        s = self.sample([
            box("강동혁", 100, 100, 200, 140),
            box("강동혁", 100, 300, 200, 340),
            box("서울시", 100, 500, 220, 540),
        ])
        assert s.query() == ["강동혁", "서울시"]

    def test_answer_keeps_every_occurrence(self) -> None:
        """한 곳만 답하면 나머지 출현이 마스킹되지 않는다 — 가장 흔한 실패다."""
        s = self.sample([
            box("강동혁", 100, 100, 200, 140),
            box("강동혁", 100, 300, 200, 340),
        ])
        assert len(s.query()) == 1
        assert len(s.locate_items()) == 2
        assert {tuple(i["bbox_2d"]) for i in s.locate_items()}.__len__() == 2

    def test_query_keeps_first_seen_order(self) -> None:
        s = self.sample([
            box("나중", 100, 500, 200, 540),
            box("먼저", 100, 100, 200, 140),
        ])
        # items 는 읽기 순서로 정렬되므로 위쪽이 먼저다
        assert s.query() == ["먼저", "나중"]

    def test_textless_boxes_are_not_asked_or_answered(self) -> None:
        """도장·손글씨는 지목할 텍스트가 없다. 묻지 않은 것을 답하라고 가르치면
        모델이 질문에 없는 것을 지어낸다."""
        s = self.sample([
            box("강동혁", 100, 100, 200, 140),
            box("", 100, 300, 200, 340, status=OcrStatus.FAILED, conf=0.1),
        ])
        assert s.query() == ["강동혁"]
        assert len(s.locate_items()) == 1
        assert len(s.items) == 2  # read 과제용으로는 남아 있다
