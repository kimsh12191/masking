"""시각화 도구 스모크 테스트.

그려진 픽셀을 검증하지는 않고, 모든 티어/플래그 조합에서 예외 없이 동작하는지만 본다.
"""

from __future__ import annotations

import pytest

from pii_pipeline.schema import OcrBox, OcrStatus, PageResult, PiiRegion, Source
from pii_pipeline.viz import SOURCE_COLORS, draw_overlay, legend_text

pytest.importorskip("PIL", reason="Pillow 미설치 환경에서는 건너뛴다")

PAGE_W, PAGE_H = 600, 800


def sample_result() -> PageResult:
    return PageResult(
        image_path="fake.png",
        width=PAGE_W,
        height=PAGE_H,
        ocr_boxes=[
            OcrBox(index=0, bbox=(40, 40, 200, 70), text="성명"),
            OcrBox(index=1, bbox=(220, 40, 380, 70), text="홍길동"),
            OcrBox(index=2, bbox=(220, 300, 380, 330), text="", status=OcrStatus.FAILED),
        ],
        regions=[
            PiiRegion(
                id="r001", type="RULE_ITEM_RRN", bbox=(40, 100, 300, 132),
                source=Source.RULE, confidence=1.0, member_index=[1],
            ),
            PiiRegion(
                id="r002", type="NAME", bbox=(220, 40, 380, 70),
                source=Source.LLM_PASS1, confidence=0.93, member_index=[1],
            ),
            PiiRegion(
                id="r003", type="NAME", bbox=(220, 300, 380, 330),
                source=Source.VLM_PASS2, confidence=0.85, member_index=[2],
                ocr_status=OcrStatus.FAILED, needs_review=True,
            ),
            PiiRegion(
                id="r004", type="SIGNATURE", bbox=(220, 500, 460, 570),
                source=Source.VLM_GROUNDING, confidence=0.61,
                coarse=True, needs_review=True, low_confidence=False,
            ),
        ],
    )


def blank_image():
    from PIL import Image

    return Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))


class TestDrawOverlay:
    def test_returns_image_of_same_size(self) -> None:
        out = draw_overlay(blank_image(), sample_result())
        assert out.size == (PAGE_W, PAGE_H)

    def test_does_not_mutate_source_image(self) -> None:
        src = blank_image()
        before = src.tobytes()
        draw_overlay(src, sample_result())
        assert src.tobytes() == before

    def test_draws_something(self) -> None:
        out = draw_overlay(blank_image(), sample_result())
        assert out.convert("RGB").getcolors(maxcolors=1 << 20) is not None
        # 흰 배경만 있지는 않아야 한다
        colors = {c for _, c in out.convert("RGB").getcolors(maxcolors=1 << 20)}
        assert len(colors) > 1

    def test_with_ocr_boxes(self) -> None:
        out = draw_overlay(blank_image(), sample_result(), show_ocr_boxes=True)
        assert out.size == (PAGE_W, PAGE_H)

    def test_region_at_top_edge_does_not_crash(self) -> None:
        """라벨을 박스 위에 그리므로 y=0 근처에서 좌표가 음수가 될 수 있다."""
        result = sample_result()
        result.regions[0].bbox = (0, 0, 120, 20)
        draw_overlay(blank_image(), result)

    def test_no_regions(self) -> None:
        result = sample_result()
        result.regions = []
        draw_overlay(blank_image(), result)

    def test_missing_font_path_falls_back(self) -> None:
        draw_overlay(blank_image(), sample_result(), font_path="/does/not/exist.ttf")

    def test_accepts_numpy_bgr_array(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8)
        out = draw_overlay(arr, sample_result())
        assert out.size == (PAGE_W, PAGE_H)


class TestLegend:
    def test_covers_every_source_tier(self) -> None:
        text = legend_text()
        for source in SOURCE_COLORS:
            assert source.value in text

    def test_mentions_review_flag(self) -> None:
        assert "needs_review" in legend_text()
