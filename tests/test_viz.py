"""시각화 도구 스모크 테스트.

그려진 픽셀을 검증하지는 않고, 모든 티어/플래그 조합에서 예외 없이 동작하는지만 본다.
"""

from __future__ import annotations

import pytest

from pii_pipeline.schema import (
    Agreement,
    OcrBox,
    OcrStatus,
    PageResult,
    PiiRegion,
    Source,
    VlmFinding,
)
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
        findings=[
            VlmFinding(text="901231-1234563", type="RRN", bbox_norm=(0.06, 0.12, 0.5, 0.17)),
            VlmFinding(text="홍길동", type="NAME", bbox_norm=(0.36, 0.05, 0.63, 0.09)),
            VlmFinding(text="김철수", type="NAME", bbox_norm=(0.36, 0.37, 0.63, 0.42)),
            VlmFinding(text="M12345678", type="PASSPORT", bbox_norm=(0.36, 0.62, 0.76, 0.71)),
        ],
        regions=[
            # 체크섬 통과 + 두 엔진 일치 — 검수자가 넘겨도 되는 건
            PiiRegion(
                id="r001", type="RRN", bbox=(40, 100, 300, 132),
                source=Source.OCR_REFINED, confidence=1.0, member_index=[1],
                text="901231-1234563", vlm_text="901231-1234563",
                verified=True, checksum="ok", agreement=Agreement.EXACT,
            ),
            # 좌표는 확정, 체크섬 없는 라벨
            PiiRegion(
                id="r002", type="NAME", bbox=(220, 40, 380, 70),
                source=Source.OCR_REFINED, confidence=0.93, member_index=[1],
                text="홍길동", vlm_text="홍길동", agreement=Agreement.EXACT,
            ),
            # 두 엔진 불일치 — 검토 필요
            PiiRegion(
                id="r003", type="NAME", bbox=(220, 300, 380, 330),
                source=Source.OCR_REFINED, confidence=0.85, member_index=[2],
                text="김철둥", vlm_text="김철수",
                ocr_status=OcrStatus.LOW_CONF, agreement=Agreement.NONE,
                needs_review=True,
            ),
            # 좌표 근사 (크롭에서 OCR 박스가 안 나왔다 — 손글씨 추정)
            PiiRegion(
                id="r004", type="PASSPORT", bbox=(220, 500, 460, 570),
                source=Source.VLM_COARSE, confidence=0.61,
                vlm_text="M12345678", coarse=True, needs_review=True,
                agreement=Agreement.NONE,
            ),
            # 체크섬 미통과 — 오독 의심
            PiiRegion(
                id="r005", type="RRN", bbox=(40, 620, 300, 652),
                source=Source.OCR_REFINED, confidence=0.7,
                text="901231-1234561", vlm_text="901231-1234561",
                checksum="failed", needs_review=True, agreement=Agreement.EXACT,
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

    def test_label_never_covers_box_interior(self) -> None:
        """라벨 배지가 박스 안을 덮으면 내용을 검수할 수 없다.

        박스 내부(테두리 안쪽)는 원본 픽셀이 그대로 남아 있어야 한다.
        """
        from PIL import Image

        src = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
        result = sample_result()
        result.regions = [result.regions[0]]
        result.regions[0].bbox = (100, 200, 400, 260)
        out = draw_overlay(src, result, line_width=3)

        x1, y1, x2, y2 = result.regions[0].bbox
        inset = 6  # 테두리 두께보다 넉넉히 안쪽
        for y in range(y1 + inset, y2 - inset):
            for x in range(x1 + inset, x2 - inset):
                assert out.getpixel((x, y)) == (255, 255, 255), (
                    f"({x},{y}) 가 덮였습니다 — 라벨이 박스 내부를 침범했습니다"
                )

    def test_label_falls_below_when_no_room_above(self) -> None:
        """위쪽에 자리가 없으면 라벨을 아래로 내려야 한다 (내부를 덮지 않도록)."""
        from PIL import Image

        src = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
        result = sample_result()
        result.regions = [result.regions[0]]
        result.regions[0].bbox = (100, 0, 400, 60)   # y1 == 0 → 위쪽 여백 없음
        out = draw_overlay(src, result, line_width=3)

        # 박스 아래쪽에 배지 색이 나타나야 한다
        color = SOURCE_COLORS[result.regions[0].source]
        below = [out.getpixel((105, y)) for y in range(61, 100)]
        assert color in below

    def test_wide_label_does_not_overflow_canvas(self) -> None:
        result = sample_result()
        result.regions = [result.regions[0]]
        result.regions[0].bbox = (PAGE_W - 20, 100, PAGE_W - 5, 130)
        out = draw_overlay(blank_image(), result)
        assert out.size == (PAGE_W, PAGE_H)

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
