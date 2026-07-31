"""PDF 입력 테스트.

``parse_page_range`` 는 의존성 없이 검증하고, 렌더링은 pypdfium2 가 있을 때만
검증한다 (PIL 로 만든 다중 페이지 PDF 를 왕복시킨다).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_pipeline.pdf import (
    MAX_SCALE,
    POINTS_PER_INCH,
    _scale_for,
    is_pdf,
    page_count,
    parse_page_range,
    render_pages,
)

A4_PT = (595.2, 841.9)


class TestIsPdf:
    @pytest.mark.parametrize("name", ["a.pdf", "a.PDF", "경로/문서.Pdf", "/x/y.pdf"])
    def test_true(self, name: str) -> None:
        assert is_pdf(name) is True

    @pytest.mark.parametrize("name", ["a.png", "a.pdf.png", "a", "a.jpg"])
    def test_false(self, name: str) -> None:
        assert is_pdf(name) is False


class TestParsePageRange:
    def test_empty_means_all(self) -> None:
        assert parse_page_range("", 3) == [1, 2, 3]
        assert parse_page_range("   ", 3) == [1, 2, 3]

    def test_single(self) -> None:
        assert parse_page_range("2", 5) == [2]

    def test_closed_range(self) -> None:
        assert parse_page_range("2-4", 10) == [2, 3, 4]

    def test_open_end(self) -> None:
        assert parse_page_range("3-", 5) == [3, 4, 5]

    def test_open_start(self) -> None:
        assert parse_page_range("-3", 5) == [1, 2, 3]

    def test_mixed_list(self) -> None:
        assert parse_page_range("1-3,7,9-", 10) == [1, 2, 3, 7, 9, 10]

    def test_deduplicates_and_sorts(self) -> None:
        assert parse_page_range("5,1-2,5,2", 10) == [1, 2, 5]

    def test_clamps_upper_bound(self) -> None:
        assert parse_page_range("2-99", 4) == [2, 3, 4]

    def test_ignores_empty_chunks(self) -> None:
        assert parse_page_range("1,,3", 5) == [1, 3]

    @pytest.mark.parametrize("spec", ["0", "0-2", "3-1", "-", "a", "1-a", "1,,x"])
    def test_invalid_raises(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_page_range(spec, 10)

    def test_start_beyond_document_raises(self) -> None:
        with pytest.raises(ValueError, match="벗어납니다"):
            parse_page_range("9-10", 4)


class TestScaleFor:
    def test_hits_target_long_side(self) -> None:
        scale = _scale_for(*A4_PT, 2480)
        assert round(A4_PT[1] * scale) == pytest.approx(2480, abs=2)

    def test_none_target_uses_144dpi(self) -> None:
        assert _scale_for(*A4_PT, None) == 2.0
        assert _scale_for(*A4_PT, None) * POINTS_PER_INCH == 144.0

    def test_capped_at_max_scale(self) -> None:
        assert _scale_for(10.0, 10.0, 100000) == MAX_SCALE

    def test_has_lower_bound(self) -> None:
        assert _scale_for(*A4_PT, 1) >= 0.5

    def test_zero_size_page_falls_back(self) -> None:
        assert _scale_for(0.0, 0.0, 2480) == 2.0

    def test_landscape_uses_long_side(self) -> None:
        portrait = _scale_for(595.2, 841.9, 2480)
        landscape = _scale_for(841.9, 595.2, 2480)
        assert portrait == pytest.approx(landscape)


# --------------------------------------------------------------------------
# 렌더링 (pypdfium2 필요)
# --------------------------------------------------------------------------

pdfium = pytest.importorskip("pypdfium2", reason="pypdfium2 미설치 환경에서는 건너뛴다")
pytest.importorskip("PIL", reason="Pillow 미설치")
pytest.importorskip("numpy", reason="numpy 미설치 (PDF -> BGR 변환에 필요)")


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    from PIL import Image

    pages = [
        Image.new("RGB", (1240, 1754), (255, 255 - 30 * i, 255 - 30 * i))
        for i in range(3)
    ]
    path = tmp_path / "doc.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=150)
    return path


class TestPageCount:
    def test_counts_pages(self, three_page_pdf: Path) -> None:
        assert page_count(three_page_pdf) == 3


class TestRenderPages:
    def test_renders_every_page_in_order(self, three_page_pdf: Path) -> None:
        pages = list(render_pages(three_page_pdf, target_long_side=1000))
        assert [p.page_no for p in pages] == [1, 2, 3]

    def test_page_numbers_are_one_based(self, three_page_pdf: Path) -> None:
        first = next(iter(render_pages(three_page_pdf, target_long_side=800)))
        assert first.page_no == 1

    def test_respects_target_long_side(self, three_page_pdf: Path) -> None:
        page = next(iter(render_pages(three_page_pdf, target_long_side=1200)))
        assert max(page.width, page.height) == pytest.approx(1200, abs=4)

    def test_image_is_bgr_ndarray(self, three_page_pdf: Path) -> None:
        page = next(iter(render_pages(three_page_pdf, target_long_side=600)))
        assert page.image.ndim == 3
        assert page.image.shape[2] == 3
        assert page.image.shape[:2] == (page.height, page.width)

    def test_records_scale_and_dpi(self, three_page_pdf: Path) -> None:
        page = next(iter(render_pages(three_page_pdf, target_long_side=1200)))
        assert page.scale > 0
        assert page.dpi == pytest.approx(page.scale * POINTS_PER_INCH, abs=0.2)

    def test_page_range_is_applied(self, three_page_pdf: Path) -> None:
        pages = list(render_pages(three_page_pdf, target_long_side=600, pages="2-3"))
        assert [p.page_no for p in pages] == [2, 3]

    def test_single_page_selection(self, three_page_pdf: Path) -> None:
        pages = list(render_pages(three_page_pdf, target_long_side=600, pages="2"))
        assert [p.page_no for p in pages] == [2]

    def test_is_lazy_generator(self, three_page_pdf: Path) -> None:
        """페이지를 전부 메모리에 모으지 않아야 한다 (장당 ~13MB)."""
        import types

        gen = render_pages(three_page_pdf, target_long_side=600)
        assert isinstance(gen, types.GeneratorType)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="찾을 수 없습니다"):
            list(render_pages(tmp_path / "nope.pdf"))

    def test_non_pdf_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "fake.pdf"
        bad.write_bytes(b"not a pdf at all")
        with pytest.raises(RuntimeError, match="열 수 없습니다"):
            list(render_pages(bad))

    def test_invalid_page_range_raises(self, three_page_pdf: Path) -> None:
        with pytest.raises(ValueError):
            list(render_pages(three_page_pdf, pages="9-10"))
