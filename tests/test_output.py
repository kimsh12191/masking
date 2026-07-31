"""결과 저장 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pii_pipeline.output import IMAGE_SUFFIX, JSON_SUFFIX, format_summary, save_result
from pii_pipeline.schema import OcrBox, OcrStatus, PageResult, PiiRegion, Source

pytest.importorskip("PIL", reason="Pillow 미설치 환경에서는 건너뛴다")

PAGE_W, PAGE_H = 400, 500


def blank_image():
    from PIL import Image

    return Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))


def sample_result(with_image: bool = True) -> PageResult:
    return PageResult(
        image_path="/data/doc_001.png",
        width=PAGE_W,
        height=PAGE_H,
        ocr_boxes=[OcrBox(index=0, bbox=(20, 20, 180, 50), text="성명")],
        regions=[
            PiiRegion(
                id="r001", type="NAME", bbox=(200, 20, 360, 50),
                source=Source.LLM_PASS1, confidence=0.93, text="홍길동",
                member_index=[1],
            ),
            PiiRegion(
                id="r002", type="SIGNATURE", bbox=(200, 300, 380, 370),
                source=Source.VLM_GROUNDING, confidence=0.7,
                coarse=True, needs_review=True, ocr_status=OcrStatus.FAILED,
            ),
        ],
        timings={"ocr": 1.234, "total": 5.678},
        image=blank_image() if with_image else None,
    )


class TestSaveResult:
    def test_writes_both_files(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path)
        assert set(written) == {"json", "image"}
        assert written["json"].name == f"doc_001{JSON_SUFFIX}"
        assert written["image"].name == f"doc_001{IMAGE_SUFFIX}"
        assert written["json"].is_file()
        assert written["image"].is_file()

    def test_stem_defaults_to_input_filename(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path)
        assert written["json"].stem == "doc_001"

    def test_explicit_stem_overrides(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path, stem="page_07")
        assert written["json"].name == f"page_07{JSON_SUFFIX}"
        assert written["image"].name == f"page_07{IMAGE_SUFFIX}"

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested"
        save_result(sample_result(), target)
        assert target.is_dir()

    def test_json_contains_boxes_and_types(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path)
        data = json.loads(written["json"].read_text(encoding="utf-8"))
        assert data["page"] == {"width": PAGE_W, "height": PAGE_H}
        assert [r["type"] for r in data["regions"]] == ["NAME", "SIGNATURE"]
        assert data["regions"][0]["bbox"] == [200, 20, 360, 50]

    def test_json_excludes_transient_image(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path)
        raw = written["json"].read_text(encoding="utf-8")
        assert "image" not in json.loads(raw)  # 최상위에 image 키가 없어야 한다
        assert "image_path" in json.loads(raw)

    def test_saved_image_matches_page_size(self, tmp_path: Path) -> None:
        from PIL import Image

        written = save_result(sample_result(), tmp_path)
        with Image.open(written["image"]) as img:
            assert img.size == (PAGE_W, PAGE_H)

    def test_no_image_flag_writes_json_only(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path, write_image=False)
        assert set(written) == {"json"}
        assert not (tmp_path / f"doc_001{IMAGE_SUFFIX}").exists()

    def test_include_ocr_adds_debug_keys(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path, include_ocr=True)
        data = json.loads(written["json"].read_text(encoding="utf-8"))
        assert "ocr_boxes" in data and "raw_llm" in data

    def test_releases_image_by_default(self, tmp_path: Path) -> None:
        result = sample_result()
        save_result(result, tmp_path)
        assert result.image is None

    def test_release_can_be_disabled(self, tmp_path: Path) -> None:
        result = sample_result()
        save_result(result, tmp_path, release=False)
        assert result.image is not None

    def test_missing_image_warns_but_json_still_written(self, tmp_path: Path) -> None:
        result = sample_result(with_image=False)
        written = save_result(result, tmp_path)
        assert set(written) == {"json"}
        assert written["json"].is_file()
        assert any("박스 표시 이미지" in w for w in result.warnings)

    def test_overwrites_existing_files(self, tmp_path: Path) -> None:
        save_result(sample_result(), tmp_path)
        first = (tmp_path / f"doc_001{JSON_SUFFIX}").read_text(encoding="utf-8")
        result = sample_result()
        result.regions = []
        save_result(result, tmp_path)
        second = (tmp_path / f"doc_001{JSON_SUFFIX}").read_text(encoding="utf-8")
        assert first != second


class TestDefaultStem:
    def test_single_image_uses_filename(self) -> None:
        from pii_pipeline.output import default_stem

        assert default_stem(sample_result()) == "doc_001"

    def test_pdf_page_gets_zero_padded_suffix(self) -> None:
        from pii_pipeline.output import default_stem

        result = sample_result()
        result.image_path = "/data/계약서.pdf"
        result.page_no = 7
        assert default_stem(result) == "계약서_p007"

    def test_padding_keeps_file_sort_order(self) -> None:
        """파일 정렬 순서가 페이지 순서와 일치해야 한다."""
        from pii_pipeline.output import default_stem

        stems = []
        for n in (2, 10, 1):
            r = sample_result()
            r.image_path = "/data/doc.pdf"
            r.page_no = n
            stems.append(default_stem(r))
        assert sorted(stems) == ["doc_p001", "doc_p002", "doc_p010"]


class TestPdfPageSaving:
    def test_filenames_include_page_number(self, tmp_path: Path) -> None:
        result = sample_result()
        result.image_path = "/data/계약서.pdf"
        result.page_no = 3
        written = save_result(result, tmp_path)
        assert written["json"].name == "계약서_p003.json"
        assert written["image"].name == "계약서_p003.boxes.png"

    def test_json_carries_page_number(self, tmp_path: Path) -> None:
        result = sample_result()
        result.image_path = "/data/doc.pdf"
        result.page_no = 5
        written = save_result(result, tmp_path)
        assert json.loads(written["json"].read_text(encoding="utf-8"))["page_no"] == 5

    def test_single_image_page_no_is_null(self, tmp_path: Path) -> None:
        written = save_result(sample_result(), tmp_path)
        assert json.loads(written["json"].read_text(encoding="utf-8"))["page_no"] is None

    def test_pages_do_not_overwrite_each_other(self, tmp_path: Path) -> None:
        for n in (1, 2, 3):
            result = sample_result()
            result.image_path = "/data/doc.pdf"
            result.page_no = n
            save_result(result, tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.json")) == [
            "doc_p001.json", "doc_p002.json", "doc_p003.json",
        ]


class TestFormatSummary:
    def test_includes_paths_and_counts(self, tmp_path: Path) -> None:
        result = sample_result()
        written = save_result(result, tmp_path)
        text = format_summary(result, written)
        assert "doc_001" in text
        assert "탐지 영역 2" in text
        assert "검토필요 1" in text
        assert str(written["image"]) in text

    def test_works_without_written_paths(self) -> None:
        text = format_summary(sample_result())
        assert "탐지 영역 2" in text

    def test_lists_warnings(self) -> None:
        result = sample_result()
        result.warnings.append("pass2 실패: connection refused")
        assert "connection refused" in format_summary(result)


class TestReleaseImage:
    def test_clears_reference(self) -> None:
        result = sample_result()
        assert result.image is not None
        result.release_image()
        assert result.image is None
