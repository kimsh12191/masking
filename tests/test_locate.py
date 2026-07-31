"""② 좌표 확정 단계 테스트.

이 단계의 계약은 두 줄이다.

  1. VLM 의 **좌표**는 신뢰하지 않고, VLM 의 **텍스트**를 신뢰한다.
  2. 못 찾은 항목도 **버리지 않는다** — vlm_coarse 로 남기고 이유를 적는다.

두 번째가 특히 중요하다. 개인정보를 조용히 떨어뜨리는 것이 이 파이프라인에서
가장 위험한 실패다. findings 와 regions 는 항상 1:1 이어야 한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline.locate import (
    LocateConfig,
    _norm,
    crop_rect,
    find_value,
    locate,
)
from pii_pipeline.schema import Agreement, OcrBox, OcrStatus, Source, VlmFinding

np = pytest.importorskip("numpy", reason="numpy 미설치 환경에서는 건너뛴다")

PAGE_W, PAGE_H = 1000, 2000


def blank_page(w: int = PAGE_W, h: int = PAGE_H) -> Any:
    return np.zeros((h, w, 3), dtype=np.uint8)


def box(text: str, x1=0, y1=0, x2=100, y2=30, status=OcrStatus.OK, conf=0.95) -> OcrBox:
    return OcrBox(index=-1, bbox=(x1, y1, x2, y2), text=text, status=status, rec_conf=conf)


def finding(text: str, label: str = "NAME", bbox=(0.1, 0.1, 0.3, 0.15), conf=0.9) -> VlmFinding:
    return VlmFinding(text=text, type=label, field="성명", bbox_norm=bbox, conf=conf)


class FakeOcr:
    """크롭마다 미리 정해둔 박스를 돌려주는 가짜 OCR.

    ``results`` 는 호출 순서대로 소비된다. 좌표는 **크롭 기준**이므로 호출부가
    페이지 좌표로 환산하는지 확인할 수 있다.
    """

    def __init__(self, results: list[list[OcrBox]]) -> None:
        self.results = results
        self.batches: list[int] = []
        self.crop_shapes: list[tuple[int, int]] = []
        self._n = 0

    def run_many(self, images: list[Any]) -> list[list[OcrBox]]:
        self.batches.append(len(images))
        out: list[list[OcrBox]] = []
        for img in images:
            self.crop_shapes.append(img.shape[:2] if img is not None else (0, 0))
            out.append(self.results[self._n] if self._n < len(self.results) else [])
            self._n += 1
        return out


# --------------------------------------------------------------------------
# 정규화
# --------------------------------------------------------------------------


class TestNorm:
    def test_strips_separators(self) -> None:
        assert _norm("901231-1234563").text == "9012311234563"

    def test_folds_evasion_notation(self) -> None:
        assert _norm("공1공-1234-5678").text == _norm("010-1234-5678").text

    def test_span_maps_back_to_original_offsets(self) -> None:
        """char_span 은 원문 기준이어야 한다 — 마스킹은 원본에 적용된다."""
        norm = _norm("성명 홍길동")
        pos = norm.text.find(_norm("홍길동").text)
        start, end = norm.span(pos, pos + 3)
        assert "성명 홍길동"[start:end] == "홍길동"


# --------------------------------------------------------------------------
# 크롭 사각형
# --------------------------------------------------------------------------


class TestCropRect:
    def test_pads_beyond_the_vlm_box(self) -> None:
        rect = crop_rect(finding("x", bbox=(0.4, 0.4, 0.6, 0.5)), 1000, 1000, 0.35, 12)
        assert rect[0] < 400 and rect[1] < 400
        assert rect[2] > 600 and rect[3] > 500

    def test_minimum_pad_applies_to_tiny_boxes(self) -> None:
        """비율만 쓰면 작은 bbox 에서 패딩이 0 이 되어 값이 잘린다."""
        rect = crop_rect(finding("x", bbox=(0.5, 0.5, 0.502, 0.502)), 1000, 1000, 0.35, 12)
        assert rect[2] - rect[0] >= 24

    def test_clamped_to_page(self) -> None:
        rect = crop_rect(finding("x", bbox=(0.0, 0.0, 0.05, 0.05)), 1000, 1000, 0.5, 40)
        assert rect[0] == 0 and rect[1] == 0

    def test_never_degenerate(self) -> None:
        rect = crop_rect(finding("x", bbox=(1.0, 1.0, 1.0, 1.0)), 1000, 1000, 0.35, 12)
        assert rect[2] > rect[0] and rect[3] > rect[1]


# --------------------------------------------------------------------------
# 크롭 안에서 값 찾기
# --------------------------------------------------------------------------


class TestFindValue:
    cfg = LocateConfig()

    def test_exact_single_box(self) -> None:
        m = find_value("홍길동", [box("홍길동")], "NAME", self.cfg)
        assert m is not None
        assert m.agreement is Agreement.EXACT
        assert len(m.boxes) == 1

    def test_prefers_the_tightest_box(self) -> None:
        """옆 칸까지 딸려 들어오면 마스킹 박스가 셀 두 개를 덮는다."""
        boxes = [box("성명", 0, 0, 60, 30), box("홍길동", 70, 0, 150, 30)]
        m = find_value("홍길동", boxes, "NAME", self.cfg)
        assert m is not None
        assert [b.text for b in m.boxes] == ["홍길동"]

    def test_substring_gives_a_char_span(self) -> None:
        """부분 마스킹(901231-1******)을 위해 박스 내 오프셋이 필요하다."""
        m = find_value("홍길동", [box("담당자: 홍길동")], "NAME", self.cfg)
        assert m is not None
        assert m.char_span is not None
        start, end = m.char_span
        assert "담당자: 홍길동"[start:end] == "홍길동"

    def test_joins_boxes_when_the_value_is_split(self) -> None:
        boxes = [box("010-1234", 0, 0, 80, 30), box("5678", 85, 0, 130, 30)]
        m = find_value("010-1234-5678", boxes, "PHONE", self.cfg)
        assert m is not None
        assert len(m.boxes) == 2
        assert m.char_span is None  # 여러 박스에 걸치면 원문 오프셋이 하나가 아니다

    def test_folds_evasion_notation(self) -> None:
        m = find_value("010-1234-5678", [box("공1공-1234-5678")], "PHONE", self.cfg)
        assert m is not None
        assert m.agreement is Agreement.EXACT

    def test_similarity_rescues_hangul_misreads(self) -> None:
        m = find_value("홍길동", [box("홍길둥")], "NAME", self.cfg)
        assert m is not None
        assert m.agreement is Agreement.SIMILAR
        assert m.similarity < 1.0

    def test_similarity_is_never_used_for_numbers(self) -> None:
        """숫자는 한 글자 다르면 오독이 아니라 그냥 다른 번호다.

        유사도로 이어붙이면 옆 칸의 **다른 사람** 주민번호에 붙는다.
        """
        assert find_value("901231-1234563", [box("901231-1234564")], "RRN", self.cfg) is None
        assert find_value("010-1234-5678", [box("010-1234-5679")], "PHONE", self.cfg) is None

    def test_no_match_returns_none(self) -> None:
        assert find_value("홍길동", [box("김철수")], "NAME", self.cfg) is None

    def test_empty_seed_returns_none(self) -> None:
        assert find_value("", [box("홍길동")], "SIGNATURE", self.cfg) is None

    def test_failed_boxes_are_ignored(self) -> None:
        """rec 실패 박스의 텍스트는 신뢰할 수 없다."""
        failed = [box("홍길동", status=OcrStatus.FAILED)]
        assert find_value("홍길동", failed, "NAME", self.cfg) is None

    def test_no_boxes_returns_none(self) -> None:
        assert find_value("홍길동", [], "NAME", self.cfg) is None

    def test_similarity_can_be_disabled(self) -> None:
        strict = LocateConfig(similarity=1.0)
        assert find_value("홍길동", [box("홍길둥")], "NAME", strict) is None


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


class TestLocate:
    def test_empty_findings_short_circuits(self) -> None:
        ocr = FakeOcr([])
        regions, boxes = locate([], blank_page(), ocr)
        assert regions == [] and boxes == []
        assert ocr.batches == []

    def test_one_region_per_finding(self) -> None:
        """개인정보를 조용히 버리지 않는다. 좌표를 못 잡아도 남는다."""
        findings = [finding("홍길동"), finding("없는값", bbox=(0.5, 0.5, 0.7, 0.55))]
        ocr = FakeOcr([[box("홍길동")], [box("전혀다름")], [box("전혀다름")]])
        regions, _ = locate(findings, blank_page(), ocr)
        assert len(regions) == len(findings)

    def test_matched_region_uses_ocr_coordinates(self) -> None:
        """VLM 좌표가 아니라 OCR 좌표가 최종 좌표다."""
        f = finding("홍길동", bbox=(0.1, 0.1, 0.3, 0.15))
        # 크롭 기준 (10,20)-(90,50) 에서 찾았다고 하자
        ocr = FakeOcr([[box("홍길동", 10, 20, 90, 50)]])
        regions, _ = locate([f], blank_page(), ocr, LocateConfig(upscale=1.0))
        r = regions[0]
        assert r.source is Source.OCR_REFINED
        assert r.coarse is False
        # 크롭 오프셋이 더해져 페이지 좌표가 되었는가
        rect = crop_rect(f, PAGE_W, PAGE_H, 0.35, 12)
        assert r.bbox[0] == pytest.approx(rect[0] + 10, abs=DEFAULT_PAD_TOLERANCE)

    def test_upscaled_coordinates_are_divided_back(self) -> None:
        """업샘플한 크롭에서 나온 좌표를 그대로 쓰면 박스가 2배로 커진다."""
        f = finding("홍길동", bbox=(0.1, 0.1, 0.3, 0.15))
        one = FakeOcr([[box("홍길동", 20, 40, 180, 100)]])
        two = FakeOcr([[box("홍길동", 40, 80, 360, 200)]])  # 2배 크롭에서의 같은 위치
        r1 = locate([f], blank_page(), one, LocateConfig(upscale=1.0))[0][0]
        r2 = locate([f], blank_page(), two, LocateConfig(upscale=2.0))[0][0]
        assert r1.bbox == r2.bbox

    def test_crop_is_actually_enlarged(self) -> None:
        f = finding("홍길동", bbox=(0.1, 0.1, 0.3, 0.15))
        plain = FakeOcr([[]])
        locate([f], blank_page(), plain, LocateConfig(upscale=1.0, retry_pad_ratio=1.2))
        big = FakeOcr([[]])
        locate([f], blank_page(), big, LocateConfig(upscale=2.0, retry_pad_ratio=1.2))
        assert big.crop_shapes[0][0] > plain.crop_shapes[0][0]

    def test_exact_match_lifts_confidence(self) -> None:
        """두 엔진이 같은 값을 읽었으면 VLM 자기확신도보다 높게 볼 근거가 있다."""
        f = finding("홍길동", conf=0.6)
        regions, _ = locate([f], blank_page(), FakeOcr([[box("홍길동")]]))
        assert regions[0].confidence >= 0.9

    def test_exact_match_never_reaches_one(self) -> None:
        """conf 1.0 은 체크섬 통과에만 준다 (verify.py)."""
        regions, _ = locate([finding("홍길동")], blank_page(), FakeOcr([[box("홍길동")]]))
        assert regions[0].confidence < 1.0

    def test_similar_match_needs_review(self) -> None:
        regions, _ = locate([finding("홍길동")], blank_page(), FakeOcr([[box("홍길둥")]]))
        r = regions[0]
        assert r.agreement is Agreement.SIMILAR
        assert r.needs_review is True
        assert "OCR" in (r.reason or "")

    def test_retries_with_a_wider_crop(self) -> None:
        """VLM bbox 가 어긋나 값이 크롭 밖으로 나간 경우를 한 번 회수한다."""
        ocr = FakeOcr([[], [box("홍길동")]])
        regions, _ = locate([finding("홍길동")], blank_page(), ocr)
        assert ocr.batches == [1, 1]
        assert regions[0].source is Source.OCR_REFINED

    def test_retry_crop_is_wider_than_the_first(self) -> None:
        ocr = FakeOcr([[], []])
        locate([finding("홍길동")], blank_page(), ocr, LocateConfig(upscale=1.0))
        assert ocr.crop_shapes[1][0] > ocr.crop_shapes[0][0]

    def test_no_retry_when_first_pass_matched(self) -> None:
        ocr = FakeOcr([[box("홍길동")]])
        locate([finding("홍길동")], blank_page(), ocr)
        assert ocr.batches == [1]

    def test_signature_is_not_retried(self) -> None:
        """읽을 글자가 없는 항목을 두 번 OCR 하는 것은 낭비다."""
        ocr = FakeOcr([[]])
        locate([finding("", label="SIGNATURE")], blank_page(), ocr)
        assert ocr.batches == [1]

    def test_signature_coarse_is_not_flagged_for_review(self) -> None:
        """검토 큐가 서명으로 가득 차면 정작 위험한 불일치가 묻힌다."""
        regions, _ = locate([finding("", label="SIGNATURE")], blank_page(), FakeOcr([[]]))
        r = regions[0]
        assert r.source is Source.VLM_COARSE
        assert r.needs_review is False
        assert "서명" in (r.reason or "")

    def test_unreadable_crop_reports_handwriting(self) -> None:
        regions, _ = locate([finding("홍길동")], blank_page(), FakeOcr([[], []]))
        r = regions[0]
        assert r.source is Source.VLM_COARSE
        assert r.needs_review is True
        assert "아무 글자도 읽지 못했다" in (r.reason or "")

    def test_disagreement_records_both_readings(self) -> None:
        """어느 엔진을 손봐야 하는지 로그만 보고 알 수 있어야 한다."""
        ocr = FakeOcr([[box("9O1112-284626")], [box("9O1112-284626")]])
        regions, _ = locate(
            [finding("901112-2846261", label="RRN", bbox=(0.2, 0.2, 0.5, 0.25))],
            blank_page(), ocr,
        )
        reason = regions[0].reason or ""
        assert "901112-2846261" in reason
        assert "9O1112-284626" in reason

    def test_coarse_region_keeps_the_vlm_text(self) -> None:
        regions, _ = locate([finding("홍길동")], blank_page(), FakeOcr([[], []]))
        assert regions[0].vlm_text == "홍길동"
        assert regions[0].text is None

    def test_page_wide_box_indices_are_sequential(self) -> None:
        findings = [finding("홍길동"), finding("김철수", bbox=(0.5, 0.5, 0.7, 0.55))]
        ocr = FakeOcr([[box("홍길동"), box("성명")], [box("김철수")]])
        regions, boxes = locate(findings, blank_page(), ocr)
        assert [b.index for b in boxes] == list(range(len(boxes)))
        for r in regions:
            for i in r.member_index:
                assert boxes[i].index == i

    def test_boxes_record_their_crop(self) -> None:
        findings = [finding("홍길동"), finding("김철수", bbox=(0.5, 0.5, 0.7, 0.55))]
        ocr = FakeOcr([[box("홍길동")], [box("김철수")]])
        _, boxes = locate(findings, blank_page(), ocr)
        assert {b.crop_id for b in boxes} == {0, 1}

    def test_warns_about_the_coarse_count(self) -> None:
        warnings: list[str] = []
        locate([finding("홍길동")], blank_page(), FakeOcr([[], []]), warnings=warnings)
        assert any("좌표 확정 실패" in w for w in warnings)

    def test_low_conf_ocr_flags_review(self) -> None:
        ocr = FakeOcr([[box("홍길동", status=OcrStatus.LOW_CONF)]])
        regions, _ = locate([finding("홍길동")], blank_page(), ocr)
        assert regions[0].needs_review is True
        assert regions[0].ocr_status is OcrStatus.LOW_CONF

    def test_field_hint_is_carried_through(self) -> None:
        regions, _ = locate([finding("홍길동")], blank_page(), FakeOcr([[box("홍길동")]]))
        assert regions[0].field == "성명"


#: ``crop_rect`` 의 정수 절단과 최종 패딩 때문에 1~3px 오차가 난다.
DEFAULT_PAD_TOLERANCE = 4
