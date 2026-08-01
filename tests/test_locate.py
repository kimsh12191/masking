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
    crop_scale,
    find_value,
    locate,
    select_by_geometry,
)
from pii_pipeline.ocr.layout import denorm_bbox
from pii_pipeline.schema import Agreement, OcrBox, OcrStatus, Source, VlmFinding

np = pytest.importorskip("numpy", reason="numpy 미설치 환경에서는 건너뛴다")

PAGE_W, PAGE_H = 1000, 2000


def blank_page(w: int = PAGE_W, h: int = PAGE_H) -> Any:
    return np.zeros((h, w, 3), dtype=np.uint8)


def box(text: str, x1=0, y1=0, x2=100, y2=30, status=OcrStatus.OK, conf=0.95) -> OcrBox:
    return OcrBox(index=-1, bbox=(x1, y1, x2, y2), text=text, status=status, rec_conf=conf)


def finding(text: str, label: str = "NAME", bbox=(0.1, 0.1, 0.3, 0.15), conf=0.9) -> VlmFinding:
    return VlmFinding(text=text, type=label, field="성명", bbox_norm=bbox, conf=conf)


def in_vlm_box(f: VlmFinding, text: str, status=OcrStatus.OK) -> OcrBox:
    """``f`` 의 VLM bbox **중앙**에 놓이도록 크롭 좌표계의 OCR 박스를 만든다.

    ``FakeOcr`` 는 크롭 기준 좌표를 돌려주고 ``locate`` 가 이를 페이지 좌표로
    환산한다(오프셋 더하기 + 업샘플 배율 나누기). 그래서 여기서는 그 역변환을
    해 둬야 한다 — 배율을 빼먹으면 박스가 크롭 원점 쪽으로 당겨져서, 테스트가
    통과하더라도 의도한 위치를 검증하지 못한다.
    """
    cfg = LocateConfig()
    rect = crop_rect(f, PAGE_W, PAGE_H, cfg.pad_ratio, cfg.min_pad_px)
    scale = crop_scale(rect, cfg)
    vx1, vy1, vx2, vy2 = denorm_bbox(f.bbox_norm, PAGE_W, PAGE_H)

    cx, cy = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0
    half_w = max(20.0, (vx2 - vx1) / 4.0)
    half_h = max(8.0, (vy2 - vy1) / 4.0)

    def to_crop(px: float, py: float) -> tuple[int, int]:
        return (int((px - rect[0]) * scale), int((py - rect[1]) * scale))

    x1, y1 = to_crop(cx - half_w, cy - half_h)
    x2, y2 = to_crop(cx + half_w, cy + half_h)
    return OcrBox(
        index=-1,
        bbox=(x1, y1, x2, y2),
        text=text,
        status=status,
        rec_conf=0.2 if status is OcrStatus.FAILED else 0.95,
    )


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
        m = find_value("홍길동", [box("홍길동")], self.cfg)
        assert m is not None
        assert m.agreement is Agreement.EXACT
        assert len(m.boxes) == 1

    def test_prefers_the_tightest_box(self) -> None:
        """옆 칸까지 딸려 들어오면 마스킹 박스가 셀 두 개를 덮는다."""
        boxes = [box("성명", 0, 0, 60, 30), box("홍길동", 70, 0, 150, 30)]
        m = find_value("홍길동", boxes, self.cfg)
        assert m is not None
        assert [b.text for b in m.boxes] == ["홍길동"]

    def test_substring_gives_a_char_span(self) -> None:
        """부분 마스킹(901231-1******)을 위해 박스 내 오프셋이 필요하다."""
        m = find_value("홍길동", [box("담당자: 홍길동")], self.cfg)
        assert m is not None
        assert m.char_span is not None
        start, end = m.char_span
        assert "담당자: 홍길동"[start:end] == "홍길동"

    def test_joins_boxes_when_the_value_is_split(self) -> None:
        boxes = [box("010-1234", 0, 0, 80, 30), box("5678", 85, 0, 130, 30)]
        m = find_value("010-1234-5678", boxes, self.cfg)
        assert m is not None
        assert len(m.boxes) == 2
        assert m.char_span is None  # 여러 박스에 걸치면 원문 오프셋이 하나가 아니다

    def test_folds_evasion_notation(self) -> None:
        m = find_value("010-1234-5678", [box("공1공-1234-5678")], self.cfg)
        assert m is not None
        assert m.agreement is Agreement.EXACT

    def test_near_misses_are_not_matched(self) -> None:
        """유사 매칭 단계는 없다. 한 글자만 달라도 여기서는 실패다.

        회수는 ``select_by_geometry`` 가 한다 — 텍스트가 비슷하면서 위치도
        겹치면 기하가 같은 박스를 고르고, 위치가 안 겹치면 애초에 다른 값이라
        유사도로 이어붙여선 안 된다. 특히 숫자는 한 글자 다르면 "오독된 같은
        번호" 가 아니라 **그냥 다른 사람의 번호**다.
        """
        assert find_value("홍길동", [box("홍길둥")], self.cfg) is None
        assert find_value("901231-1234563", [box("901231-1234564")], self.cfg) is None
        assert find_value("010-1234-5678", [box("010-1234-5679")], self.cfg) is None

    def test_no_match_returns_none(self) -> None:
        assert find_value("홍길동", [box("김철수")], self.cfg) is None

    def test_empty_seed_returns_none(self) -> None:
        assert find_value("", [box("홍길동")], self.cfg) is None

    def test_failed_boxes_are_ignored(self) -> None:
        """rec 실패 박스의 텍스트는 신뢰할 수 없다."""
        failed = [box("홍길동", status=OcrStatus.FAILED)]
        assert find_value("홍길동", failed, self.cfg) is None

    def test_no_boxes_returns_none(self) -> None:
        assert find_value("홍길동", [], self.cfg) is None



# --------------------------------------------------------------------------
# 기하 선택
# --------------------------------------------------------------------------


class TestSelectByGeometry:
    """텍스트를 전혀 보지 않는 선택. 페이지 좌표로 직접 검증한다."""

    cfg = LocateConfig()
    VLM = (200, 400, 500, 460)   # VLM 이 지목한 사각형

    def test_picks_the_box_whose_center_is_inside(self) -> None:
        target = box("홍길동", 210, 405, 320, 450)
        outside = box("옆칸", 600, 405, 700, 450)
        m = select_by_geometry(self.VLM, [outside, target], self.cfg)
        assert m is not None
        assert [b.text for b in m.boxes] == ["홍길동"]

    def test_ignores_text_entirely(self) -> None:
        """VLM 텍스트와 전혀 다른 값이어도 위치가 맞으면 고른다.

        이게 목적이다 — 숫자 오독이 좌표 확정을 막지 않게 한다.
        """
        m = select_by_geometry(self.VLM, [box("전혀다른값", 210, 405, 320, 450)], self.cfg)
        assert m is not None
        assert m.how == "geometry"
        assert m.agreement is Agreement.NONE

    def test_includes_failed_boxes(self) -> None:
        """rec 실패 박스도 좌표는 유효하다. 손글씨 회수 경로."""
        failed = box("", 210, 405, 320, 450, status=OcrStatus.FAILED)
        m = select_by_geometry(self.VLM, [failed], self.cfg)
        assert m is not None
        assert m.boxes[0].status is OcrStatus.FAILED

    def test_takes_all_qualifying_boxes_for_split_values(self) -> None:
        parts = [box("010-1234", 210, 405, 320, 450), box("5678", 330, 405, 420, 450)]
        m = select_by_geometry(self.VLM, parts, self.cfg)
        assert m is not None
        assert len(m.boxes) == 2

    def test_result_is_in_reading_order(self) -> None:
        later = box("b", 330, 405, 420, 450)
        earlier = box("a", 210, 405, 320, 450)
        m = select_by_geometry(self.VLM, [later, earlier], self.cfg)
        assert m is not None
        assert [b.text for b in m.boxes] == ["a", "b"]

    def test_falls_back_to_area_coverage(self) -> None:
        """긴 주소 줄이 VLM bbox 를 관통해 중심이 밖으로 나간 경우."""
        long_line = box("서울특별시 강남구 테헤란로 123", 250, 405, 900, 450)
        m = select_by_geometry(self.VLM, [long_line], self.cfg)
        # 중심(575)은 VLM bbox 밖이고 면적 커버도 0.5 미만이므로 최대 겹침으로 잡힌다
        assert m is not None
        assert m.boxes[0].text.startswith("서울")

    def test_returns_none_when_nothing_overlaps(self) -> None:
        far = box("먼칸", 700, 900, 800, 950)
        assert select_by_geometry(self.VLM, [far], self.cfg) is None

    def test_returns_none_for_empty_boxes(self) -> None:
        assert select_by_geometry(self.VLM, [], self.cfg) is None

    def test_over_selection_is_capped(self) -> None:
        """넉넉한 bbox 가 옆 칸까지 덮었을 때 무한정 딸려오지 않아야 한다."""
        wide = (0, 0, 1000, 1000)
        many = [box(f"b{i}", 10 + i * 30, 10, 35 + i * 30, 40) for i in range(12)]
        m = select_by_geometry(wide, many, LocateConfig(max_join=4))
        assert m is not None
        assert len(m.boxes) == 4

    def test_cap_keeps_boxes_nearest_the_vlm_centre(self) -> None:
        vlm = (0, 0, 1000, 100)
        near = box("near", 480, 20, 520, 60)      # 중심 500 — vlm 중심과 일치
        far = box("far", 10, 20, 50, 60)
        m = select_by_geometry(vlm, [far, near], LocateConfig(max_join=1))
        assert m is not None
        assert [b.text for b in m.boxes] == ["near"]


class TestInVlmBoxHelper:
    """헬퍼가 정말 VLM bbox 안에 박스를 놓는지 검산한다.

    업샘플 배율을 빼먹으면 박스가 크롭 원점 쪽으로 당겨지는데, 그래도 테스트가
    통과해버릴 수 있다. 그러면 기하 선택을 검증하는 게 아니라 우연을 검증한다.
    """

    def test_lands_inside_the_vlm_bbox(self) -> None:
        f = finding("901112-2846261", label="RRN", bbox=(0.2, 0.2, 0.5, 0.25))
        crop_box = in_vlm_box(f, "901112-2846261")
        cfg = LocateConfig()
        rect = crop_rect(f, PAGE_W, PAGE_H, cfg.pad_ratio, cfg.min_pad_px)
        scale = crop_scale(rect, cfg)
        # locate 가 하는 것과 같은 환산
        page = (
            rect[0] + int(crop_box.bbox[0] / scale),
            rect[1] + int(crop_box.bbox[1] / scale),
            rect[0] + int(round(crop_box.bbox[2] / scale)),
            rect[1] + int(round(crop_box.bbox[3] / scale)),
        )
        vlm = denorm_bbox(f.bbox_norm, PAGE_W, PAGE_H)
        cx, cy = (page[0] + page[2]) / 2, (page[1] + page[3]) / 2
        assert vlm[0] <= cx <= vlm[2]
        assert vlm[1] <= cy <= vlm[3]


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

    def test_hangul_misread_falls_through_to_geometry(self) -> None:
        """한 글자 오독은 이제 유사 매칭이 아니라 기하가 회수한다.

        좌표는 여전히 OCR 것이고(``ocr_refined``), 값 교차검증은 실패했으므로
        ``agreement=NONE`` + ``needs_review`` 로 남는다. ``reason`` 에 두 엔진이
        각각 뭘 읽었는지가 함께 적혀야 사람이 판단할 수 있다.
        """
        f = finding("홍길동")
        regions, _ = locate([f], blank_page(), FakeOcr([[in_vlm_box(f, "홍길둥")]] * 2))
        r = regions[0]
        assert r.source is Source.OCR_REFINED
        assert r.agreement is Agreement.NONE
        assert r.needs_review is True
        assert "홍길동" in (r.reason or "") and "홍길둥" in (r.reason or "")

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

    def test_unreadable_crop_reports_detection_failure(self) -> None:
        """크롭에 박스가 하나도 없다 = OCR 검출 문제. 처방이 다르다."""
        regions, _ = locate([finding("홍길동")], blank_page(), FakeOcr([[], []]))
        r = regions[0]
        assert r.source is Source.VLM_COARSE
        assert r.needs_review is True
        assert "검출되지 않았다" in (r.reason or "")

    def test_digit_misread_still_gets_precise_coordinates(self) -> None:
        """이 설계의 핵심 케이스.

        VLM 이 숫자 한두 자리를 틀려도 좌표는 확정되어야 한다. 긴 숫자열 정확
        전사는 VLM 의 최대 약점이고, 그걸 좌표 확정의 전제로 두면 잘하는 일
        (위치 판단)의 결과가 버려진다.
        """
        f = finding("901112-2846261", label="RRN", bbox=(0.2, 0.2, 0.5, 0.25))
        # VLM bbox 안에 놓인, 두 자리 오독된 OCR 박스
        ocr = FakeOcr([[in_vlm_box(f, "901112-2864261")]] * 2)
        regions, _ = locate([f], blank_page(), ocr)
        r = regions[0]
        assert r.source is Source.OCR_REFINED   # 좌표는 확정됐다
        assert r.coarse is False
        assert r.agreement is Agreement.NONE    # 값 교차검증은 실패했다
        assert r.needs_review is True

    def test_disagreement_records_both_readings(self) -> None:
        """어느 엔진을 손봐야 하는지 로그만 보고 알 수 있어야 한다."""
        f = finding("901112-2846261", label="RRN", bbox=(0.2, 0.2, 0.5, 0.25))
        ocr = FakeOcr([[in_vlm_box(f, "9O1112-284626")]] * 2)
        regions, _ = locate([f], blank_page(), ocr)
        reason = regions[0].reason or ""
        assert "901112-2846261" in reason
        assert "9O1112-284626" in reason

    def test_no_overlap_falls_back_to_the_nearest_line_in_the_crop(self) -> None:
        """겹치는 박스가 없어도 크롭 안에 글자가 있으면 좌표를 포기하지 않는다.

        예전에는 여기서 ``vlm_coarse`` 로 떨어졌다. 그게 이 파이프라인의 실제
        고장이었다 — VLM 좌표가 밀리면 페이지의 **모든** 항목이 이 경로로 가서
        전부 VLM 원본 좌표(빨간 박스)로 나왔다. 크롭을 뜰 때 인정한 공차를
        고를 때도 인정하지 않았기 때문이다.
        """
        f = finding("홍길동", bbox=(0.6, 0.6, 0.8, 0.65))
        ocr = FakeOcr([[box("전혀다른칸", 0, 0, 40, 20)]] * 2)
        regions, _ = locate([f], blank_page(), ocr)
        r = regions[0]
        assert r.source is Source.OCR_REFINED
        assert r.coarse is False
        assert r.needs_review is True
        assert "가장 가까운 줄" in (r.reason or "")

    def test_nearby_selection_covers_the_vlm_box_too(self) -> None:
        """어느 쪽이 맞는지 모르므로 둘 다 덮는다 — 한쪽만 택하면 미탐이 된다."""
        f = finding("홍길동", bbox=(0.6, 0.6, 0.8, 0.65))
        ocr = FakeOcr([[box("전혀다른칸", 0, 0, 40, 20)]] * 2)
        page = blank_page()
        regions, _ = locate([f], page, ocr)
        r = regions[0]
        h, w = page.shape[:2]
        assert r.vlm_bbox is not None
        assert r.bbox[0] <= r.vlm_bbox[0] and r.bbox[1] <= r.vlm_bbox[1]
        assert r.bbox[2] >= r.vlm_bbox[2] and r.bbox[3] >= r.vlm_bbox[3]
        # 크롭 범위로 닫혀 있다 — 페이지 전체로 번지지 않는다
        assert (r.bbox[2] - r.bbox[0]) < w * 0.5
        assert (r.bbox[3] - r.bbox[1]) < h * 0.5

    def test_nearby_selection_is_warned(self) -> None:
        """이 건수가 크면 grounding 이 전반적으로 밀렸다는 신호다."""
        warnings: list[str] = []
        f = finding("홍길동", bbox=(0.6, 0.6, 0.8, 0.65))
        ocr = FakeOcr([[box("전혀다른칸", 0, 0, 40, 20)]] * 2)
        locate([f], blank_page(), ocr, warnings=warnings)
        assert any("근접 선택" in w for w in warnings)

    def test_empty_crop_is_still_the_coarse_path(self) -> None:
        """크롭에 박스가 하나도 없으면 여전히 VLM 좌표를 쓴다 (OCR 검출 문제)."""
        f = finding("홍길동", bbox=(0.6, 0.6, 0.8, 0.65))
        regions, _ = locate([f], blank_page(), FakeOcr([[], []]))
        r = regions[0]
        assert r.source is Source.VLM_COARSE
        assert "검출되지 않았다" in (r.reason or "")

    def test_geometry_fallback_can_be_disabled(self) -> None:
        f = finding("901112-2846261", label="RRN", bbox=(0.2, 0.2, 0.5, 0.25))
        ocr = FakeOcr([[in_vlm_box(f, "901112-2864261")]] * 2)
        regions, _ = locate(
            [f], blank_page(), ocr, LocateConfig(geometry_fallback=False)
        )
        assert regions[0].source is Source.VLM_COARSE

    def test_failed_box_gets_precise_coordinates_via_geometry(self) -> None:
        """손글씨 회수 — 텍스트 매칭으로는 불가능했던 경로.

        det 는 성공하고 rec 만 실패한 박스는 "여기 글자는 있다" 는 뜻이다.
        VLM 이 그 자리를 지목했다면 그 좌표가 VLM 근사치보다 정확하다.
        """
        f = finding("홍길동")
        handwritten = in_vlm_box(f, "", status=OcrStatus.FAILED)
        regions, _ = locate([f], blank_page(), FakeOcr([[handwritten]] * 2))
        r = regions[0]
        assert r.source is Source.OCR_REFINED
        assert r.coarse is False
        assert r.ocr_status is OcrStatus.FAILED
        assert r.text is None            # 읽지 못했으므로 텍스트는 없다
        assert r.vlm_text == "홍길동"     # VLM 이 읽은 값은 남는다
        assert r.needs_review is True

    def test_geometry_selection_is_warned(self) -> None:
        warnings: list[str] = []
        f = finding("901112-2846261", label="RRN", bbox=(0.2, 0.2, 0.5, 0.25))
        ocr = FakeOcr([[in_vlm_box(f, "901112-2864261")]] * 2)
        locate([f], blank_page(), ocr, warnings=warnings)
        assert any("기하 선택" in w for w in warnings)

    def test_text_match_wins_over_geometry(self) -> None:
        """텍스트가 일치하는 박스가 있으면 그게 더 좁고 확실하다."""
        f = finding("홍길동", bbox=(0.1, 0.1, 0.4, 0.15))
        vx1, vy1 = 100, 200   # VLM bbox 좌상단 (0.1, 0.1) x 페이지 크기
        wrong = OcrBox(index=-1, bbox=(vx1 + 5, vy1 + 5, vx1 + 60, vy1 + 35),
                       text="다른값", status=OcrStatus.OK, rec_conf=0.95)
        right = OcrBox(index=-1, bbox=(vx1 + 80, vy1 + 5, vx1 + 160, vy1 + 35),
                       text="홍길동", status=OcrStatus.OK, rec_conf=0.95)
        regions, _ = locate([f], blank_page(), FakeOcr([[wrong, right]]))
        r = regions[0]
        assert r.agreement is Agreement.EXACT
        assert r.text == "홍길동"

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
