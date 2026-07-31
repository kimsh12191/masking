"""병합 및 검증 테스트.

LLM 출력을 신뢰하지 않는다는 설계가 실제로 지켜지는지 확인한다.
"""

from __future__ import annotations

from pii_pipeline.merge import (
    confirmed_map,
    finalize,
    regions_from_pass1,
    regions_from_pass2,
    regions_from_rules,
)
from pii_pipeline.ocr.layout import assign_reading_order
from pii_pipeline.rules.detectors import RuleHit
from pii_pipeline.schema import OcrBox, OcrStatus, Source

PAGE_W, PAGE_H = 2000, 3000


def make_boxes(rows: list[list[str]]) -> list[OcrBox]:
    boxes: list[OcrBox] = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x1 = 100 + c * 400
            y1 = 100 + r * 100
            boxes.append(OcrBox(index=-1, bbox=(x1, y1, x1 + 380, y1 + 60), text=text))
    return assign_reading_order(boxes)


def sample_boxes() -> list[OcrBox]:
    return make_boxes(
        [
            ["성명", "홍길동"],
            ["주민등록번호", "901231-1234563"],
            ["주소", "서울특별시 강남구"],
            ["테헤란로 123", "○○빌딩 5층"],
            ["직장명", "하나은행 리스크관리부"],
        ]
    )


class TestRegionsFromRules:
    def test_checksum_pass_is_confident_and_no_review(self) -> None:
        boxes = sample_boxes()
        hit = RuleHit(box_index=3, label="RRN", text="901231-1234563",
                      span=(0, 14), checksum_ok=True)
        regions = regions_from_rules([hit], boxes, PAGE_W, PAGE_H)
        assert len(regions) == 1
        r = regions[0]
        assert r.source is Source.RULE
        assert r.confidence == 1.0
        assert r.needs_review is False
        assert r.char_span == (0, 14)

    def test_checksum_fail_flags_review(self) -> None:
        boxes = sample_boxes()
        hit = RuleHit(box_index=3, label="RRN", text="901231-1234564",
                      span=(0, 14), checksum_ok=False, needs_review=True)
        r = regions_from_rules([hit], boxes, PAGE_W, PAGE_H)[0]
        assert r.needs_review is True
        assert r.low_confidence is True

    def test_out_of_range_index_dropped(self) -> None:
        boxes = sample_boxes()
        hit = RuleHit(box_index=999, label="RRN", text="x", span=(0, 1), checksum_ok=True)
        assert regions_from_rules([hit], boxes, PAGE_W, PAGE_H) == []


class TestConfirmedMap:
    def test_single(self) -> None:
        hits = [RuleHit(3, "RRN", "x", (0, 1), True)]
        assert confirmed_map(hits) == {3: "RRN"}

    def test_multiple_labels_in_one_box_are_joined(self) -> None:
        hits = [RuleHit(3, "PHONE", "x", (0, 1), True),
                RuleHit(3, "EMAIL", "y", (5, 6), True)]
        assert confirmed_map(hits) == {3: "PHONE+EMAIL"}

    def test_duplicate_label_not_repeated(self) -> None:
        hits = [RuleHit(3, "PHONE", "x", (0, 1), True),
                RuleHit(3, "PHONE", "y", (5, 6), True)]
        assert confirmed_map(hits) == {3: "PHONE"}


class TestRegionsFromPass1:
    def test_single_box_name(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.97}]}
        regions = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, warnings)
        assert len(regions) == 1
        assert regions[0].type == "NAME"
        assert regions[0].text == "홍길동"
        assert regions[0].source is Source.LLM_PASS1

    def test_multi_box_address_is_unioned(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"regions": [{"idx": [5, 6, 7], "type": "ADDRESS", "conf": 0.94}]}
        regions = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, warnings)
        assert len(regions) == 1
        r = regions[0]
        assert r.member_index == [5, 6, 7]
        assert len(r.member_boxes) == 3
        # 합집합 박스가 구성 박스 전체를 덮어야 한다
        for mb in r.member_boxes:
            assert r.bbox[0] <= mb[0] and r.bbox[1] <= mb[1]
            assert r.bbox[2] >= mb[2] and r.bbox[3] >= mb[3]

    def test_text_reassembled_from_ocr_not_llm(self) -> None:
        """LLM 이 엉뚱한 텍스트를 보내도 OCR 원문으로 재조립되어야 한다."""
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {
            "regions": [
                {"idx": [5, 6, 7], "type": "ADDRESS", "conf": 0.9, "text": "환각된주소"}
            ]
        }
        r = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, warnings)[0]
        assert "환각" not in (r.text or "")
        assert r.text == "서울특별시 강남구 테헤란로 123 ○○빌딩 5층"

    def test_out_of_range_index_warns_and_drops(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"regions": [{"idx": [1, 999], "type": "NAME", "conf": 0.9}]}
        regions = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, warnings)
        assert regions[0].member_index == [1]
        assert any("999" in w for w in warnings)

    def test_already_claimed_box_is_skipped(self) -> None:
        """규칙 레이어가 확정한 박스는 LLM 이 다시 주장해도 무시한다."""
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"regions": [{"idx": [3], "type": "OTHER", "conf": 0.9}]}
        regions = regions_from_pass1(payload, boxes, {3}, PAGE_W, PAGE_H, warnings)
        assert regions == []

    def test_spatially_distant_group_is_split(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        # 1(성명값) 과 9(직장명값) 은 행 간격이 크다
        payload = {"regions": [{"idx": [1, 9], "type": "NAME", "conf": 0.8}]}
        regions = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, warnings)
        assert len(regions) == 2

    def test_low_conf_kept_with_flag(self) -> None:
        """recall 우선 — 낮은 확신도도 폐기하지 않는다."""
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.2}]}
        r = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, warnings)[0]
        assert r.low_confidence is True
        assert r.needs_review is True

    def test_claimed_set_is_updated(self) -> None:
        boxes = sample_boxes()
        claimed: set[int] = set()
        payload = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.9}]}
        regions_from_pass1(payload, boxes, claimed, PAGE_W, PAGE_H, [])
        assert 1 in claimed

    def test_empty_payload(self) -> None:
        boxes = sample_boxes()
        assert regions_from_pass1({}, boxes, set(), PAGE_W, PAGE_H, []) == []
        assert regions_from_pass1({"regions": []}, boxes, set(), PAGE_W, PAGE_H, []) == []


class TestRegionsFromPass2:
    def test_recovers_ocr_failed_box_with_precise_coords(self) -> None:
        boxes = sample_boxes()
        boxes[1].status = OcrStatus.FAILED
        boxes[1].text = ""
        warnings: list[str] = []
        payload = {
            "missed": [
                {"idx": [1], "type": "NAME", "conf": 0.85, "reason": "손글씨 성명"}
            ]
        }
        r = regions_from_pass2(payload, boxes, set(), PAGE_W, PAGE_H, warnings)[0]
        assert r.source is Source.VLM_PASS2
        assert r.coarse is False           # OCR 좌표를 썼으므로 정확
        assert r.member_boxes == [boxes[1].bbox]
        assert r.needs_review is True      # OCR 실패 박스이므로 검토 대상
        assert r.reason == "손글씨 성명"

    def test_bbox_norm_fallback_is_marked_coarse(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {
            "missed": [
                {
                    "bbox_norm": [0.6, 0.7, 0.8, 0.75],
                    "type": "SIGNATURE",
                    "conf": 0.8,
                    "reason": "자필 서명",
                }
            ]
        }
        r = regions_from_pass2(payload, boxes, set(), PAGE_W, PAGE_H, warnings)[0]
        assert r.source is Source.VLM_GROUNDING
        assert r.coarse is True
        assert r.needs_review is True
        assert r.member_index == []

    def test_idx_preferred_over_bbox_norm(self) -> None:
        """idx 가 있으면 OCR 좌표를 쓰고 bbox_norm 은 무시해야 한다."""
        boxes = sample_boxes()
        payload = {
            "missed": [
                {
                    "idx": [1],
                    "bbox_norm": [0.0, 0.0, 0.1, 0.1],
                    "type": "NAME",
                    "conf": 0.9,
                    "reason": "x",
                }
            ]
        }
        r = regions_from_pass2(payload, boxes, set(), PAGE_W, PAGE_H, [])[0]
        assert r.source is Source.VLM_PASS2
        assert r.coarse is False

    def test_already_detected_idx_silently_ignored(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"missed": [{"idx": [1], "type": "NAME", "conf": 0.9, "reason": "x"}]}
        regions = regions_from_pass2(payload, boxes, {1}, PAGE_W, PAGE_H, warnings)
        assert regions == []
        assert warnings == []

    def test_item_without_coords_is_warned(self) -> None:
        boxes = sample_boxes()
        warnings: list[str] = []
        payload = {"missed": [{"type": "NAME", "conf": 0.9, "reason": "좌표 없음"}]}
        assert regions_from_pass2(payload, boxes, set(), PAGE_W, PAGE_H, warnings) == []
        assert any("bbox_norm" in w for w in warnings)

    def test_empty_missed(self) -> None:
        boxes = sample_boxes()
        assert regions_from_pass2({"missed": []}, boxes, set(), PAGE_W, PAGE_H, []) == []


class TestFinalize:
    def test_assigns_deterministic_ids_in_reading_order(self) -> None:
        boxes = sample_boxes()
        payload = {
            "regions": [
                {"idx": [9], "type": "ORG", "conf": 0.9},
                {"idx": [1], "type": "NAME", "conf": 0.9},
            ]
        }
        regions = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, [])
        out = finalize(regions, boxes, [])
        assert [r.id for r in out] == ["r001", "r002"]
        # 위쪽(성명)이 먼저
        assert out[0].type == "NAME"
        assert out[1].type == "ORG"

    def test_is_deterministic_across_runs(self) -> None:
        boxes = sample_boxes()
        payload = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.9},
                               {"idx": [9], "type": "ORG", "conf": 0.9}]}
        first = finalize(
            regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, []), boxes, []
        )
        second = finalize(
            regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, []), boxes, []
        )
        assert [(r.id, r.type, r.bbox) for r in first] == [
            (r.id, r.type, r.bbox) for r in second
        ]

    def test_duplicate_same_label_keeps_higher_conf(self) -> None:
        boxes = sample_boxes()
        payload = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.6}]}
        low = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, [])
        payload2 = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.95}]}
        high = regions_from_pass1(payload2, boxes, set(), PAGE_W, PAGE_H, [])
        out = finalize(low + high, boxes, [])
        assert len(out) == 1
        assert out[0].confidence == 0.95

    def test_coarse_regions_are_never_deduped_away(self) -> None:
        boxes = sample_boxes()
        payload = {
            "missed": [
                {"bbox_norm": [0.1, 0.1, 0.2, 0.2], "type": "SIGNATURE",
                 "conf": 0.7, "reason": "a"},
                {"bbox_norm": [0.5, 0.5, 0.6, 0.6], "type": "SIGNATURE",
                 "conf": 0.7, "reason": "b"},
            ]
        }
        regions = regions_from_pass2(payload, boxes, set(), PAGE_W, PAGE_H, [])
        assert len(finalize(regions, boxes, [])) == 2

    def test_exclusions_hook_is_noop_by_default(self) -> None:
        """학습데이터 구축 단계에서는 아무것도 제외하지 않는다."""
        boxes = sample_boxes()
        payload = {"regions": [{"idx": [1], "type": "NAME", "conf": 0.9}]}
        regions = regions_from_pass1(payload, boxes, set(), PAGE_W, PAGE_H, [])
        warnings: list[str] = []
        assert len(finalize(regions, boxes, warnings)) == 1
        assert warnings == []
