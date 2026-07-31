"""프롬프트 렌더링 테스트."""

from __future__ import annotations

import json

from pii_pipeline.llm.prompts import (
    SYSTEM_PASS1,
    SYSTEM_PASS2,
    build_pass1_user,
    build_pass2_user,
    render_box_list,
)
from pii_pipeline.ocr.layout import assign_reading_order
from pii_pipeline.schema import (
    CONTEXT_LABELS,
    PASS1_SCHEMA,
    PASS2_SCHEMA,
    PII_LABELS,
    OcrBox,
    OcrStatus,
)

PAGE_W, PAGE_H = 2000, 3000


def boxes() -> list[OcrBox]:
    out: list[OcrBox] = []
    for r, row in enumerate([["성명", "홍길동"], ["주민등록번호", "901231-1234563"]]):
        for c, text in enumerate(row):
            x1, y1 = 100 + c * 400, 100 + r * 100
            out.append(OcrBox(index=-1, bbox=(x1, y1, x1 + 380, y1 + 60), text=text))
    return assign_reading_order(out)


class TestRenderBoxList:
    def test_includes_index_coords_and_text(self) -> None:
        text = render_box_list(boxes(), PAGE_W, PAGE_H)
        lines = text.splitlines()
        assert len(lines) == 4
        assert lines[0].startswith("[00] (0.05,0.03)")
        assert "성명" in lines[0]
        assert "홍길동" in lines[1]

    def test_confirmed_tag_is_appended_not_removed(self) -> None:
        """확정된 박스도 목록에 남아야 한다 — 서식 구조 이해에 필요하다."""
        text = render_box_list(boxes(), PAGE_W, PAGE_H, confirmed={3: "RRN"})
        assert "<CONFIRMED:RRN>" in text
        assert "901231-1234563" in text

    def test_failed_box_gets_placeholder_and_tag(self) -> None:
        bs = boxes()
        bs[1].status = OcrStatus.FAILED
        bs[1].text = ""
        text = render_box_list(bs, PAGE_W, PAGE_H)
        assert "<OCR_FAILED>" in text
        assert "???" in text

    def test_low_conf_tagged(self) -> None:
        bs = boxes()
        bs[1].status = OcrStatus.LOW_CONF
        assert "<LOW_CONF>" in render_box_list(bs, PAGE_W, PAGE_H)

    def test_empty_box_list(self) -> None:
        assert render_box_list([], PAGE_W, PAGE_H) == ""


class TestPass1User:
    def test_contains_box_list(self) -> None:
        user = build_pass1_user(boxes(), PAGE_W, PAGE_H)
        assert "[00]" in user and "[03]" in user


class TestPass2User:
    def test_non_blind_shows_detected_indices(self) -> None:
        user = build_pass2_user(
            boxes(), PAGE_W, PAGE_H, detected_idx=[1, 3], blind=False
        )
        assert "1, 3" in user
        assert "다시 보고하지 마라" in user

    def test_blind_hides_detected_indices(self) -> None:
        user = build_pass2_user(
            boxes(), PAGE_W, PAGE_H, detected_idx=[1, 3], blind=True
        )
        assert "이미 탐지된 박스 번호" not in user
        assert "독립적으로" in user

    def test_empty_detected_list_rendered(self) -> None:
        user = build_pass2_user(boxes(), PAGE_W, PAGE_H, detected_idx=[], blind=False)
        assert "(없음)" in user


class TestSystemPrompts:
    def test_pass1_lists_only_context_labels(self) -> None:
        """pass1 은 규칙 레이어가 처리하는 라벨을 판단하지 않는다."""
        assert "RRN" not in SYSTEM_PASS1.split("type 은 다음 중")[-1]
        for label in CONTEXT_LABELS:
            assert label in SYSTEM_PASS1

    def test_pass1_states_recall_priority(self) -> None:
        assert "누락이 오탐보다 위험" in SYSTEM_PASS1

    def test_pass1_forbids_coordinate_guessing(self) -> None:
        assert "좌표를 추측하지 마라" in SYSTEM_PASS1

    def test_pass1_covers_label_and_value_in_one_box(self) -> None:
        """실제 문서는 "담당자: 조민석" 처럼 라벨과 값이 한 박스에 섞여 있다.

        박스 단위로만 답할 수 있다는 제약을 알려주지 않으면 모델이 그 박스를
        건너뛴다 (라벨은 제외하라는 지시와 충돌하기 때문).
        """
        assert "박스 단위로만 답할 수 있다" in SYSTEM_PASS1
        assert "담당자: 조민석" in SYSTEM_PASS1
        assert "그 박스를 포함" in SYSTEM_PASS1

    def test_pass1_excludes_label_only_boxes(self) -> None:
        assert "라벨만 있고 값이 없는 박스" in SYSTEM_PASS1

    def test_pass1_handles_unlabeled_values(self) -> None:
        """라벨 없는 값(괄호 안 생년월일 등)도 형태로 판단해야 한다."""
        assert "라벨이 없어도" in SYSTEM_PASS1
        assert "1978-04-15" in SYSTEM_PASS1

    def test_pass2_also_states_box_unit_constraint(self) -> None:
        assert "박스 단위로만 답할 수 있다" in SYSTEM_PASS2
        assert "라벨이 없어도" in SYSTEM_PASS2

    def test_pass2_prefers_idx_over_bbox(self) -> None:
        assert "idx" in SYSTEM_PASS2 and "bbox_norm" in SYSTEM_PASS2
        assert "하나만" in SYSTEM_PASS2

    def test_pass2_allows_empty_result(self) -> None:
        """억지 생성을 막기 위해 빈 결과를 명시적으로 허용한다."""
        assert '{"missed":[]}' in SYSTEM_PASS2
        assert "억지로 만들어내지 마라" in SYSTEM_PASS2

    def test_pass2_covers_all_labels(self) -> None:
        for label in PII_LABELS:
            assert label in SYSTEM_PASS2


class TestSchemas:
    def test_pass1_schema_is_valid_json(self) -> None:
        json.dumps(PASS1_SCHEMA)

    def test_pass1_schema_enum_matches_context_labels(self) -> None:
        enum = PASS1_SCHEMA["properties"]["regions"]["items"]["properties"]["type"]["enum"]
        assert enum == list(CONTEXT_LABELS)

    def test_pass1_schema_requires_idx(self) -> None:
        item = PASS1_SCHEMA["properties"]["regions"]["items"]
        assert "idx" in item["required"]
        assert item["additionalProperties"] is False

    def test_pass2_schema_does_not_require_coords(self) -> None:
        """pass2 는 idx 또는 bbox_norm 중 하나만 채우므로 둘 다 optional 이다."""
        item = PASS2_SCHEMA["properties"]["missed"]["items"]
        assert "idx" not in item["required"]
        assert "bbox_norm" not in item["required"]
        assert set(item["required"]) == {"type", "conf", "reason"}

    def test_pass2_bbox_norm_is_four_numbers(self) -> None:
        spec = PASS2_SCHEMA["properties"]["missed"]["items"]["properties"]["bbox_norm"]
        assert spec["minItems"] == 4 and spec["maxItems"] == 4
