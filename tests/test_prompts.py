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
        assert lines[0].startswith("[00] (0.05,0.03,0.24,0.05)")
        assert "성명" in lines[0]
        assert "홍길동" in lines[1]

    def test_renders_full_extent_not_just_top_left(self) -> None:
        """좌상단만 주면 모델이 박스 폭을 몰라 인접 판단(그룹화)을 못 한다."""
        text = render_box_list(boxes(), PAGE_W, PAGE_H)
        coords = text.splitlines()[0].split("(")[1].split(")")[0].split(",")
        assert len(coords) == 4
        x1, y1, x2, y2 = (float(c) for c in coords)
        assert x2 > x1 and y2 > y1

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


class TestOutputFormatIsStated:
    """두 pass 모두 **감싸는 객체**를 명시해야 한다.

    현장 사고: pass2 프롬프트에 항목 형식만 있고 ``{"missed":[...]}`` 래퍼가
    없어서 모델이 항목을 한 줄에 하나씩(JSONL) 뱉었다. ``json.loads`` 가
    ``Extra data: line 2 column 1`` 로 죽어 pass2 결과가 전 페이지 전량
    날아갔다. 스키마(guided decoding)에 의존하지 말 것 — 서버 설정에 따라
    실제로 강제되지 않는다.
    """

    def test_pass1_states_wrapper_object(self) -> None:
        assert "출력 형식" in SYSTEM_PASS1
        assert '{"regions":[' in SYSTEM_PASS1

    def test_pass2_states_wrapper_object(self) -> None:
        assert "출력 형식" in SYSTEM_PASS2
        assert '{"missed":[' in SYSTEM_PASS2

    def test_pass2_forbids_one_item_per_line(self) -> None:
        assert "하나의 `missed` 배열 안에" in SYSTEM_PASS2
        assert "한 줄에 하나씩 나열하지 마라" in SYSTEM_PASS2

    def test_pass2_example_nests_items_in_wrapper(self) -> None:
        """항목 예시만 보여주면 모델이 그 모양 그대로 출력한다."""
        example = SYSTEM_PASS2.split("`missed` 배열에 담는 항목 형식")[-1]
        assert '{"missed":[' in example


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

    def test_both_passes_cover_label_glued_to_value(self) -> None:
        """OCR 이 라벨과 값을 묶으면 구분자가 아예 없다 ("신청인홍길동").

        구분자가 있는 예시만 주면 모델이 "라벨이 붙은 값" 으로 인식하지 못하고
        박스를 건너뛴다. 뒤에 붙는 경칭("귀하", "(인)")도 같은 문제다.
        """
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "신청인홍길동" in prompt
            assert "홍길동 귀하" in prompt
            assert "홍길동(인)" in prompt

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

    def test_pass1_states_output_wrapper_for_nonempty_case(self) -> None:
        """빈 결과 형식만 알려주면 정상 출력 구조는 문법 제약에만 의존한다."""
        assert '{"regions":[{"idx"' in SYSTEM_PASS1.replace(" ", "")

    def test_pass1_examples_include_required_conf(self) -> None:
        """예시가 스키마 required 를 위반하면 few-shot 이 계약과 어긋난다."""
        for line in SYSTEM_PASS1.splitlines():
            if '"type"' in line and '"idx"' in line:
                assert '"conf"' in line, f"conf 없는 예시: {line}"

    def test_pass2_example_includes_required_reason(self) -> None:
        for line in SYSTEM_PASS2.splitlines():
            if '"type"' in line and '"conf"' in line and line.strip().startswith("예:"):
                assert '"reason"' in line, f"reason 없는 예시: {line}"

    def test_both_passes_recover_rule_layer_misses(self) -> None:
        """OCR 오독으로 규칙 정규식이 빗나간 정형 식별자를 회수해야 한다.

        pass-1 의 라벨 집합에는 RRN/PHONE 이 없으므로, 지시가 없으면 모델이
        눈으로 보고도 붙일 라벨이 없어 그냥 버린다.
        """
        assert "규칙 검사가 놓친 정형 식별자" in SYSTEM_PASS1
        assert "0IO-l234-5678" in SYSTEM_PASS1
        assert "규칙 검사가 놓친 정형 식별자" in SYSTEM_PASS2

    def test_both_passes_handle_mixed_labels_as_other(self) -> None:
        """한 박스에 여러 종류가 섞이면 종류를 고르다 나머지를 버리면 안 된다."""
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "여러 종류가 섞여 있으면" in prompt
            assert "OTHER" in prompt

    def test_confirmed_tag_does_not_silence_other_pii_in_same_box(self) -> None:
        """<CONFIRMED:RRN> 박스의 이름까지 침묵시키면 부분 마스킹 시 유출된다."""
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "그 **타입만**" in prompt
            assert "다른 종류" in prompt

    def test_both_passes_include_issuing_institution(self) -> None:
        """방침은 넓게 잡기다 — 은행명·지점명·부서명도 마스킹 대상이다."""
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "기관 자체의 정보도 포함한다" in prompt
            assert "하나은행 강남지점" in prompt
            assert "기관 직원이라도 NAME" in prompt

    def test_both_passes_include_all_dates(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "모든 날짜" in prompt
            assert "만기일" in prompt

    def test_both_passes_include_identifier_numbers(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "모든 식별번호" in prompt
            assert "사번" in prompt

    def test_only_non_identifying_values_are_excluded(self) -> None:
        """제외 대상은 아무도 식별하지 않는 값뿐이어야 한다."""
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "아무도 식별하지 않는 값" in prompt
            assert "이자율" in prompt

    def test_both_passes_give_conf_calibration(self) -> None:
        """conf 앵커가 없으면 모델이 전부 0.9 로 채워 needs_review 가 무의미해진다."""
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "conf 기준" in prompt
            assert "0.5 미만" in prompt


class TestEvasionGuidance:
    """마스킹 회피 표기 지침.

    규칙 레이어는 ``normalize.py`` 로 정규화해서 대응하지만, 규칙이 구조적으로
    못 잡는 형태(가려진 이름, 벌려 쓴 숫자)는 LLM 이 최종 안전망이다.
    """

    def test_both_passes_warn_about_evasion(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "회피 표기를 놓치지 마라" in prompt

    def test_hangul_numeral_evasion_is_shown(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "구1공8공4" in prompt
            assert "공일공-일이삼사-오육칠팔" in prompt

    def test_homoglyph_evasion_is_shown(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "0lO-1234-5678" in prompt
            assert "9O1231-1234567" in prompt

    def test_spacing_and_partial_masking_evasion_is_shown(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "홍 길 동" in prompt
            assert "901231-1******" in prompt

    def test_email_separator_evasion_is_shown(self) -> None:
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "hong(at)hana(dot)com" in prompt
            assert "골뱅이" in prompt

    def test_partially_masked_values_are_still_pii(self) -> None:
        """이미 가려져 있다고 건너뛰면 남은 부분으로 재식별된다."""
        for prompt in (SYSTEM_PASS1, SYSTEM_PASS2):
            assert "일부만 가려진 값도 개인정보다" in prompt

    def test_pass1_maps_evasion_to_available_labels(self) -> None:
        """pass-1 의 라벨 집합에는 PHONE 이 없으므로 OTHER 로 보내야 한다."""
        assert "번호류를 OTHER 로 보고하면 된다" in SYSTEM_PASS1

    def test_pass2_maps_evasion_to_exact_labels(self) -> None:
        """pass-2 는 전체 라벨을 쓸 수 있으므로 정확한 type 을 요구한다."""
        assert "원래 값에 맞는 정확한 type" in SYSTEM_PASS2

    def test_pass1_defines_ocr_failed_behaviour(self) -> None:
        """<OCR_FAILED> 처리 지침이 없으면 동작이 정의되지 않는다."""
        assert "<OCR_FAILED>" in SYSTEM_PASS1
        assert "근거 없이 추측하지 마라" in SYSTEM_PASS1

    def test_pass2_requires_one_coordinate_field(self) -> None:
        """"하나만" 만 있으면 둘 다 비운 응답이 조용히 폐기된다."""
        assert "반드시 하나는 채워야 한다" in SYSTEM_PASS2
        assert "버려진다" in SYSTEM_PASS2

    def test_example_box_numbers_are_not_reused_for_different_items(self) -> None:
        """번호가 의미를 갖는 프롬프트에서 같은 번호를 다른 항목에 쓰면 혼란스럽다."""
        import re

        seen: dict[str, set[str]] = {}
        for m in re.finditer(r'\{"idx":\[([\d,]+)\],"type":"(\w+)"', SYSTEM_PASS1):
            for idx in m.group(1).split(","):
                seen.setdefault(idx, set()).add(m.group(2))
        collisions = {k: v for k, v in seen.items() if len(v) > 1}
        assert not collisions, f"같은 번호가 다른 라벨로 쓰였다: {collisions}"


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
