"""값 전파 테스트.

핵심 요구: 한 곳에서 개인정보로 확정된 값은 문서의 나머지 위치에서도
일괄 회수돼야 한다. 동시에 과검 보호 장치(최소 길이, 필드 라벨 제거,
라벨 좁히기)가 실제로 동작해야 한다.
"""

from __future__ import annotations

from pii_pipeline.merge import ClaimLedger
from pii_pipeline.ocr.layout import assign_reading_order
from pii_pipeline.propagate import (
    PropagateConfig,
    normalize,
    propagate_regions,
    seed_candidates,
    seed_value,
)
from pii_pipeline.schema import OcrBox, OcrStatus, PiiRegion, Source

PAGE_W, PAGE_H = 2000, 3000


def make_boxes(rows: list[list[str]]) -> list[OcrBox]:
    boxes: list[OcrBox] = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x1 = 100 + c * 400
            y1 = 100 + r * 100
            boxes.append(OcrBox(index=-1, bbox=(x1, y1, x1 + 380, y1 + 60), text=text))
    return assign_reading_order(boxes)


def region(
    type_: str,
    text: str,
    idx: int,
    boxes: list[OcrBox],
    source: Source = Source.LLM_PASS1,
    conf: float = 0.9,
) -> PiiRegion:
    return PiiRegion(
        id="",
        type=type_,
        bbox=boxes[idx].bbox,
        source=source,
        confidence=conf,
        text=text,
        member_boxes=[boxes[idx].bbox],
        member_index=[idx],
    )


class TestNormalize:
    def test_strips_separators_and_space(self) -> None:
        assert normalize("010-1234-5678") == "01012345678"
        assert normalize("901231 - 1234567") == "9012311234567"
        assert normalize("서울시 강남구") == "서울시강남구"

    def test_keeps_alphanumeric_and_hangul(self) -> None:
        assert normalize("M12345678") == "M12345678"


class TestSeedValue:
    def test_strips_prefix_field_label(self) -> None:
        assert seed_value("담당자: 조민석") == "조민석"
        assert seed_value("성명 홍길동") == "홍길동"
        assert seed_value("주소 서울특별시 강남구") == "서울특별시 강남구"

    def test_leaves_bare_value_alone(self) -> None:
        assert seed_value("홍길동") == "홍길동"

    def test_strips_stacked_labels(self) -> None:
        assert seed_value("신청인 성명: 홍길동") == "홍길동"

    def test_strips_label_without_delimiter(self) -> None:
        """OCR 이 라벨과 값을 한 박스로 묶으면 공백조차 없다."""
        assert seed_value("신청인홍길동") == "홍길동"
        assert seed_value("성명홍길동") == "홍길동"

    def test_strips_bracketed_prefix_label(self) -> None:
        assert seed_value("(신청인)홍길동") == "홍길동"

    def test_strips_suffix_label(self) -> None:
        """한국 서식은 라벨이 뒤에 붙는 경우가 흔하다.

        떼지 않으면 씨앗이 "홍길동귀하" 가 되어 문서의 다른 "홍길동" 과
        매칭되지 않고, 그 값의 전파가 전부 죽는다.
        """
        assert seed_value("홍길동 귀하") == "홍길동"
        assert seed_value("홍길동(인)") == "홍길동"
        assert seed_value("홍길동 (서명)") == "홍길동"
        assert seed_value("홍길동님") == "홍길동"
        assert seed_value("홍길동 신청인") == "홍길동"

    def test_candidates_keep_the_original(self) -> None:
        """라벨을 뗀 값이 맞다고 확정할 수 없으므로 원문도 후보로 남긴다."""
        assert seed_candidates("홍길동 귀하") == ["홍길동 귀하", "홍길동"]
        assert seed_candidates("홍길동") == ["홍길동"]


class TestPropagation:
    def test_same_name_elsewhere_is_recovered(self) -> None:
        """이게 이 모듈의 존재 이유다 — 같은 이름의 나머지 등장 위치."""
        boxes = make_boxes(
            [
                ["성명", "홍길동"],
                ["위 본인은 동의합니다", "확인자 홍길동"],
                ["대리인", "김철수"],
            ]
        )
        seed = region("NAME", "홍길동", 1, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        recovered = {r.member_index[0] for r in out}
        assert 3 in recovered          # "확인자 홍길동"
        assert 1 not in recovered      # 씨앗 자기 자신은 다시 만들지 않는다
        assert all(r.source is Source.PROPAGATED for r in out)

    def test_label_and_value_in_one_box_is_recovered(self) -> None:
        """이름이 라벨과 한 박스에 붙어 있어도 마스킹 대상이다.

        "신청인홍길동" 은 OCR 이 라벨과 값을 공백 없이 묶은 흔한 형태다.
        한 곳에서 이름이 확정됐으면 이 박스도 당연히 덮여야 한다.
        """
        boxes = make_boxes([["성명", "홍길동"], ["신청인홍길동", "x"]])
        out = propagate_regions(
            [region("NAME", "홍길동", 1, boxes)], boxes, PAGE_W, PAGE_H
        )
        hit = next(r for r in out if r.member_index == [2])
        start, end = hit.char_span
        assert boxes[2].text[start:end] == "홍길동"

    def test_seed_from_suffix_labeled_box_still_propagates(self) -> None:
        """씨앗 쪽에 라벨이 붙어 있어도 나머지 위치를 회수해야 한다.

        회귀 방지: 접두 라벨만 떼던 시절에는 씨앗이 "홍길동귀하" 가 되어
        문서의 다른 "홍길동" 이 **전부** 안 잡혔다.
        """
        boxes = make_boxes([["홍길동 귀하"], ["홍길동"], ["신청인홍길동"]])
        out = propagate_regions(
            [region("NAME", "홍길동 귀하", 0, boxes)], boxes, PAGE_W, PAGE_H
        )
        assert {r.member_index[0] for r in out} == {1, 2}

    def test_over_stripped_short_seed_does_not_sweep_the_page(self) -> None:
        """접미 라벨을 떼다 너무 짧아진 후보가 문서를 쓸어버리면 안 된다.

        "대한상호" 의 "상호" 는 필드 라벨이라 떼면 "대한" 이 남는데, ORG 최소
        길이(3)에 걸려 후보에서 탈락한다. 원문 후보는 그대로 살아 있다.
        """
        boxes = make_boxes([["대한상호"], ["대한민국"], ["대한상호저축"], ["주식회사 대한"]])
        out = propagate_regions(
            [region("ORG", "대한상호", 0, boxes)], boxes, PAGE_W, PAGE_H
        )
        assert {r.member_index[0] for r in out} == {2}

    def test_partial_containment_sets_char_span(self) -> None:
        """부분 마스킹을 구현할 다운스트림이 쓸 오프셋이 맞아야 한다."""
        boxes = make_boxes([["성명", "홍길동"], ["확인자 홍길동 (인)", "x"]])
        seed = region("NAME", "홍길동", 1, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        hit = next(r for r in out if r.member_index == [2])
        start, end = hit.char_span
        assert boxes[2].text[start:end] == "홍길동"

    def test_seed_boxes_are_not_duplicated(self) -> None:
        boxes = make_boxes([["성명", "홍길동"]])
        ledger = ClaimLedger()
        ledger.add(1, "NAME")
        seed = region("NAME", "홍길동", 1, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H, claimed=ledger) == []

    def test_rrn_with_different_separators_matches(self) -> None:
        boxes = make_boxes([["901231-1234563"], ["901231 1234563"]])
        seed = region("RRN", "901231-1234563", 0, boxes, source=Source.RULE, conf=1.0)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]
        assert out[0].needs_review is False   # 완전일치는 검토 불필요

    def test_ocr_confusable_typo_is_matched_and_flagged(self) -> None:
        """OCR 오독(0↔O, 1↔l)을 흡수하되 사람 검토 대상으로 남긴다.

        유사도 임계값만으로는 이걸 잡을 수 없다 — 11자 값에서 한 글자만 틀려도
        ``SequenceMatcher`` 비율이 0.909 로 떨어지기 때문에, 임계값을 그 위에
        두면 아무것도 안 걸리고 아래로 내리면 전혀 다른 번호가 걸린다. 그래서
        혼동문자는 ``normalize`` 가 정규화로 처리한다.
        """
        boxes = make_boxes([["010-1234-5678"], ["0lO-1234-5678"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE, conf=1.0)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]
        assert out[0].needs_review is True
        assert "회피 표기 정규화" in (out[0].reason or "")

    def test_hangul_numeral_evasion_propagates(self) -> None:
        """"공1공-1234-5678" 도 같은 전화번호다."""
        boxes = make_boxes([["010-1234-5678"], ["공1공-1234-5678"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE, conf=1.0)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]
        assert out[0].needs_review is True

    def test_fullwidth_evasion_propagates(self) -> None:
        boxes = make_boxes([["010-1234-5678"], ["０１０－１２３４－５６７８"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]

    def test_zero_width_evasion_propagates(self) -> None:
        boxes = make_boxes([["010-1234-5678"], ["010-1234\u200b-5678"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]

    def test_spaced_out_name_propagates(self) -> None:
        """"홍 길 동" 은 구분자 제거만으로 잡힌다 (정규화 이전 단계)."""
        boxes = make_boxes([["홍길동"], ["확인자 홍 길 동"]])
        seed = region("NAME", "홍길동", 0, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]

    def test_exact_match_is_not_flagged_for_review(self) -> None:
        """정규화가 필요 없었던 완전일치는 검토 대상이 아니다."""
        boxes = make_boxes([["010-1234-5678"], ["010-1234-5678 (자택)"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE, conf=1.0)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert out[0].needs_review is False
        assert (out[0].reason or "").endswith("동일")

    def test_folding_never_matches_a_genuinely_different_number(self) -> None:
        boxes = make_boxes([["010-1234-5678"], ["010-1234-9999"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_similarity_path_covers_hangul_ocr_noise(self) -> None:
        """혼동문자 접기는 숫자용이다. 한글 오독은 유사도 경로가 받는다."""
        boxes = make_boxes([["서울특별시 강남구 테헤란로"], ["서울특별시 강남구 테헤린로"]])
        seed = region("ADDRESS", "서울특별시 강남구 테헤란로", 0, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]
        assert out[0].needs_review is True

    def test_similarity_can_be_disabled(self) -> None:
        boxes = make_boxes([["서울특별시 강남구 테헤란로"], ["서울특별시 강남구 테헤린로"]])
        seed = region("ADDRESS", "서울특별시 강남구 테헤란로", 0, boxes)
        cfg = PropagateConfig(min_similarity=1.0)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H, config=cfg) == []

    def test_disabled_config_is_noop(self) -> None:
        boxes = make_boxes([["홍길동"], ["홍길동"]])
        seed = region("NAME", "홍길동", 0, boxes)
        cfg = PropagateConfig(enabled=False)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H, config=cfg) == []

    def test_ledger_is_updated_so_reruns_are_idempotent(self) -> None:
        boxes = make_boxes([["홍길동"], ["홍길동 귀하"]])
        seed = region("NAME", "홍길동", 0, boxes)
        ledger = ClaimLedger()
        ledger.add(0, "NAME")
        first = propagate_regions([seed], boxes, PAGE_W, PAGE_H, claimed=ledger)
        second = propagate_regions([seed], boxes, PAGE_W, PAGE_H, claimed=ledger)
        assert len(first) == 1
        assert second == []

    def test_failed_boxes_are_skipped(self) -> None:
        boxes = make_boxes([["홍길동"], [""]])
        boxes[1].status = OcrStatus.FAILED
        seed = region("NAME", "홍길동", 0, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []


class TestPrecisionGuards:
    def test_field_label_seed_is_rejected(self) -> None:
        """"주소" 같은 라벨이 씨앗이 되면 서식 라벨이 전부 걸린다."""
        boxes = make_boxes([["주소"], ["주소"], ["주소"]])
        seed = region("ADDRESS", "주소", 0, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_field_label_target_box_is_not_masked(self) -> None:
        boxes = make_boxes([["서울특별시 강남구 테헤란로"], ["주소"]])
        seed = region("ADDRESS", "서울특별시 강남구 테헤란로", 0, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert out == []

    def test_label_prefix_is_stripped_before_propagating(self) -> None:
        """씨앗이 "주소 서울시..." 면 "주소" 만 있는 박스가 걸리면 안 된다."""
        boxes = make_boxes(
            [["주소 서울특별시 강남구"], ["주소"], ["서울특별시 강남구 5층"]]
        )
        seed = region("ADDRESS", "주소 서울특별시 강남구", 0, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[2]]

    def test_short_value_is_not_propagated(self) -> None:
        """한 글자 값은 문서 곳곳에 우연히 들어 있다."""
        boxes = make_boxes([["김"], ["김밥천국 영수증"], ["김"]])
        seed = region("NAME", "김", 0, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_org_is_propagated_by_default(self) -> None:
        """방침은 넓게 잡기다 — 기관명도 등장하는 모든 위치를 덮는다."""
        boxes = make_boxes(
            [["하나은행 강남지점"], ["하나은행 강남지점 귀중"], ["기타 문구"]]
        )
        seed = region("ORG", "하나은행 강남지점", 0, boxes)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]

    def test_seed_is_the_whole_value_not_its_tokens(self) -> None:
        """알려진 한계 — 씨앗은 값 전체다. 값의 일부는 씨앗이 되지 않는다.

        "하나은행 리스크관리부" 가 씨앗이면 "하나은행 강남지점" 은 걸리지 않는다
        ("하나은행" 만 따로 씨앗이 되지는 않기 때문). 토큰 단위 씨앗을 만들면
        회수는 늘지만 짧은 토큰("관리부")이 문서 전체를 덮어버린다.
        """
        boxes = make_boxes([["하나은행 리스크관리부"], ["하나은행 강남지점"]])
        seed = region("ORG", "하나은행 리스크관리부", 0, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_labels_can_be_narrowed(self) -> None:
        """과검이 문제가 되면 propagate_labels 로 좁힐 수 있다."""
        boxes = make_boxes([["하나은행 리스크관리부"], ["하나은행 강남지점"]])
        seed = region("ORG", "하나은행 리스크관리부", 0, boxes)
        cfg = PropagateConfig(propagate_labels=("NAME", "RRN"))
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H, config=cfg) == []

    def test_signature_is_never_propagated(self) -> None:
        boxes = make_boxes([["서명"], ["서명"]])
        seed = region("SIGNATURE", "서명", 0, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_seed_cap_warns_instead_of_silently_truncating(self) -> None:
        boxes = make_boxes([[f"홍길동{i:03d}" for i in range(5)]])
        seeds = [region("NAME", f"홍길동{i:03d}", i, boxes) for i in range(5)]
        warnings: list[str] = []
        cfg = PropagateConfig(max_seeds=2)
        propagate_regions(
            seeds, boxes, PAGE_W, PAGE_H, config=cfg, warnings=warnings
        )
        assert any("씨앗" in w for w in warnings)


class TestSimilarityIsRestrictedToTextLabels:
    """번호류에 유사도를 쓰면 다른 번호가 붙는다.

    숫자는 문자열로서 중복성이 없다. 한 글자 다른 번호는 "오독된 같은 번호" 가
    아니라 그냥 다른 번호다. 번호류의 회피 표기는 정규화가 결정론적으로 접으므로
    유사도가 필요하지도 않다.
    """

    def test_passport_does_not_fuzzy_match_a_phone_number(self) -> None:
        """실제로 겪은 오탐 — 정규화 후 "S12345678" vs "012345678" = 0.89."""
        boxes = make_boxes([["S12345678"], ["010-1234-5678"]])
        seed = region("PASSPORT", "S12345678", 0, boxes, source=Source.RULE)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_phone_does_not_fuzzy_match_a_different_phone(self) -> None:
        boxes = make_boxes([["010-1234-5678"], ["010-1234-5679"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_rrn_does_not_fuzzy_match(self) -> None:
        boxes = make_boxes([["901231-1234563"], ["901231-1234564"]])
        seed = region("RRN", "901231-1234563", 0, boxes, source=Source.RULE)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_number_labels_still_match_exactly_after_normalization(self) -> None:
        """유사도를 막아도 회피 표기는 정규화 경로로 잡힌다."""
        boxes = make_boxes([["010-1234-5678"], ["공1공-1234-5678"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]

    def test_text_labels_keep_similarity(self) -> None:
        boxes = make_boxes([["서울특별시 강남구 테헤란로"], ["서울특별시 강남구 테헤린로"]])
        seed = region("ADDRESS", "서울특별시 강남구 테헤란로", 0, boxes)
        assert len(propagate_regions([seed], boxes, PAGE_W, PAGE_H)) == 1
