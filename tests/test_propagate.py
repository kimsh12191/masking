"""값 전파 테스트.

핵심 요구: 한 곳에서 개인정보로 확정된 값은 문서의 나머지 위치에서도
일괄 회수돼야 한다. 동시에 과검 보호 장치(최소 길이, 필드 라벨 제거,
ORG/TITLE 제외)가 실제로 동작해야 한다.
"""

from __future__ import annotations

from pii_pipeline.merge import ClaimLedger
from pii_pipeline.ocr.layout import assign_reading_order
from pii_pipeline.propagate import (
    PropagateConfig,
    fold_confusables,
    normalize,
    propagate_regions,
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

        유사도 임계값으로는 이걸 잡을 수 없다 — 11자 값에서 한 글자만 틀려도
        ``SequenceMatcher`` 비율이 0.909 로 떨어지기 때문에, 임계값을 그 위에
        두면 아무것도 안 걸리고 아래로 내리면 전혀 다른 번호가 걸린다.
        """
        boxes = make_boxes([["010-1234-5678"], ["0lO-1234-5678"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE, conf=1.0)
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H)
        assert [r.member_index for r in out] == [[1]]
        assert out[0].needs_review is True
        assert "혼동문자" in (out[0].reason or "")

    def test_confusable_folding_can_be_disabled(self) -> None:
        boxes = make_boxes([["010-1234-5678"], ["0lO-1234-5678"]])
        seed = region("PHONE", "010-1234-5678", 0, boxes, source=Source.RULE)
        cfg = PropagateConfig(fold_confusables=False, min_similarity=1.0)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H, config=cfg) == []

    def test_confusable_folding_does_not_apply_to_names(self) -> None:
        """"Bob" 을 "8ob" 으로 접으면 전혀 다른 값이 같아진다."""
        assert fold_confusables("Bob") == "Bob"
        assert fold_confusables("0lO12345678") == "01012345678"

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

    def test_org_and_title_are_excluded_by_default(self) -> None:
        """"하나은행"/"과장" 을 전파하면 서식 전체가 마스킹된다."""
        boxes = make_boxes(
            [["하나은행 리스크관리부"], ["하나은행 강남지점"], ["하나은행"]]
        )
        seed = region("ORG", "하나은행 리스크관리부", 0, boxes)
        assert propagate_regions([seed], boxes, PAGE_W, PAGE_H) == []

    def test_org_can_be_opted_in(self) -> None:
        boxes = make_boxes([["삼성전자 반도체사업부"], ["삼성전자 반도체사업부"]])
        seed = region("ORG", "삼성전자 반도체사업부", 0, boxes)
        cfg = PropagateConfig(propagate_labels=("ORG",))
        out = propagate_regions([seed], boxes, PAGE_W, PAGE_H, config=cfg)
        assert [r.member_index for r in out] == [[1]]

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
