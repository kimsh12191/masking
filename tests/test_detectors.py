"""규칙 레이어 탐지 테스트.

사용된 번호는 모두 체크섬 알고리즘으로 합성한 가짜 값이다.
"""

from __future__ import annotations

import pytest

from pii_pipeline.ocr.layout import assign_reading_order
from pii_pipeline.rules.detectors import detect, is_field_label
from pii_pipeline.schema import OcrBox, OcrStatus


def make_boxes(rows: list[list[str]]) -> list[OcrBox]:
    """행 단위 텍스트 목록으로 정렬된 OCR 박스를 만든다."""
    boxes: list[OcrBox] = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x1 = 100 + c * 300
            y1 = 100 + r * 60
            boxes.append(OcrBox(index=-1, bbox=(x1, y1, x1 + 280, y1 + 40), text=text))
    return assign_reading_order(boxes)


def labels(boxes: list[OcrBox]) -> set[str]:
    return {h.label for h in detect(boxes)}


class TestIsFieldLabel:
    def test_pure_label(self) -> None:
        assert is_field_label("성명") is True
        assert is_field_label("주민등록번호") is True
        assert is_field_label("성명:") is True

    def test_value_is_not_label(self) -> None:
        assert is_field_label("홍길동") is False
        assert is_field_label("성명 홍길동") is False

    def test_empty_treated_as_label(self) -> None:
        assert is_field_label("   ") is True


class TestRrnDetection:
    def test_detects_hyphenated(self) -> None:
        boxes = make_boxes([["주민등록번호", "901231-1234563"]])
        hits = detect(boxes)
        assert len(hits) == 1
        assert hits[0].label == "RRN"
        assert hits[0].checksum_ok is True
        assert hits[0].needs_review is False

    def test_detects_without_separator(self) -> None:
        boxes = make_boxes([["주민등록번호", "9012311234563"]])
        assert labels(boxes) == {"RRN"}

    def test_checksum_failure_kept_as_candidate(self) -> None:
        """OCR 오독 가능성이 있으므로 폐기하지 않고 검토 대상으로 남긴다."""
        boxes = make_boxes([["주민등록번호", "901231-1234564"]])
        hits = detect(boxes)
        assert len(hits) == 1
        assert hits[0].label == "RRN"
        assert hits[0].checksum_ok is False
        assert hits[0].needs_review is True

    def test_span_points_at_match(self) -> None:
        boxes = make_boxes([["주민 901231-1234563 확인"]])
        hits = detect(boxes)
        assert len(hits) == 1
        s, e = hits[0].span
        assert boxes[0].text[s:e] == "901231-1234563"


class TestForeignIdDetection:
    def test_gender_code_five_maps_to_foreign_id(self) -> None:
        boxes = make_boxes([["외국인등록번호", "950615-5234561"]])
        assert labels(boxes) == {"FOREIGN_ID"}

    def test_domestic_gender_code_stays_rrn(self) -> None:
        boxes = make_boxes([["주민등록번호", "950615-2345672"]])
        assert labels(boxes) == {"RRN"}


class TestCorpNoDetection:
    def test_corp_number_distinguished_from_rrn(self) -> None:
        boxes = make_boxes([["법인등록번호", "110111-2345670"]])
        assert labels(boxes) == {"CORP_NO"}


class TestBizNoDetection:
    def test_detects_valid(self) -> None:
        boxes = make_boxes([["사업자등록번호", "214-81-08288"]])
        hits = detect(boxes)
        assert [h.label for h in hits] == ["BIZ_NO"]
        assert hits[0].checksum_ok is True

    def test_checksum_failure_without_context_is_dropped(self) -> None:
        """문맥 키워드가 없으면 임의의 10자리 숫자를 사업자번호로 보지 않는다."""
        boxes = make_boxes([["참조", "123-45-67890"]])
        assert "BIZ_NO" not in labels(boxes)

    def test_checksum_failure_with_context_is_kept(self) -> None:
        boxes = make_boxes([["사업자등록번호", "123-45-67890"]])
        hits = [h for h in detect(boxes) if h.label == "BIZ_NO"]
        assert len(hits) == 1
        assert hits[0].needs_review is True


class TestCardDetection:
    def test_detects_luhn_valid(self) -> None:
        boxes = make_boxes([["카드번호", "4111-1111-1111-1111"]])
        hits = [h for h in detect(boxes) if h.label == "CARD_NO"]
        assert len(hits) == 1
        assert hits[0].checksum_ok is True

    def test_luhn_failure_without_context_dropped(self) -> None:
        boxes = make_boxes([["승인번호", "4111-1111-1111-1112"]])
        assert "CARD_NO" not in labels(boxes)


class TestPhoneDetection:
    def test_mobile(self) -> None:
        boxes = make_boxes([["연락처", "010-1234-5678"]])
        assert "PHONE" in labels(boxes)

    def test_mobile_without_separator(self) -> None:
        boxes = make_boxes([["연락처", "01012345678"]])
        assert "PHONE" in labels(boxes)

    def test_landline(self) -> None:
        boxes = make_boxes([["전화번호", "02-1234-5678"]])
        assert "PHONE" in labels(boxes)


class TestEmailDetection:
    def test_detects(self) -> None:
        boxes = make_boxes([["이메일", "hong.gildong@example.co.kr"]])
        assert "EMAIL" in labels(boxes)


class TestIpDetection:
    @pytest.mark.parametrize(
        "value",
        ["192.168.0.1", "10.0.0.255", "8.8.8.8", "0.0.0.0", "255.255.255.255"],
    )
    def test_detects_ipv4(self, value: str) -> None:
        boxes = make_boxes([["접속IP", value]])
        hits = [h for h in detect(boxes) if h.label == "IP"]
        assert len(hits) == 1
        assert hits[0].text == value

    @pytest.mark.parametrize(
        "value",
        [
            "fe80::1",
            "::1",
            "fe80::",
            "2001:db8::8a2e:370:7334",
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        ],
    )
    def test_detects_ipv6(self, value: str) -> None:
        boxes = make_boxes([["접속IP", value]])
        hits = [h for h in detect(boxes) if h.label == "IP"]
        assert len(hits) == 1
        assert hits[0].text == value

    @pytest.mark.parametrize(
        "value",
        [
            "256.1.1.1",      # 옥텟 범위 초과
            "300.1.2.3",      # 옥텟 범위 초과
            "1.2.3",          # 3옥텟
            "192.168.0",      # 3옥텟
            "1.2.3.4.5",      # 5옥텟
        ],
    )
    def test_rejects_malformed_ipv4(self, value: str) -> None:
        boxes = make_boxes([["접속IP", value]])
        assert "IP" not in labels(boxes)

    @pytest.mark.parametrize("value", ["12:34:56", "09:30", "23:59:59", "::"])
    def test_rejects_time_like_strings(self, value: str) -> None:
        """IPv6 그룹 수 하한을 낮추면 시각 표기를 IP 로 잡는다."""
        boxes = make_boxes([["처리시각", value]])
        assert "IP" not in labels(boxes)

    def test_rejects_date(self) -> None:
        boxes = make_boxes([["약정일자", "2022. 10. 06."]])
        assert "IP" not in labels(boxes)

    def test_ip_claims_span_before_numeric_patterns(self) -> None:
        """구조 검증을 통과한 IP 는 다른 숫자 패턴에 구간을 빼앗기지 않아야 한다."""
        boxes = make_boxes([["접속IP 192.168.100.200 기록"]])
        hits = detect(boxes)
        assert [h.label for h in hits] == ["IP"]
        assert hits[0].text == "192.168.100.200"

    def test_ip_and_email_in_one_box(self) -> None:
        boxes = make_boxes([["a@b.com 에서 10.1.2.3 접속"]])
        assert labels(boxes) == {"EMAIL", "IP"}

    def test_checksum_flag_set_for_structural_validation(self) -> None:
        """IP 는 체크섬이 없지만 옥텟 범위로 구조 검증이 되므로 확정 처리한다."""
        boxes = make_boxes([["접속IP", "192.168.0.1"]])
        hit = next(h for h in detect(boxes) if h.label == "IP")
        assert hit.checksum_ok is True
        assert hit.needs_review is False


class TestPassportDetection:
    def test_detects_letter_prefixed(self) -> None:
        boxes = make_boxes([["여권번호", "M12345678"]])
        assert "PASSPORT" in labels(boxes)


class TestAccountDetection:
    def test_requires_context_keyword(self) -> None:
        with_ctx = make_boxes([["계좌번호", "110-123-456789"]])
        assert "ACCOUNT_NO" in labels(with_ctx)

    def test_no_keyword_no_detection(self) -> None:
        without_ctx = make_boxes([["관리번호", "110-123-456789"]])
        assert "ACCOUNT_NO" not in labels(without_ctx)

    def test_always_needs_review(self) -> None:
        """계좌번호는 표준 체크섬이 없으므로 항상 검토 대상이다."""
        boxes = make_boxes([["계좌번호", "110-123-456789"]])
        hits = [h for h in detect(boxes) if h.label == "ACCOUNT_NO"]
        assert hits and all(h.needs_review for h in hits)


class TestSkipBehaviour:
    def test_failed_ocr_box_skipped(self) -> None:
        boxes = make_boxes([["주민등록번호", "901231-1234563"]])
        boxes[1].status = OcrStatus.FAILED
        assert detect(boxes) == []

    def test_no_double_claim_on_same_span(self) -> None:
        """한 구간이 두 라벨로 중복 탐지되지 않아야 한다."""
        boxes = make_boxes([["주민등록번호", "901231-1234563"]])
        hits = detect(boxes)
        spans = [h.span for h in hits]
        assert len(spans) == len(set(spans))


class TestMultiplePiiInOneBox:
    def test_two_items_in_one_box(self) -> None:
        boxes = make_boxes([["연락처 010-1234-5678 이메일 a@b.com"]])
        assert labels(boxes) == {"PHONE", "EMAIL"}
