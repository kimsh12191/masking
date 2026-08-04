"""체크섬 검증 테스트.

여기서 쓰는 번호는 모두 **체크섬 알고리즘으로 합성한 가짜 값**이다.
실제 개인정보가 아니다.
"""

from __future__ import annotations

import pytest

from pii_pipeline.rules.checksums import (
    VALIDATORS,
    digits_only,
    validate_luhn,
    validate_rrn,
)
from pii_pipeline.schema import PII_LABELS


class TestDigitsOnly:
    def test_strips_separators(self) -> None:
        assert digits_only("901231-1234563") == "9012311234563"
        assert digits_only("214-81-08288") == "2148108288"
        assert digits_only("없음") == ""


class TestRrn:
    @pytest.mark.parametrize(
        "value",
        [
            "9012311234563",
            "901231-1234563",
            "901231 1234563",
            "9506152345672",
            "0301012345678",
        ],
    )
    def test_valid(self, value: str) -> None:
        assert validate_rrn(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "9012311234564",   # 체크섬 불일치
            "901231123456",    # 12자리
            "90123112345631",  # 14자리
            "9013311234563",   # 13월
            "9012321234563",   # 12월 32일
            "9002301234563",   # 2월 30일
            "9004311234563",   # 4월 31일
        ],
    )
    def test_invalid(self, value: str) -> None:
        assert validate_rrn(value) is False


class TestForeignRegistrationNumbersAreAcceptedAsRrn:
    """외국인등록번호 전용 검증기는 지웠다 (그 라벨이 없어졌다).

    프롬프트가 13자리 6-7 을 성별코드와 무관하게 ``RRN`` 으로 보고하게 하므로,
    **``validate_rrn`` 이 그 값을 받아내야** 검증 단계에서 "체크섬 실패" 로
    떨어지지 않는다. 2020-10 이전 발급분은 주민등록번호와 같은 체크섬을 쓰고,
    ``validate_rrn`` 이 성별코드를 따로 검사하지 않으므로 그대로 통과한다.
    """

    @pytest.mark.parametrize("value", ["9506155234561", "8803046123454"])
    def test_legacy_foreign_id_passes_the_rrn_checksum(self, value: str) -> None:
        assert validate_rrn(value) is True

    def test_gender_code_is_not_used_to_reject(self) -> None:
        """성별코드 5~8(외국인)을 거르면 그 값이 미탐이 된다."""
        assert digits_only("9506155234561")[6] == "5"
        assert validate_rrn("9506155234561") is True


class TestValidatorTable:
    def test_only_covers_labels_that_exist(self) -> None:
        """모델이 낼 수 없는 라벨의 검증기는 호출될 길이 없다."""
        assert set(VALIDATORS) <= set(PII_LABELS)

    def test_the_two_labels_with_a_checksum(self) -> None:
        assert set(VALIDATORS) == {"RRN", "CARD_NO"}


class TestLuhn:
    @pytest.mark.parametrize(
        "value",
        [
            "4111111111111111",
            "4111-1111-1111-1111",
            "5555555555554444",
            "1234567812345670",
        ],
    )
    def test_valid(self, value: str) -> None:
        assert validate_luhn(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "4111111111111112",   # 체크디짓 불일치
            "411111111111",       # 12자리 (하한 미달)
            "41111111111111111111",  # 20자리 (상한 초과)
        ],
    )
    def test_invalid(self, value: str) -> None:
        assert validate_luhn(value) is False
