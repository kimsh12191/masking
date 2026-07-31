"""체크섬 검증 테스트.

여기서 쓰는 번호는 모두 **체크섬 알고리즘으로 합성한 가짜 값**이다.
실제 개인정보가 아니다.
"""

from __future__ import annotations

import pytest

from pii_pipeline.rules.checksums import (
    digits_only,
    validate_biz_no,
    validate_corp_no,
    validate_foreign_id,
    validate_luhn,
    validate_rrn,
)


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


class TestForeignId:
    @pytest.mark.parametrize("value", ["9506155234561", "8803046123454"])
    def test_valid_legacy_with_checksum(self, value: str) -> None:
        assert validate_foreign_id(value) is True

    def test_new_format_without_checksum_passes(self) -> None:
        """2020-10 이후 발급분은 체크섬이 없다. recall 우선으로 통과시킨다."""
        assert validate_foreign_id("9506155234569") is True

    @pytest.mark.parametrize(
        "value",
        [
            "9012311234563",  # 성별코드 1 -> 내국인
            "950615523456",   # 12자리
            "9513155234561",  # 13월
        ],
    )
    def test_invalid(self, value: str) -> None:
        assert validate_foreign_id(value) is False


class TestBizNo:
    @pytest.mark.parametrize(
        "value", ["2148108288", "214-81-08288", "1208600633", "1018112341"]
    )
    def test_valid(self, value: str) -> None:
        assert validate_biz_no(value) is True

    @pytest.mark.parametrize("value", ["2148108289", "214810828", "21481082888"])
    def test_invalid(self, value: str) -> None:
        assert validate_biz_no(value) is False


class TestCorpNo:
    def test_valid(self) -> None:
        assert validate_corp_no("1101112345670") is True

    def test_rrn_checksum_does_not_apply(self) -> None:
        """법인번호는 주민번호 체크섬을 통과하지 않아야 구분이 가능하다."""
        assert validate_corp_no("1101112345670") is True
        assert validate_rrn("1101112345670") is False

    @pytest.mark.parametrize("value", ["1101112345671", "110111234567"])
    def test_invalid(self, value: str) -> None:
        assert validate_corp_no(value) is False


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
