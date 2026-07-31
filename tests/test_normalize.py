"""회피 표기 정규화 테스트.

두 가지를 동시에 지켜야 한다.

1. **회피 표기는 접힌다** — 안 접히면 규칙 레이어가 놓친다.
2. **평범한 한국어는 건드리지 않는다** — "구"/"이"/"사"/"영" 은 매우 흔한
   음절이다. 무조건 접으면 "강남구" 가 "강남9" 가 되어 주소 탐지가 망가진다.

2번이 더 깨지기 쉽다. 오탐 케이스를 넉넉히 둔다.
"""

from __future__ import annotations

import pytest

from pii_pipeline.normalize import canonical_text, canonicalize, identity


class TestUnicodeNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("０１０－１２３４－５６７８", "010-1234-5678"),   # 전각
            ("①②③④⑤", "12345"),                          # 묶음 문자
            ("⁰¹²³", "0123"),                             # 상단첨자
            ("𝟎𝟏𝟐𝟑", "0123"),                             # 수학 기호 숫자
            ("901231‑1234567", "901231-1234567"),    # non-breaking hyphen
            ("901231–1234567", "901231-1234567"),    # en dash
            ("010​-1234-5678", "010-1234-5678"),     # zero-width space
            ("010﻿-1234­-5678", "010-1234-5678"),  # BOM + soft hyphen
            ("홍​길​동", "홍길동"),
            ("서울시 강남구", "서울시 강남구"),        # NBSP
        ],
    )
    def test_folds(self, raw: str, expected: str) -> None:
        assert canonical_text(raw) == expected


class TestHangulNumerals:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("구1공8공4", "910804"),
            ("공일공-일이삼사-오육칠팔", "010-1234-5678"),
            ("901231-1이3사5육7", "901231-1234567"),
            ("공1공-1234-5678", "010-1234-5678"),
        ],
    )
    def test_folds_when_mixed_with_digits(self, raw: str, expected: str) -> None:
        assert canonical_text(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "서울특별시 강남구 테헤란로",
            "하나은행 강남지점",
            "리스크관리부",
            "제3구역 1234",
            "3구 1234-5678",
            "영업일 기준 3일",
            "본 계약은 서울지점에서 체결한다",
            "이사관리부",
            "일이삼사",          # 4자 순수 한글 수사 — 우연일 수 있어 접지 않는다
            "오사구",            # 3자 — 길이 하한 미달
            "구매팀 김철수",
            "사업자 등록 안내",
            "이 계약의 목적",
        ],
    )
    def test_leaves_ordinary_korean_alone(self, raw: str) -> None:
        assert canonical_text(raw) == raw

    def test_pure_hangul_run_needs_length(self) -> None:
        """짧은 순수 한글 수사는 접지 않고, 번호 길이가 되면 접는다."""
        assert canonical_text("일이삼사") == "일이삼사"
        assert canonical_text("공일공일이삼사오육칠팔") == "01012345678"


class TestHomoglyphs:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0lO-1234-5678", "010-1234-5678"),
            ("9O1231-1234563", "901231-1234563"),
            ("O1O-l234-5678", "010-1234-5678"),
        ],
    )
    def test_folds_in_digit_dense_tokens(self, raw: str, expected: str) -> None:
        assert canonical_text(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "Bob",
            "Seoul Branch",
            "GLOBAL",
            "IBK",
            "Risk Management Dept",
            "OTHER",
        ],
    )
    def test_leaves_words_alone(self, raw: str) -> None:
        """숫자가 없는 영문 단어를 접으면 "Bob" 이 "806" 이 된다."""
        assert canonical_text(raw) == raw


class TestEmailObfuscation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("hong(at)hana(dot)com", "hong@hana.com"),
            ("hong[at]hana[dot]com", "hong@hana.com"),
            ("hong (at) hana.com", "hong@hana.com"),
            ("hong골뱅이hana.co.kr", "hong@hana.co.kr"),
            ("hong앳hana.co.kr", "hong@hana.co.kr"),
            ("hong at hana.co.kr", "hong@hana.co.kr"),
        ],
    )
    def test_restores_at_and_dot(self, raw: str, expected: str) -> None:
        assert canonical_text(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "하나은행 강남지점에서 처리",
            "지점 방문 시 신분증 지참",
            "본 상품의 판매 시점",
            "we will look at the document",
        ],
    )
    def test_does_not_touch_bare_at_or_dot_words(self, raw: str) -> None:
        """전체 모양이 이메일이 아니면 "지점"/"at" 을 건드리면 안 된다."""
        assert canonical_text(raw) == raw


class TestOffsetMapping:
    """``char_span`` 은 원문 기준이어야 한다 — 마스킹은 원본에 적용된다."""

    def test_span_maps_back_to_original_slice(self) -> None:
        raw = "연락처 공일공-일이삼사-오육칠팔 입니다"
        canon = canonicalize(raw)
        pos = canon.text.index("010-1234-5678")
        start, end = canon.span(pos, pos + len("010-1234-5678"))
        assert raw[start:end] == "공일공-일이삼사-오육칠팔"

    def test_span_survives_length_changing_folds(self) -> None:
        raw = "hong(at)hana(dot)com 으로 연락"
        canon = canonicalize(raw)
        pos = canon.text.index("hong@hana.com")
        start, end = canon.span(pos, pos + len("hong@hana.com"))
        assert raw[start:end] == "hong(at)hana(dot)com"

    def test_span_survives_removed_characters(self) -> None:
        raw = "010​-1234-5678"
        canon = canonicalize(raw)
        start, end = canon.span(0, len(canon.text))
        assert raw[start:end] == raw

    def test_empty_text(self) -> None:
        canon = canonicalize("")
        assert canon.text == ""
        assert canon.changed is False
        assert canon.span(0, 0) == (0, 0)

    def test_changed_flag_is_false_when_nothing_folded(self) -> None:
        assert canonicalize("홍길동 010-1234-5678").changed is False

    def test_changed_flag_is_true_when_folded(self) -> None:
        assert canonicalize("구1공8공4").changed is True


class TestIdentity:
    def test_returns_text_unchanged_with_trivial_mapping(self) -> None:
        canon = identity("홍길동")
        assert canon.text == "홍길동"
        assert canon.changed is False
        assert canon.span(0, 3) == (0, 3)
        assert canon.span(1, 2) == (1, 2)
