"""라벨 스키마 고정 테스트.

라벨을 늘리거나 줄일 때 프롬프트·guided-decoding 스키마·규칙 레이어가
함께 움직이는지 확인한다.
"""

from __future__ import annotations

from pii_pipeline.rules.detectors import ACCOUNT_PATTERN, PATTERNS
from pii_pipeline.schema import (
    CONTEXT_LABELS,
    PASS1_SCHEMA,
    PASS2_SCHEMA,
    PII_LABELS,
    RULE_LABELS,
)

#: 사용자가 확정한 탐지 대상. 순서까지 고정한다.
EXPECTED_LABELS = (
    # 핵심 9종
    "NAME", "RRN", "ADDRESS", "EMAIL", "IP",
    "ACCOUNT_NO", "CARD_NO", "PHONE", "PASSPORT",
    # 추가 9종
    "FOREIGN_ID", "DRIVER_LICENSE", "BIZ_NO", "CORP_NO",
    "BIRTH", "ORG", "TITLE", "SIGNATURE", "OTHER",
)


class TestLabelSet:
    def test_exact_labels_and_order(self) -> None:
        assert PII_LABELS == EXPECTED_LABELS

    def test_count(self) -> None:
        assert len(PII_LABELS) == 18

    def test_no_duplicates(self) -> None:
        assert len(set(PII_LABELS)) == len(PII_LABELS)

    def test_core_nine_present(self) -> None:
        core = {
            "NAME", "RRN", "ADDRESS", "EMAIL", "IP",
            "ACCOUNT_NO", "CARD_NO", "PHONE", "PASSPORT",
        }
        assert core <= set(PII_LABELS)


class TestRuleContextSplit:
    def test_rule_labels_are_subset(self) -> None:
        assert set(PII_LABELS) >= RULE_LABELS

    def test_context_is_the_complement(self) -> None:
        assert set(CONTEXT_LABELS) == set(PII_LABELS) - RULE_LABELS

    def test_split_is_disjoint_and_total(self) -> None:
        assert not (set(CONTEXT_LABELS) & RULE_LABELS)
        assert set(CONTEXT_LABELS) | RULE_LABELS == set(PII_LABELS)

    def test_context_labels_keep_declaration_order(self) -> None:
        assert list(CONTEXT_LABELS) == [
            lbl for lbl in PII_LABELS if lbl not in RULE_LABELS
        ]

    def test_expected_rule_labels(self) -> None:
        assert {
            "RRN", "FOREIGN_ID", "PASSPORT", "DRIVER_LICENSE",
            "BIZ_NO", "CORP_NO", "CARD_NO", "PHONE", "EMAIL", "IP", "ACCOUNT_NO",
        } == RULE_LABELS

    def test_expected_context_labels(self) -> None:
        assert set(CONTEXT_LABELS) == {
            "NAME", "ADDRESS", "BIRTH", "ORG", "TITLE", "SIGNATURE", "OTHER",
        }


class TestRuleLayerCoverage:
    """``RULE_LABELS`` 로 선언한 라벨은 실제 탐지 경로가 있어야 한다."""

    #: 자체 정규식 없이 다른 경로로 산출되는 라벨
    INDIRECT = {
        # 13자리 RRN 패턴에서 체크섬 + 성별코드로 분기된다 (_classify_13digit)
        "FOREIGN_ID": "RRN 패턴 + _classify_13digit 분기",
        "CORP_NO": "RRN 패턴 + _classify_13digit 분기",
        # 문맥 키워드 기반 별도 경로 (ACCOUNT_PATTERN)
        "ACCOUNT_NO": "ACCOUNT_PATTERN + 문맥 키워드",
    }

    def test_every_rule_label_has_a_detection_path(self) -> None:
        covered = {label for label, _ in PATTERNS} | set(self.INDIRECT)
        missing = RULE_LABELS - covered
        assert not missing, f"탐지 경로가 없는 규칙 라벨: {sorted(missing)}"

    def test_thirteen_digit_dispatch_returns_all_three_labels(self) -> None:
        """RRN / FOREIGN_ID / CORP_NO 가 한 패턴에서 갈라져 나온다."""
        from pii_pipeline.rules.detectors import _classify_13digit

        assert _classify_13digit("9012311234563") == ("RRN", True)
        assert _classify_13digit("9506155234561")[0] == "FOREIGN_ID"
        assert _classify_13digit("1101112345670") == ("CORP_NO", True)

    def test_thirteen_digit_dispatch_keeps_unverified_as_rrn_candidate(self) -> None:
        """체크섬 실패는 폐기하지 않고 RRN 후보로 남긴다 (recall 우선)."""
        from pii_pipeline.rules.detectors import _classify_13digit

        assert _classify_13digit("9012311234564") == ("RRN", False)

    def test_no_pattern_for_unknown_label(self) -> None:
        for label, _ in PATTERNS:
            assert label in PII_LABELS, f"스키마에 없는 라벨의 패턴: {label}"

    def test_account_pattern_exists(self) -> None:
        assert ACCOUNT_PATTERN.pattern

    def test_ip_has_two_patterns(self) -> None:
        """IPv4 와 IPv6 를 따로 다룬다."""
        assert sum(1 for label, _ in PATTERNS if label == "IP") == 2

    def test_ip_patterns_come_first(self) -> None:
        """IP 는 구조 검증을 통과하면 다른 숫자 패턴보다 먼저 구간을 선점한다."""
        assert [label for label, _ in PATTERNS][:2] == ["IP", "IP"]


class TestGuidedSchemas:
    def test_pass1_enum_matches_context_labels(self) -> None:
        enum = PASS1_SCHEMA["properties"]["regions"]["items"]["properties"]["type"]["enum"]
        assert enum == list(CONTEXT_LABELS)

    def test_pass2_enum_matches_all_labels(self) -> None:
        enum = PASS2_SCHEMA["properties"]["missed"]["items"]["properties"]["type"]["enum"]
        assert enum == list(PII_LABELS)
