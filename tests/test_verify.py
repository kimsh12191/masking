"""③ 검증 및 최종 정리 테스트.

이 모듈의 존재 이유는 옛 규칙 레이어의 실패를 반복하지 않는 것이다.
그래서 **그 실패 사례 자체를 회귀 테스트로 박아 둔다** — 12자리/13자리 혼동,
체크섬 없는 라벨의 conf 1.00, 항목명 셀 오탐.
"""

from __future__ import annotations

from pii_pipeline.rules.checksums import validate_rrn
from pii_pipeline.schema import Agreement, PiiRegion, Source
from pii_pipeline.verify import (
    DIGIT_LENGTHS,
    VerifyConfig,
    finalize,
    is_field_label,
    verify_regions,
)

#: 체크섬을 통과하는 실제 형식의 가짜 주민등록번호.
VALID_RRN = "901231-1234563"
#: 형식은 맞지만 체크섬이 깨진 값.
BAD_RRN = "901231-1234561"


def region(**kw) -> PiiRegion:
    base = dict(
        id="", type="NAME", bbox=(100, 100, 300, 140),
        source=Source.OCR_REFINED, confidence=0.9, agreement=Agreement.EXACT,
    )
    base.update(kw)
    return PiiRegion(**base)  # type: ignore[arg-type]


class TestFixtures:
    """테스트 상수 자체를 검산한다 (틀린 상수로 통과하는 테스트를 막는다)."""

    def test_valid_rrn_passes_checksum(self) -> None:
        assert validate_rrn(VALID_RRN)

    def test_bad_rrn_fails_checksum(self) -> None:
        assert not validate_rrn(BAD_RRN)


class TestIsFieldLabel:
    def test_exact_label_words(self) -> None:
        assert is_field_label("성명")
        assert is_field_label("주민등록번호")
        assert is_field_label("성 명")        # 공백 제거 후 일치
        assert is_field_label("(성명)")

    def test_label_with_a_value_is_not_a_label(self) -> None:
        assert not is_field_label("성명 홍길동")
        assert not is_field_label("담당자: 조민석")

    def test_values_are_not_labels(self) -> None:
        assert not is_field_label("홍길동")
        assert not is_field_label(VALID_RRN)

    def test_empty_is_not_a_label(self) -> None:
        assert not is_field_label(None)
        assert not is_field_label("")


class TestChecksum:
    def test_passing_checksum_sets_verified(self) -> None:
        r = region(type="RRN", text=VALID_RRN)
        verify_regions([r])
        assert r.verified is True
        assert r.checksum == "ok"

    def test_failing_checksum_flags_review(self) -> None:
        r = region(type="RRN", text=BAD_RRN)
        verify_regions([r])
        assert r.verified is False
        assert r.checksum == "failed"
        assert r.needs_review is True

    def test_label_without_a_checksum_reports_none(self) -> None:
        """이게 옛 버그의 핵심이다.

        옛 규칙 레이어는 라벨이 ``VALIDATORS`` 에 없다는 이유로 ``checksum_ok``
        초기값 ``True`` 를 그대로 남겨 confidence 1.00 + 검토 불필요로 확정했다.
        여기서는 "체크섬이 없다" 가 ``None`` 이고 ``verified`` 는 절대 참이 되지
        않는다. 10종 중 체크섬이 있는 것은 ``RRN``/``CARD_NO`` 둘뿐이므로 이
        경로가 대부분의 라벨에 해당한다.
        """
        r = region(type="PASSPORT", text="M12345678")
        verify_regions([r])
        assert r.checksum is None
        assert r.verified is False

    def test_either_engine_can_satisfy_the_checksum(self) -> None:
        """OCR 이 한 자리를 흘렸어도 VLM 이 온전하면 값은 실재한다."""
        r = region(type="RRN", text="901231-123456", vlm_text=VALID_RRN)
        verify_regions([r])
        assert r.verified is True

    def test_verified_and_agreed_clears_review(self) -> None:
        """검토 큐를 이 조건으로 비워야 정작 위험한 건이 눈에 띈다."""
        r = region(type="RRN", text=VALID_RRN, agreement=Agreement.EXACT, needs_review=True)
        verify_regions([r])
        assert r.confidence == 1.0
        assert r.needs_review is False

    def test_verified_but_disagreed_stays_in_review(self) -> None:
        r = region(type="RRN", text=VALID_RRN, agreement=Agreement.NONE)
        verify_regions([r])
        assert r.verified is True
        assert r.confidence < 1.0

    def test_no_text_is_not_a_failure(self) -> None:
        """VLM 이 값을 못 읽은 항목을 '체크섬 실패' 로 만들면 안 된다.

        검증할 문자열이 없는 것과 검증에서 떨어진 것은 다르다. 전자는
        ``locate.py`` 가 이미 ``needs_review`` 로 넘겼다.
        """
        r = region(type="RRN", text=None, vlm_text=None, source=Source.VLM_COARSE, coarse=True)
        verify_regions([r])
        assert r.checksum is None
        assert r.verified is False


class TestDigitLengths:
    def test_table_matches_the_prompt_format_table(self) -> None:
        assert DIGIT_LENGTHS["RRN"] == (13,)
        assert DIGIT_LENGTHS["CARD_NO"] == (15, 16)

    def test_only_covers_labels_with_a_fixed_length(self) -> None:
        """이름·주소·전화·계좌는 자리수가 값마다 달라 검사 대상이 아니다."""
        assert set(DIGIT_LENGTHS) == {"RRN", "CARD_NO"}

    def test_wrong_digit_count_flags_review(self) -> None:
        r = region(type="CARD_NO", text="1234-5678-9012")  # 12자리
        verify_regions([r])
        assert r.needs_review is True
        assert "자리수 불일치" in (r.reason or "")

    def test_reason_names_the_label_the_digits_fit(self) -> None:
        """'15자리인데 13자리다' 만으로는 무엇으로 고쳐야 할지 알 수 없다."""
        r = region(type="CARD_NO", text=BAD_RRN)  # 13자리, RRN 체크섬 실패
        verify_regions([r], VerifyConfig(retype_on_checksum=False))
        assert "RRN" in (r.reason or "")

    def test_untracked_labels_are_not_checked(self) -> None:
        r = region(type="NAME", text="홍길동")
        verify_regions([r])
        assert "자리수" not in (r.reason or "")


class TestRetype:
    def test_thirteen_digits_with_rrn_checksum_is_an_rrn(self) -> None:
        """스크린샷에서 관측된 오류의 현재 형태.

        옛 관측은 주민등록번호가 **면허번호로** 확정된 것이었다. 면허번호 라벨이
        없어진 뒤 같은 오독은 다른 숫자 라벨(계좌·카드)로 나타난다.
        """
        r = region(type="ACCOUNT_NO", text=VALID_RRN)
        verify_regions([r])
        assert r.type == "RRN"
        assert r.verified is True
        assert "type 교정" in (r.reason or "")

    def test_retype_is_logged_as_a_warning(self) -> None:
        warnings: list[str] = []
        verify_regions([region(type="CARD_NO", text=VALID_RRN)], warnings=warnings)
        assert any("type 교정" in w for w in warnings)

    def test_can_be_turned_off(self) -> None:
        """끈다고 신호를 버리는 것은 아니다 — 검토 플래그와 사유는 남는다.

        예전에는 이 플래그가 **우연히** 세워지고 있었다. 면허·사업자·법인번호가
        전부 ``DIGIT_LENGTHS`` 에 있어서 13자리 값이 어느 라벨에 붙어도
        "자리수 불일치" 가 떴기 때문이다. 라벨셋이 10종으로 좁아지면서 그 우연이
        없어졌고, 그때 이 건이 조용히 무플래그로 통과했다.
        """
        r = region(type="ACCOUNT_NO", text=VALID_RRN)
        verify_regions([r], VerifyConfig(retype_on_checksum=False))
        assert r.type == "ACCOUNT_NO"
        assert r.needs_review is True
        assert "type 의심" in (r.reason or "")

    def test_turning_it_off_does_not_let_an_exact_match_clear_the_review(self) -> None:
        """값이 실재한다는 것과 라벨이 맞다는 것은 다른 문제다.

        체크섬 통과 + 두 엔진 일치는 검토 큐를 비우는 조건인데, 라벨이 의심스러운
        건까지 비우면 그게 유일한 신호였다.
        """
        r = region(
            type="CARD_NO", text=VALID_RRN, vlm_text=VALID_RRN,
            agreement=Agreement.EXACT,
        )
        verify_regions([r], VerifyConfig(retype_on_checksum=False))
        assert r.needs_review is True

    def test_does_not_touch_a_value_already_labelled_rrn(self) -> None:
        """이미 맞는 라벨을 다시 쓰면 사유 문구만 늘어난다.

        예전에는 ``RRN``/``FOREIGN_ID``/``CORP_NO`` 를 한 가족으로 묶어 그 안의
        혼동을 건드리지 않았다. 13자리 6-7 을 담을 라벨이 ``RRN`` 하나뿐이 된
        뒤로는 그 예외가 필요 없다.
        """
        r = region(type="RRN", text=VALID_RRN)
        verify_regions([r])
        assert r.type == "RRN"
        assert "type 교정" not in (r.reason or "")

    def test_does_not_retype_on_a_failed_checksum(self) -> None:
        """체크섬이 통과하지 않으면 증명이 아니다. 추측으로 라벨을 바꾸지 않는다."""
        r = region(type="ACCOUNT_NO", text=BAD_RRN)
        verify_regions([r])
        assert r.type == "ACCOUNT_NO"

    def test_does_not_retype_a_twelve_digit_value(self) -> None:
        r = region(type="ACCOUNT_NO", text="11-12-345678-01")
        verify_regions([r])
        assert r.type == "ACCOUNT_NO"


class TestFinalizeFieldLabels:
    def test_label_only_region_is_dropped(self) -> None:
        """스크린샷에서 관측: '성 명' 헤더 셀에 NAME 박스가 붙었다."""
        warnings: list[str] = []
        out = finalize([region(type="NAME", vlm_text="성명", text="성명")], warnings=warnings)
        assert out == []
        assert any("항목명만 있는 영역 제외" in w for w in warnings)

    def test_vlm_label_error_is_caught_even_if_ocr_read_a_value(self) -> None:
        out = finalize([region(type="NAME", vlm_text="주민등록번호", text="주민등록번호")])
        assert out == []

    def test_label_with_a_value_survives(self) -> None:
        out = finalize([region(type="NAME", vlm_text="조민석", text="담당자: 조민석")])
        assert len(out) == 1

    def test_ocr_value_under_a_label_vlm_text_is_kept(self) -> None:
        """VLM 은 항목명을 읽었지만 기하 선택이 값 칸을 골랐다면 **버리지 않는다.**

        미탐이 확정되는 경로다. 헤더를 덧칠하는 손해가 이름을 놓치는 손해보다
        작으므로 남기고 검토로 넘긴다.
        """
        out = finalize([region(type="NAME", vlm_text="성 명", text="홍길동")])
        assert len(out) == 1
        assert out[0].needs_review is True


class TestFinalizeDedup:
    def test_overlapping_same_label_keeps_one(self) -> None:
        out = finalize([
            region(type="NAME", bbox=(100, 100, 300, 140), confidence=0.7),
            region(type="NAME", bbox=(102, 101, 298, 139), confidence=0.95),
        ])
        assert len(out) == 1
        assert out[0].confidence == 0.95

    def test_refined_beats_coarse_regardless_of_confidence(self) -> None:
        """정확한 좌표와 근사 좌표가 둘 다 있으면 정확한 쪽만 남아야 한다."""
        out = finalize([
            region(source=Source.VLM_COARSE, coarse=True, confidence=0.99),
            region(source=Source.OCR_REFINED, confidence=0.5),
        ])
        assert len(out) == 1
        assert out[0].source is Source.OCR_REFINED

    def test_verified_beats_unverified_at_equal_path(self) -> None:
        out = finalize([
            region(type="RRN", confidence=0.95, verified=False),
            region(type="RRN", confidence=0.9, verified=True),
        ])
        assert len(out) == 1
        assert out[0].verified is True

    def test_different_labels_at_the_same_spot_both_survive(self) -> None:
        """한 칸에 이름과 번호가 같이 있는 서식이 흔하다."""
        out = finalize([
            region(type="NAME", text="홍길동"),
            region(type="RRN", text=VALID_RRN),
        ])
        assert len(out) == 2

    def test_adjacent_cells_are_not_merged(self) -> None:
        out = finalize([
            region(type="NAME", bbox=(100, 100, 200, 140)),
            region(type="NAME", bbox=(210, 100, 310, 140)),
        ])
        assert len(out) == 2


class TestFinalizeIds:
    def test_ids_follow_reading_order(self) -> None:
        out = finalize([
            region(type="NAME", bbox=(100, 900, 200, 940), text="c"),
            region(type="NAME", bbox=(600, 100, 700, 140), text="b"),
            region(type="NAME", bbox=(100, 100, 200, 140), text="a"),
        ])
        assert [r.text for r in out] == ["a", "b", "c"]
        assert [r.id for r in out] == ["r001", "r002", "r003"]

    def test_ids_are_deterministic_across_runs(self) -> None:
        """같은 ID 가 나와야 사람이 붙인 어노테이션을 재연결할 수 있다."""
        def build() -> list[PiiRegion]:
            return [
                region(type="NAME", bbox=(100, 100, 200, 140)),
                region(type="RRN", bbox=(100, 300, 400, 340)),
            ]
        assert [r.id for r in finalize(build())] == [r.id for r in finalize(build())]

    def test_input_list_is_not_returned(self) -> None:
        src = [region()]
        assert finalize(src) is not src
