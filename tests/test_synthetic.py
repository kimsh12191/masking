"""합성 데이터 생성기 테스트.

합성기가 체크섬이 **유효한** 번호를 만들어야 한다. 그렇지 않으면 규칙 레이어를
검증할 수 없다 (전부 needs_review 로 떨어져 정답이 무의미해진다).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_synthetic import (  # noqa: E402
    gen_account,
    gen_biz_no,
    gen_card,
    gen_email,
    gen_ip,
    gen_passport,
    gen_phone,
    gen_rrn,
)

from pii_pipeline.rules.checksums import (  # noqa: E402
    validate_luhn,
    validate_rrn,
)
from pii_pipeline.schema import Agreement, PiiRegion, Source  # noqa: E402
from pii_pipeline.verify import verify_regions  # noqa: E402

SEEDS = list(range(30))


@pytest.mark.parametrize("seed", SEEDS)
class TestGeneratedNumbersAreValid:
    def test_rrn_has_13_digits_and_valid_checksum(self, seed: int) -> None:
        value = gen_rrn(random.Random(seed))
        assert len(value.replace("-", "")) == 13
        assert validate_rrn(value) is True

    def test_foreign_registration_number_is_accepted_as_an_rrn(self, seed: int) -> None:
        """외국인등록번호는 형태가 같으므로 ``RRN`` 으로 보고된다.

        전용 라벨과 전용 검증기가 없어졌으니, 합성기가 만든 값이
        ``validate_rrn`` 을 통과해야 검증 단계에서 체크섬 실패로 떨어지지 않는다.
        """
        value = gen_rrn(random.Random(seed), foreign=True)
        assert validate_rrn(value) is True
        assert value.replace("-", "")[6] in "5678"

    def test_biz_no_is_shaped_like_one_but_is_not_ground_truth(self, seed: int) -> None:
        """사업자등록번호는 탐지 대상에서 빠졌다 — 과검 측정용으로만 인쇄된다.

        체크섬을 맞추지 않는다 (검산할 코드 경로가 없다). 형태만 지킨다.
        """
        value = gen_biz_no(random.Random(seed))
        assert len(value.replace("-", "")) == 10
        assert [len(part) for part in value.split("-")] == [3, 2, 5]

    def test_card_valid_luhn(self, seed: int) -> None:
        value = gen_card(random.Random(seed))
        assert len(value.replace("-", "")) == 16
        assert validate_luhn(value) is True


class TestGeneratedFormats:
    def test_phone_format(self) -> None:
        value = gen_phone(random.Random(1))
        assert value.startswith("010-")
        assert len(value.split("-")) == 3

    def test_account_format(self) -> None:
        assert len(gen_account(random.Random(1)).split("-")) == 3

    def test_email_format(self) -> None:
        assert "@" in gen_email(random.Random(1), "user")

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ip_is_private_range(self, seed: int) -> None:
        """실제 공인 IP 를 만들지 않는다."""
        import ipaddress

        addr = ipaddress.ip_address(gen_ip(random.Random(seed)))
        assert addr.is_private

    @pytest.mark.parametrize("seed", SEEDS)
    def test_passport_format(self, seed: int) -> None:
        value = gen_passport(random.Random(seed))
        assert len(value) == 9
        assert value[0] in "MSR"
        assert value[1:].isdigit()


class TestVerifierAcceptsGeneratedValues:
    """합성 값이 **검증 단계**를 통과하는지 (엔드투엔드 연결 확인).

    이전에는 이 자리에 "규칙 레이어가 합성 값을 탐지하는지" 를 확인하는
    테스트가 있었다. 정규식 탐지 레이어를 제거했으므로 확인해야 하는 것이
    바뀌었다 — 이제 탐지는 VLM 이 하고, 여기서 볼 것은 "VLM 이 이 값을
    보고했을 때 검증기가 통과시키는가" 다.
    """

    def _region(self, label: str, text: str) -> PiiRegion:
        return PiiRegion(
            id="", type=label, bbox=(100, 100, 400, 150),
            source=Source.OCR_REFINED, confidence=0.9,
            text=text, vlm_text=text, agreement=Agreement.EXACT,
        )

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_rrn_passes_checksum(self, seed: int) -> None:
        region = self._region("RRN", gen_rrn(random.Random(seed)))
        verify_regions([region])
        assert region.verified is True
        assert region.checksum == "ok"
        assert region.needs_review is False

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_card_passes_checksum(self, seed: int) -> None:
        region = self._region("CARD_NO", gen_card(random.Random(seed)))
        verify_regions([region])
        assert region.checksum == "ok"

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_rrn_digit_count_is_accepted(self, seed: int) -> None:
        region = self._region("RRN", gen_rrn(random.Random(seed)))
        verify_regions([region])
        assert "자리수 불일치" not in (region.reason or "")

    def test_account_has_no_checksum_but_is_not_a_failure(self) -> None:
        """계좌번호는 표준 체크섬이 없다. '검증 불가' 와 '검증 실패' 는 다르다."""
        region = self._region("ACCOUNT_NO", gen_account(random.Random(3)))
        verify_regions([region])
        assert region.checksum is None
        assert region.verified is False

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_ip_needs_no_checksum(self, seed: int) -> None:
        region = self._region("IP", gen_ip(random.Random(seed)))
        verify_regions([region])
        assert region.checksum is None

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_passport_needs_no_checksum(self, seed: int) -> None:
        region = self._region("PASSPORT", gen_passport(random.Random(seed)))
        verify_regions([region])
        assert region.checksum is None

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_generated_rrn_would_survive_a_mislabel(self, seed: int) -> None:
        """옛 버그 회귀: 주민등록번호가 다른 숫자 라벨로 잘못 붙어도 교정된다."""
        region = self._region("ACCOUNT_NO", gen_rrn(random.Random(seed)))
        verify_regions([region])
        assert region.type == "RRN"


class TestDifficultyCoverage:
    """합성 문서가 실제 문서의 어려운 형태를 실제로 담고 있는지 확인한다.

    이 커버리지가 없어서 "담당자: 조민석" 처럼 라벨과 값이 한 박스에 섞인
    케이스를 프롬프트가 놓치는 결함을 잡지 못했다.
    """

    def _truth(self, tmp_path):
        from make_synthetic import find_font, make_page

        pytest.importorskip("PIL")
        _, truth = make_page(random.Random(7), find_font(), 1)
        return truth

    def test_has_inline_label_cases(self, tmp_path) -> None:
        kinds = [t.get("difficulty") for t in self._truth(tmp_path)]
        assert kinds.count("inline_label") >= 2

    def test_has_unlabeled_value_case(self, tmp_path) -> None:
        kinds = [t.get("difficulty") for t in self._truth(tmp_path)]
        assert "unlabeled_value" in kinds

    def test_keeps_ocr_failure_cases(self, tmp_path) -> None:
        kinds = [t.get("difficulty") for t in self._truth(tmp_path)]
        for kind in ("handwriting", "stamp_overlap"):
            assert kind in kinds

    def test_signature_is_drawn_but_is_not_ground_truth(self, tmp_path) -> None:
        """서명·인영은 탐지 대상 10종에서 빠졌다.

        그래도 페이지에는 그린다 — 정답이 없어야 "읽을 글자가 없는 것을 억지로
        이름으로 보고했다" 를 과검으로 셀 수 있다. 정답에 다시 들어오면
        그 측정이 조용히 사라진다.
        """
        kinds = [t.get("difficulty") for t in self._truth(tmp_path)]
        assert "signature" not in kinds

    def test_inline_label_text_excludes_the_label(self, tmp_path) -> None:
        """text 는 개인정보 값만 담는다 (bbox 는 라벨을 포함한 박스 전체)."""
        for t in self._truth(tmp_path):
            if t.get("difficulty") == "inline_label" and t["text"]:
                assert "담당자" not in str(t["text"])
                assert "신청인" not in str(t["text"])
                assert "전화" not in str(t["text"])

    def test_every_truth_entry_has_a_known_label(self, tmp_path) -> None:
        from pii_pipeline.schema import PII_LABELS

        for t in self._truth(tmp_path):
            assert t["type"] in PII_LABELS
