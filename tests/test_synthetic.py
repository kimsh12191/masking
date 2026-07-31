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

from pii_pipeline.ocr.layout import assign_reading_order  # noqa: E402
from pii_pipeline.rules.checksums import (  # noqa: E402
    validate_biz_no,
    validate_foreign_id,
    validate_luhn,
    validate_rrn,
)
from pii_pipeline.rules.detectors import detect  # noqa: E402
from pii_pipeline.schema import OcrBox  # noqa: E402

SEEDS = list(range(30))


@pytest.mark.parametrize("seed", SEEDS)
class TestGeneratedNumbersAreValid:
    def test_rrn_has_13_digits_and_valid_checksum(self, seed: int) -> None:
        value = gen_rrn(random.Random(seed))
        assert len(value.replace("-", "")) == 13
        assert validate_rrn(value) is True

    def test_foreign_id_valid(self, seed: int) -> None:
        value = gen_rrn(random.Random(seed), foreign=True)
        assert validate_foreign_id(value) is True
        assert value.replace("-", "")[6] in "5678"

    def test_biz_no_valid(self, seed: int) -> None:
        assert validate_biz_no(gen_biz_no(random.Random(seed))) is True

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


class TestRuleLayerFindsGeneratedValues:
    """합성 값이 규칙 레이어에 실제로 잡히는지 (엔드투엔드 연결 확인)."""

    def _boxes(self, rows: list[list[str]]) -> list[OcrBox]:
        boxes: list[OcrBox] = []
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                x1, y1 = 100 + c * 400, 100 + r * 80
                boxes.append(
                    OcrBox(index=-1, bbox=(x1, y1, x1 + 380, y1 + 50), text=text)
                )
        return assign_reading_order(boxes)

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_rrn_detected_with_checksum_ok(self, seed: int) -> None:
        rng = random.Random(seed)
        boxes = self._boxes([["주민등록번호", gen_rrn(rng)]])
        hits = [h for h in detect(boxes) if h.label == "RRN"]
        assert len(hits) == 1
        assert hits[0].checksum_ok is True

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_biz_no_detected_with_checksum_ok(self, seed: int) -> None:
        rng = random.Random(seed)
        boxes = self._boxes([["사업자등록번호", gen_biz_no(rng)]])
        hits = [h for h in detect(boxes) if h.label == "BIZ_NO"]
        assert len(hits) == 1
        assert hits[0].checksum_ok is True

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_card_detected_with_checksum_ok(self, seed: int) -> None:
        rng = random.Random(seed)
        boxes = self._boxes([["결제카드번호", gen_card(rng)]])
        hits = [h for h in detect(boxes) if h.label == "CARD_NO"]
        assert len(hits) == 1
        assert hits[0].checksum_ok is True

    def test_account_detected_via_context_keyword(self) -> None:
        boxes = self._boxes([["입금계좌번호", f"하나은행 {gen_account(random.Random(3))}"]])
        assert any(h.label == "ACCOUNT_NO" for h in detect(boxes))

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_ip_detected(self, seed: int) -> None:
        boxes = self._boxes([["전자서명 접속IP", gen_ip(random.Random(seed))]])
        assert any(h.label == "IP" for h in detect(boxes))

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_passport_detected(self, seed: int) -> None:
        boxes = self._boxes([["여권번호", gen_passport(random.Random(seed))]])
        assert any(h.label == "PASSPORT" for h in detect(boxes))
