"""라벨 스키마 및 출력 계약 고정 테스트.

라벨을 늘리거나 줄일 때 프롬프트 형식표와 guided-decoding 스키마가 함께
움직이는지 확인한다.
"""

from __future__ import annotations

from pii_pipeline.llm.prompts import SYSTEM_VLM
from pii_pipeline.schema import (
    PII_LABELS,
    TEXTLESS_LABELS,
    VLM_SCHEMA,
    Agreement,
    OcrStatus,
    PageResult,
    PiiRegion,
    Source,
    VlmFinding,
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


def region(**kw) -> PiiRegion:
    base = dict(
        id="r001", type="NAME", bbox=(10, 10, 100, 40),
        source=Source.OCR_REFINED, confidence=0.9,
    )
    base.update(kw)
    return PiiRegion(**base)  # type: ignore[arg-type]


class TestLabelSet:
    def test_exact_labels_and_order(self) -> None:
        assert PII_LABELS == EXPECTED_LABELS

    def test_no_duplicates(self) -> None:
        assert len(set(PII_LABELS)) == len(PII_LABELS)

    def test_textless_is_subset(self) -> None:
        assert set(PII_LABELS) >= TEXTLESS_LABELS

    def test_prompt_lists_every_label(self) -> None:
        """모델이 쓸 수 없는 라벨이 스키마에만 있으면 guided decoding 이 막는다."""
        for label in PII_LABELS:
            assert label in SYSTEM_VLM, label


class TestVlmSchema:
    def test_enum_matches_labels(self) -> None:
        item = VLM_SCHEMA["properties"]["findings"]["items"]
        assert item["properties"]["type"]["enum"] == list(PII_LABELS)

    def test_requires_text_before_bbox(self) -> None:
        """생성 순서가 곧 조건화 순서다. 값을 먼저 확정해야 한다."""
        keys = list(VLM_SCHEMA["properties"]["findings"]["items"]["properties"])
        assert keys.index("text") < keys.index("bbox_2d")
        assert keys.index("type") < keys.index("bbox_2d")

    def test_all_fields_required(self) -> None:
        item = VLM_SCHEMA["properties"]["findings"]["items"]
        assert set(item["required"]) == {"text", "type", "field", "bbox_2d", "conf"}

    def test_bbox_is_four_per_mille_integers(self) -> None:
        """Qwen-VL 의 native grounding 형식이다 — 0~1000 정수, 키는 ``bbox_2d``.

        우리 편의대로 0.0~1.0 소수와 ``bbox_norm`` 을 요구하던 때가 있었는데,
        그건 어느 Qwen 버전의 native 형식도 아니다. grounding 은 분포에 가장
        민감한 과제라 형식을 바꾸는 대가가 좌표 정확도로 나온다.
        """
        bbox = VLM_SCHEMA["properties"]["findings"]["items"]["properties"]["bbox_2d"]
        assert bbox["minItems"] == bbox["maxItems"] == 4
        assert bbox["items"]["type"] == "integer"
        assert bbox["items"]["minimum"] == 0
        assert bbox["items"]["maximum"] == 1000

    def test_closed_object(self) -> None:
        assert VLM_SCHEMA["additionalProperties"] is False
        assert VLM_SCHEMA["properties"]["findings"]["items"]["additionalProperties"] is False


class TestSourceEnum:
    def test_only_two_coordinate_paths(self) -> None:
        """색과 판단이 두 갈래로 끝나야 검수자가 결정을 빨리 내린다."""
        assert {s.value for s in Source} == {"ocr_refined", "vlm_coarse"}


class TestVlmFinding:
    def test_to_dict_rounds_bbox(self) -> None:
        f = VlmFinding(text="홍길동", type="NAME", bbox_norm=(0.123456, 0.2, 0.3, 0.4))
        assert f.to_dict()["bbox_norm"] == [0.1235, 0.2, 0.3, 0.4]


class TestPiiRegionSerialization:
    def test_enums_become_strings(self) -> None:
        d = region(ocr_status=OcrStatus.LOW_CONF, agreement=Agreement.NONE).to_dict()
        assert d["source"] == "ocr_refined"
        assert d["ocr_status"] == "low_conf"
        assert d["agreement"] == "none"

    def test_bboxes_become_lists(self) -> None:
        d = region(member_boxes=[(1, 2, 3, 4)]).to_dict()
        assert d["bbox"] == [10, 10, 100, 40]
        assert d["member_boxes"] == [[1, 2, 3, 4]]

    def test_carries_both_engines_text(self) -> None:
        """어느 엔진이 무엇을 읽었는지가 남아야 진단이 된다."""
        d = region(text="홍길둥", vlm_text="홍길동").to_dict()
        assert (d["text"], d["vlm_text"]) == ("홍길둥", "홍길동")


class TestPageResultStats:
    def _page(self) -> PageResult:
        return PageResult(
            image_path="x.png", width=100, height=100,
            findings=[
                VlmFinding(text="a", type="NAME"),
                VlmFinding(text="b", type="RRN"),
                VlmFinding(text="c", type="SIGNATURE"),
            ],
            regions=[
                region(type="NAME", agreement=Agreement.EXACT),
                region(type="RRN", agreement=Agreement.EXACT, checksum="failed"),
                region(
                    type="SIGNATURE", source=Source.VLM_COARSE, coarse=True,
                    agreement=Agreement.NONE, needs_review=True,
                ),
            ],
        )

    def test_three_headline_metrics(self) -> None:
        stats = self._page().stats()
        assert stats["n_findings"] == 3
        assert stats["localized_rate"] == round(2 / 3, 3)
        assert stats["n_disagreement"] == 1
        assert stats["n_checksum_failed"] == 1

    def test_localized_rate_is_zero_when_empty(self) -> None:
        empty = PageResult(image_path="x.png", width=1, height=1)
        assert empty.stats()["localized_rate"] == 0.0

    def test_findings_only_in_debug_json(self) -> None:
        """기본 JSON 은 다운스트림용이라 진단 데이터를 싣지 않는다."""
        page = self._page()
        assert "findings" not in page.to_dict()
        assert "findings" in page.to_dict(include_ocr=True)

    def test_json_is_utf8_readable(self) -> None:
        page = self._page()
        page.regions[0].text = "홍길동"
        assert "홍길동" in page.to_json()
