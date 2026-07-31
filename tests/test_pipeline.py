"""파이프라인 통합 테스트.

OCR 엔진과 vLLM 서버 없이 오케스트레이션 배선을 검증한다.
전처리/OCR/LLM 을 가짜 구현으로 갈아끼우고 ③→④→④'→⑤ 흐름이 맞물리는지 본다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline import pipeline as pipeline_mod
from pii_pipeline.ocr.layout import assign_reading_order
from pii_pipeline.pipeline import PiiPipeline, PipelineConfig
from pii_pipeline.preprocess import PreprocessResult
from pii_pipeline.schema import OcrBox, OcrStatus, Source

PAGE_W, PAGE_H = 1748, 2480


def fake_boxes() -> list[OcrBox]:
    """서식 한 장 분량의 가짜 OCR 결과.

    행 구성:
      0: 성명           | 홍길동
      1: 주민등록번호   | 901231-1234563   (규칙 레이어가 확정)
      2: 주소           | 서울특별시 강남구 | 테헤란로 123
      3: (빈칸)         | ○○빌딩 5층
      4: 보증인 성명    | ???  <OCR_FAILED>  (손글씨 → pass2 가 회수)
    """
    rows: list[list[tuple[str, OcrStatus]]] = [
        [("성명", OcrStatus.OK), ("홍길동", OcrStatus.OK)],
        [("주민등록번호", OcrStatus.OK), ("901231-1234563", OcrStatus.OK)],
        [("주소", OcrStatus.OK), ("서울특별시 강남구", OcrStatus.OK),
         ("테헤란로 123", OcrStatus.OK)],
        [("", OcrStatus.OK), ("○○빌딩 5층", OcrStatus.OK)],
        [("보증인 성명", OcrStatus.OK), ("", OcrStatus.FAILED)],
    ]
    boxes: list[OcrBox] = []
    for r, row in enumerate(rows):
        for c, (text, status) in enumerate(row):
            if not text and status is OcrStatus.OK:
                continue  # 빈칸은 박스가 없다
            x1 = 200 + c * 420
            y1 = 300 + r * 90
            boxes.append(
                OcrBox(
                    index=-1,
                    bbox=(x1, y1, x1 + 400, y1 + 50),
                    text=text,
                    status=status,
                    rec_conf=0.2 if status is OcrStatus.FAILED else 0.95,
                )
            )
    return assign_reading_order(boxes)


class FakeOcr:
    def __init__(self, boxes: list[OcrBox]) -> None:
        self._boxes = boxes
        self.calls = 0

    def run(self, image: Any) -> list[OcrBox]:
        self.calls += 1
        return self._boxes


class FakeLlm:
    """pass1/pass2 응답을 미리 정해두는 가짜 클라이언트."""

    def __init__(self, pass1: dict[str, Any], pass2: dict[str, Any]) -> None:
        self._responses = [pass1, pass2]
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], image: Any | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append({"system": system, "user": user, "has_image": image is not None})
        payload = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return payload, {"raw": "", "attempt": 0, "finish_reason": "stop"}


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch):
    """전처리/OCR/LLM 을 가짜로 교체한 파이프라인 팩토리를 돌려준다."""
    boxes = fake_boxes()

    def fake_preprocess(path: str, **kwargs: Any) -> PreprocessResult:
        return PreprocessResult(
            image=object(), width=PAGE_W, height=PAGE_H, applied=["fake"]
        )

    monkeypatch.setattr(pipeline_mod, "preprocess", fake_preprocess)

    def build(pass1: dict[str, Any], pass2: dict[str, Any], **cfg: Any):
        pipe = PiiPipeline(PipelineConfig(**cfg))
        pipe.ocr = FakeOcr(boxes)          # type: ignore[assignment]
        pipe.llm = FakeLlm(pass1, pass2)   # type: ignore[assignment]
        return pipe, boxes

    return build


# 박스 인덱스: 0 성명 / 1 홍길동 / 2 주민등록번호 / 3 901231-... /
#              4 주소 / 5 서울특별시 강남구 / 6 테헤란로 123 /
#              7 ○○빌딩 5층 / 8 보증인 성명 / 9 <OCR_FAILED>
PASS1_OK = {
    "regions": [
        {"idx": [1], "type": "NAME", "conf": 0.97},
        {"idx": [5, 6, 7], "type": "ADDRESS", "conf": 0.94},
    ]
}
PASS2_OK = {
    "missed": [
        {"idx": [9], "type": "NAME", "conf": 0.85, "reason": "손글씨 보증인 성명"}
    ]
}


class TestBoxFixture:
    def test_indices_match_expectation(self, wired) -> None:
        _, boxes = wired(PASS1_OK, PASS2_OK)
        assert [b.text for b in boxes][:4] == ["성명", "홍길동", "주민등록번호", "901231-1234563"]
        assert boxes[9].status is OcrStatus.FAILED


class TestFullRun:
    def test_all_three_layers_contribute(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")

        by_source = result.stats()["by_source"]
        assert by_source["rule"] == 1        # 주민등록번호
        assert by_source["llm_pass1"] == 2   # 성명 + 주소
        assert by_source["vlm_pass2"] == 1   # 손글씨 성명

    def test_rule_layer_wins_and_llm_sees_confirmed_tag(self, wired) -> None:
        """규칙이 확정한 박스는 프롬프트에 CONFIRMED 로 표시되어야 한다."""
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        pipe.run("fake.png")
        pass1_user = pipe.llm.calls[0]["user"]  # type: ignore[attr-defined]
        assert "<CONFIRMED:RRN>" in pass1_user
        assert "901231-1234563" in pass1_user  # 지우지 않고 남긴다

    def test_pass2_receives_image_and_pass1_does_not(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        pipe.run("fake.png")
        calls = pipe.llm.calls  # type: ignore[attr-defined]
        assert calls[0]["has_image"] is False
        assert calls[1]["has_image"] is True

    def test_pass2_prompt_lists_already_detected_indices(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        pipe.run("fake.png")
        pass2_user = pipe.llm.calls[1]["user"]  # type: ignore[attr-defined]
        assert "이미 탐지된 박스 번호" in pass2_user
        assert "<OCR_FAILED>" in pass2_user

    def test_address_is_merged_into_one_region(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")
        addr = [r for r in result.regions if r.type == "ADDRESS"]
        assert len(addr) == 1
        assert addr[0].member_index == [5, 6, 7]
        assert addr[0].text == "서울특별시 강남구 테헤란로 123 ○○빌딩 5층"

    def test_ocr_failed_box_recovered_with_precise_coords(self, wired) -> None:
        pipe, boxes = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")
        recovered = [r for r in result.regions if r.source is Source.VLM_PASS2]
        assert len(recovered) == 1
        assert recovered[0].coarse is False           # OCR 좌표를 썼다
        assert recovered[0].needs_review is True      # 그래도 검토 대상
        assert recovered[0].member_boxes == [boxes[9].bbox]

    def test_ids_are_assigned_and_unique(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")
        ids = [r.id for r in result.regions]
        assert all(ids)
        assert len(ids) == len(set(ids))

    def test_timings_recorded_for_every_stage(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")
        for key in ("preprocess", "ocr", "rules", "llm_pass1", "llm_pass2", "merge", "total"):
            assert key in result.timings

    def test_run_is_deterministic(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK)
        first = pipe.run("fake.png")
        pipe2, _ = wired(PASS1_OK, PASS2_OK)
        second = pipe2.run("fake.png")
        assert [(r.id, r.type, r.bbox) for r in first.regions] == [
            (r.id, r.type, r.bbox) for r in second.regions
        ]


class TestPass2Disabled:
    def test_no_second_call_and_handwriting_is_missed(self, wired) -> None:
        """텍스트 pass 만으로는 OCR 실패 박스를 구조적으로 회수할 수 없다."""
        pipe, _ = wired(PASS1_OK, PASS2_OK, enable_pass2=False)
        result = pipe.run("fake.png")
        assert len(pipe.llm.calls) == 1  # type: ignore[attr-defined]
        assert "llm_pass2" not in result.timings
        assert all(r.source is not Source.VLM_PASS2 for r in result.regions)


class TestPass2Blind:
    def test_blind_prompt_hides_first_pass_results(self, wired) -> None:
        pipe, _ = wired(PASS1_OK, PASS2_OK, pass2_blind=True)
        pipe.run("fake.png")
        pass2_user = pipe.llm.calls[1]["user"]  # type: ignore[attr-defined]
        assert "이미 탐지된 박스 번호" not in pass2_user


class TestFailureHandling:
    def test_llm_error_is_recorded_but_rules_survive(self, wired) -> None:
        """LLM 이 죽어도 규칙 레이어 결과는 남아야 한다."""
        pipe, _ = wired(PASS1_OK, PASS2_OK)

        def failing(system, user, schema, image=None):
            return {}, {"error": "connection refused"}

        pipe.llm.complete_json = failing  # type: ignore[assignment]
        result = pipe.run("fake.png")

        assert any("pass1 실패" in w for w in result.warnings)
        assert any("pass2 실패" in w for w in result.warnings)
        assert [r.type for r in result.regions] == ["RRN"]

    def test_hallucinated_index_is_dropped_with_warning(self, wired) -> None:
        pipe, _ = wired(
            {"regions": [{"idx": [999], "type": "NAME", "conf": 0.9}]},
            {"missed": []},
        )
        result = pipe.run("fake.png")
        assert any("999" in w for w in result.warnings)
        assert all(r.type != "NAME" for r in result.regions)

    def test_no_ocr_boxes_returns_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_preprocess(path: str, **kwargs: Any) -> PreprocessResult:
            return PreprocessResult(image=object(), width=PAGE_W, height=PAGE_H)

        monkeypatch.setattr(pipeline_mod, "preprocess", fake_preprocess)
        pipe = PiiPipeline(PipelineConfig())
        pipe.ocr = FakeOcr([])  # type: ignore[assignment]
        pipe.llm = FakeLlm({}, {})  # type: ignore[assignment]

        result = pipe.run("fake.png")
        assert result.regions == []
        assert any("검출되지 않았" in w for w in result.warnings)
        assert pipe.llm.calls == []  # type: ignore[attr-defined]

    def test_batch_continues_after_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        boxes = fake_boxes()
        calls: list[str] = []

        def fake_preprocess(path: str, **kwargs: Any) -> PreprocessResult:
            calls.append(path)
            if path == "bad.png":
                raise RuntimeError("이미지를 읽을 수 없습니다")
            return PreprocessResult(image=object(), width=PAGE_W, height=PAGE_H)

        monkeypatch.setattr(pipeline_mod, "preprocess", fake_preprocess)
        pipe = PiiPipeline(PipelineConfig(enable_pass2=False))
        pipe.ocr = FakeOcr(boxes)  # type: ignore[assignment]
        pipe.llm = FakeLlm(PASS1_OK, PASS2_OK)  # type: ignore[assignment]

        results = pipe.run_batch(["ok1.png", "bad.png", "ok2.png"])
        assert len(results) == 3
        assert calls == ["ok1.png", "bad.png", "ok2.png"]
        assert results[1].regions == []
        assert any("처리 실패" in w for w in results[1].warnings)
        assert results[2].regions  # 실패 후에도 계속 처리된다


class TestOutputContract:
    def test_json_roundtrip(self, wired) -> None:
        import json

        pipe, _ = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")
        parsed = json.loads(result.to_json())

        assert parsed["page"] == {"width": PAGE_W, "height": PAGE_H}
        assert parsed["regions"]
        for region in parsed["regions"]:
            assert set(region) >= {
                "id", "type", "bbox", "source", "confidence",
                "member_boxes", "needs_review", "coarse",
            }
            assert len(region["bbox"]) == 4

    def test_include_ocr_adds_debug_payload(self, wired) -> None:
        import json

        pipe, _ = wired(PASS1_OK, PASS2_OK)
        result = pipe.run("fake.png")
        parsed = json.loads(result.to_json(include_ocr=True))
        assert "ocr_boxes" in parsed
        assert "raw_llm" in parsed
        assert len(parsed["ocr_boxes"]) == 10
