"""파이프라인 통합 테스트.

OCR 엔진과 vLLM 서버 없이 오케스트레이션 배선을 검증한다.
전처리/VLM/OCR 을 가짜 구현으로 갈아끼우고 ①→②→③→④ 가 맞물리는지 본다.

특히 **무음 실패가 없는지**를 본다. 이 파이프라인에서 가장 위험한 결과는
"개인정보 없음" 처럼 보이는 실패다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline import pipeline as pipeline_mod
from pii_pipeline.detect import DetectConfig
from pii_pipeline.llm.client import LlmConfig
from pii_pipeline.locate import LocateConfig
from pii_pipeline.pipeline import PiiPipeline, PipelineConfig
from pii_pipeline.preprocess import PreprocessResult
from pii_pipeline.schema import Agreement, OcrBox, OcrStatus, Source

np = pytest.importorskip("numpy", reason="numpy 미설치 환경에서는 건너뛴다")

PAGE_W, PAGE_H = 1000, 2000

#: 체크섬을 통과하는 가짜 주민등록번호.
VALID_RRN = "901231-1234563"


def blank_page(w: int = PAGE_W, h: int = PAGE_H) -> Any:
    return np.zeros((h, w, 3), dtype=np.uint8)


def vlm_item(text: str, label: str, bbox, conf: float = 0.9, field: str = "") -> dict:
    return {
        "text": text, "type": label, "field": field,
        "bbox_2d": list(bbox), "conf": conf,
    }


class FakeClient:
    """타일 호출마다 payload 를 순서대로 돌려준다."""

    def __init__(self, payloads: list[dict], metas: list[dict] | None = None) -> None:
        self.payloads = payloads
        self.metas = metas
        self.calls = 0

    client = None  # detect() 의 지연 초기화 프라이밍 대상
    #: detect() 가 '모델이 본 크기' 를 계산할 때 image_max_side 를 읽는다.
    config = LlmConfig()

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        image: Any = None,
        temperature: float | None = None,
    ):
        n = self.calls
        self.calls += 1
        payload = self.payloads[n] if n < len(self.payloads) else {"findings": []}
        meta = (self.metas[n] if self.metas and n < len(self.metas) else {}) or {}
        return payload, dict(meta)


class FakeOcr:
    """크롭 순서대로 박스를 돌려준다. 좌표는 크롭 기준이다."""

    def __init__(self, results: list[list[OcrBox]]) -> None:
        self.results = results
        self.calls = 0

    def run_many(self, images: list[Any]) -> list[list[OcrBox]]:
        out = []
        for _ in images:
            out.append(self.results[self.calls] if self.calls < len(self.results) else [])
            self.calls += 1
        return out


def box(text: str, x1=5, y1=5, x2=200, y2=45, status=OcrStatus.OK) -> OcrBox:
    return OcrBox(index=-1, bbox=(x1, y1, x2, y2), text=text, status=status, rec_conf=0.95)


@pytest.fixture
def no_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """전처리를 항등 함수로 만든다 (opencv 의존 제거)."""

    def fake(
        img: Any,
        target_long_side: Any = None,
        deskew: bool = True,
        align: int = 32,
        canvas: Any = None,
    ) -> PreprocessResult:
        h, w = img.shape[:2]
        return PreprocessResult(image=img, width=w, height=h, applied=[])

    def fake_path(
        path: str,
        target_long_side: Any = None,
        deskew: bool = True,
        align: int = 32,
        canvas: Any = None,
    ):
        return fake(blank_page())

    monkeypatch.setattr(pipeline_mod, "preprocess_array", fake)
    monkeypatch.setattr(pipeline_mod, "preprocess", fake_path)


def build(
    payloads: list[dict],
    ocr_results: list[list[OcrBox]],
    metas: list[dict] | None = None,
    **cfg_kw: Any,
) -> PiiPipeline:
    cfg_kw.setdefault("detect", DetectConfig(tiles=1, workers=1))
    cfg_kw.setdefault("locate", LocateConfig(upscale=1.0))
    config = PipelineConfig(**cfg_kw)
    pipe = PiiPipeline(config)
    pipe.llm = FakeClient(payloads, metas)  # type: ignore[assignment]
    pipe.ocr = FakeOcr(ocr_results)  # type: ignore[assignment]
    return pipe


class TestHappyPath:
    def test_end_to_end(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [
                vlm_item("홍길동", "NAME", (0.1, 0.1, 0.3, 0.14), field="성명"),
                vlm_item(VALID_RRN, "RRN", (0.1, 0.2, 0.5, 0.24)),
            ]}],
            [[box("홍길동")], [box(VALID_RRN)]],
        )
        result = pipe.run("x.png", image=blank_page())

        assert len(result.findings) == 2
        assert len(result.regions) == 2
        assert {r.type for r in result.regions} == {"NAME", "RRN"}
        assert all(r.source is Source.OCR_REFINED for r in result.regions)

    def test_checksum_verified_region_is_final(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [vlm_item(VALID_RRN, "RRN", (0.1, 0.2, 0.5, 0.24))]}],
            [[box(VALID_RRN)]],
        )
        region = pipe.run("x.png", image=blank_page()).regions[0]
        assert region.verified is True
        assert region.confidence == 1.0
        assert region.needs_review is False
        assert region.agreement is Agreement.EXACT

    def test_ids_are_assigned(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [
                vlm_item("홍길동", "NAME", (0.1, 0.5, 0.3, 0.54)),
                vlm_item("김철수", "NAME", (0.1, 0.1, 0.3, 0.14)),
            ]}],
            [[box("홍길동")], [box("김철수")]],
        )
        ids = [r.id for r in pipe.run("x.png", image=blank_page()).regions]
        assert ids == ["r001", "r002"]

    def test_timings_cover_every_stage(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [vlm_item("홍길동", "NAME", (0.1, 0.1, 0.3, 0.14))]}],
            [[box("홍길동")]],
        )
        timings = pipe.run("x.png", image=blank_page()).timings
        assert set(timings) == {"preprocess", "detect", "locate", "verify", "total"}
        assert timings["total"] > 0

    def test_page_dimensions_come_from_preprocess(self, no_preprocess: None) -> None:
        pipe = build([{"findings": []}], [])
        result = pipe.run("x.png", image=blank_page(640, 480))
        assert (result.width, result.height) == (640, 480)


class TestOldBugRegressions:
    """이전 구조에서 스크린샷으로 관측된 오류들."""

    def test_rrn_mislabelled_as_license_is_corrected(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [vlm_item(VALID_RRN, "DRIVER_LICENSE", (0.1, 0.2, 0.5, 0.24))]}],
            [[box(VALID_RRN)]],
        )
        region = pipe.run("x.png", image=blank_page()).regions[0]
        assert region.type == "RRN"

    def test_header_cell_is_not_reported_as_a_name(self, no_preprocess: None) -> None:
        """VLM 이 인쇄된 헤더를 값으로 보고한 경우 — 판단 자체가 틀렸으므로 버린다."""
        pipe = build(
            [{"findings": [vlm_item("성명", "NAME", (0.05, 0.1, 0.15, 0.14))]}],
            [[box("성 명")]],
        )
        result = pipe.run("x.png", image=blank_page())
        assert result.regions == []
        assert any("항목명만" in w for w in result.warnings)

    def test_geometry_landing_on_a_header_is_kept_not_dropped(
        self, no_preprocess: None
    ) -> None:
        """미탐 확정을 막는다.

        VLM 은 실제 이름을 봤는데 박스 선택이 옆 헤더 셀에 떨어진 상황이다.
        여기서 버리면 그 이름은 마스킹되지 않은 채 남는다. 헤더를 덧칠하는
        손해가 이름을 놓치는 손해보다 작다.
        """
        pipe = build(
            [{"findings": [vlm_item("홍길동", "NAME", (0.1, 0.1, 0.3, 0.14))]}],
            [[box("성 명")], [box("성 명")]],
        )
        result = pipe.run("x.png", image=blank_page())
        assert len(result.regions) == 1
        r = result.regions[0]
        assert r.vlm_text == "홍길동"
        assert r.needs_review is True
        assert "박스 선택 오류 의심" in (r.reason or "")

    def test_a_missed_value_is_never_silently_dropped(self, no_preprocess: None) -> None:
        """좌표를 못 잡아도 개인정보는 결과에 남아야 한다."""
        pipe = build(
            [{"findings": [vlm_item("901112-2846261", "RRN", (0.1, 0.2, 0.5, 0.24))]}],
            [[], []],  # 1차·2차 배치 모두 아무것도 못 읽음
        )
        result = pipe.run("x.png", image=blank_page())
        assert len(result.regions) == 1
        region = result.regions[0]
        assert region.source is Source.VLM_COARSE
        assert region.needs_review is True
        assert region.vlm_text == "901112-2846261"


class TestSilentFailureGuards:
    def test_empty_result_from_a_failed_call_is_labelled_as_such(
        self, no_preprocess: None
    ) -> None:
        """'개인정보 없음' 과 '호출이 죽었음' 은 반드시 구분되어야 한다."""
        pipe = build([{}], [], metas=[{"error": "connection refused"}])
        result = pipe.run("x.png", image=blank_page())
        assert result.regions == []
        assert any("개인정보 없음이 아님" in w for w in result.warnings)

    def test_genuinely_empty_page_says_so(self, no_preprocess: None) -> None:
        pipe = build([{"findings": []}], [])
        result = pipe.run("x.png", image=blank_page())
        assert any("찾지 못했습니다" in w for w in result.warnings)
        assert not any("개인정보 없음이 아님" in w for w in result.warnings)

    def test_coarse_count_is_warned(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [vlm_item("홍길동", "NAME", (0.1, 0.1, 0.3, 0.14))]}],
            [[], []],
        )
        result = pipe.run("x.png", image=blank_page())
        assert any("좌표 확정 실패" in w for w in result.warnings)

    def test_disagreement_is_recorded_not_hidden(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [vlm_item("901112-2846261", "RRN", (0.1, 0.2, 0.5, 0.24))]}],
            [[box("9O1112-284626")], [box("9O1112-284626")]],
        )
        region = pipe.run("x.png", image=blank_page()).regions[0]
        assert region.needs_review is True
        assert "9O1112-284626" in (region.reason or "")


class TestDiagnostics:
    def test_findings_survive_for_recall_measurement(self, no_preprocess: None) -> None:
        """findings 가 recall 의 분모다. finalize 가 버린 건도 여기 남는다.

        주의: ``detect._dedup`` 이 findings 를 읽기 순서로 재정렬하므로,
        가짜 OCR 결과의 순서도 **읽기 순서**에 맞춰야 한다.
        """
        pipe = build(
            [{"findings": [
                vlm_item("성명", "NAME", (0.05, 0.1, 0.25, 0.14)),   # 위 — finalize 가 버린다
                vlm_item("홍길동", "NAME", (0.1, 0.5, 0.3, 0.54)),   # 아래
            ]}],
            [[box("성 명")], [box("홍길동")]],
        )
        result = pipe.run("x.png", image=blank_page())
        assert len(result.findings) == 2
        assert [r.vlm_text for r in result.regions] == ["홍길동"]

    def test_localized_rate_reflects_the_geometry_stage(self, no_preprocess: None) -> None:
        """두 번째는 크롭에 박스가 아예 없어 vlm_coarse 로 떨어진다."""
        pipe = build(
            [{"findings": [
                vlm_item("홍길동", "NAME", (0.1, 0.1, 0.3, 0.14)),
                vlm_item("김철수", "NAME", (0.1, 0.5, 0.3, 0.54)),
            ]}],
            [[box("홍길동")], [], []],
        )
        assert pipe.run("x.png", image=blank_page()).stats()["localized_rate"] == 0.5

    def test_raw_llm_keeps_every_tile_meta(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": []}, {"findings": []}, {"findings": []}],
            [],
            detect=DetectConfig(tiles=3, workers=1),
        )
        result = pipe.run("x.png", image=blank_page())
        assert len(result.raw_llm["vlm"]) == 3

    def test_crop_ocr_boxes_are_exposed(self, no_preprocess: None) -> None:
        pipe = build(
            [{"findings": [vlm_item("홍길동", "NAME", (0.1, 0.1, 0.3, 0.14))]}],
            [[box("홍길동"), box("성명", 210, 5, 300, 45)]],
        )
        result = pipe.run("x.png", image=blank_page())
        assert len(result.ocr_boxes) == 2


class TestTiling:
    def test_one_call_per_tile(self, no_preprocess: None) -> None:
        pipe = build([{"findings": []}] * 4, [], detect=DetectConfig(tiles=4, workers=1))
        pipe.run("x.png", image=blank_page())
        assert pipe.llm.calls == 4  # type: ignore[attr-defined]

    def test_findings_from_later_tiles_land_lower_on_the_page(
        self, no_preprocess: None
    ) -> None:
        pipe = build(
            [{"findings": []}, {"findings": [vlm_item("홍길동", "NAME", (0.1, 0.0, 0.3, 0.2))]}],
            [[box("홍길동")]],
            detect=DetectConfig(tiles=2, overlap=0.0, workers=1),
        )
        result = pipe.run("x.png", image=blank_page())
        # 정확히 0.5 가 아니다 — 조각 경계는 패치 격자(32px)로 스냅된다.
        # 2000/2 = 1000 은 32 의 배수가 아니라 896 으로 내려간다.
        assert result.findings[0].bbox_norm[1] >= 0.44


class TestImagePassthrough:
    def test_image_is_attached_for_overlay(self, no_preprocess: None) -> None:
        pipe = build([{"findings": []}], [])
        assert pipe.run("x.png", image=blank_page()).image is not None

    def test_release_clears_it(self, no_preprocess: None) -> None:
        pipe = build([{"findings": []}], [])
        result = pipe.run("x.png", image=blank_page())
        result.release_image()
        assert result.image is None


class TestBatch:
    def test_one_failure_does_not_stop_the_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = build([{"findings": []}], [])

        calls: list[str] = []

        def flaky(path: str, out_dir: Any = None, **kw: Any):
            calls.append(path)
            if path == "bad.png":
                raise RuntimeError("깨진 파일")
            from pii_pipeline.schema import PageResult

            return [PageResult(image_path=path, width=1, height=1)]

        monkeypatch.setattr(pipe, "run_any", flaky)
        results = pipe.run_batch(["a.png", "bad.png", "c.png"])
        assert calls == ["a.png", "bad.png", "c.png"]
        assert len(results) == 3
        assert any("처리 실패" in w for r in results for w in r.warnings)


# --------------------------------------------------------------------------
# 비동기 배치
# --------------------------------------------------------------------------


class FakeAsyncClient:
    """``complete_json_many`` 만 흉내낸다.

    **동시에 몇 개가 떠 있었는지 기록한다** — 상한이 실제로 걸리는지 보려면
    반환값만으로는 알 수 없다.
    """

    config = LlmConfig()

    def __init__(self, payloads: list[dict], concurrency: int = 8) -> None:
        self.payloads = payloads
        self.concurrency = concurrency
        self.calls = 0
        self.in_flight = 0
        self.peak = 0
        self.seen_tags: list[Any] = []

    async def complete_json_many(
        self, requests: list[Any], concurrency: int | None = None, on_done: Any = None
    ):
        import asyncio

        gate = asyncio.Semaphore(concurrency or self.concurrency)

        async def one(request: Any):
            async with gate:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                await asyncio.sleep(0)  # 다른 요청에 실행 기회를 준다
                n = self.calls
                self.calls += 1
                self.seen_tags.append(request.tag)
                payload = (
                    self.payloads[n] if n < len(self.payloads) else {"findings": []}
                )
                self.in_flight -= 1
                return payload, {}

        return list(await asyncio.gather(*(one(r) for r in requests)))

    async def aclose(self) -> None:
        return None


def build_async(
    payloads: list[dict],
    ocr_results: list[list[OcrBox]],
    **cfg_kw: Any,
) -> PiiPipeline:
    cfg_kw.setdefault("detect", DetectConfig(tiles=1, workers=1))
    cfg_kw.setdefault("locate", LocateConfig(upscale=1.0))
    pipe = PiiPipeline(PipelineConfig(**cfg_kw))
    pipe.async_llm = FakeAsyncClient(payloads)  # type: ignore[assignment]
    pipe.ocr = FakeOcr(ocr_results)  # type: ignore[assignment]
    return pipe


class TestRunBatchAsync:
    """배치 경로는 동기 경로와 **같은 결과**를 내야 한다.

    다른 것은 요청을 넣는 방식뿐이다. 결과가 갈리면 "async 로 돌렸더니 탐지가
    조금 다르다" 가 되는데, 그건 원인을 찾기 어려운 종류의 차이다.
    """

    def test_matches_the_sync_path(self, no_preprocess: None) -> None:
        import asyncio

        payloads = [{"findings": [vlm_item(VALID_RRN, "RRN", (100, 200, 300, 240))]}]
        ocr = [[box(VALID_RRN)]]

        sync = build(list(payloads), [list(r) for r in ocr]).run("a.png")
        batch = asyncio.run(
            build_async(list(payloads), [list(r) for r in ocr]).run_batch_async(["a.png"])
        )

        assert len(batch) == 1
        assert [r.type for r in batch[0].regions] == [r.type for r in sync.regions]
        assert [r.bbox for r in batch[0].regions] == [r.bbox for r in sync.regions]
        assert [r.text for r in batch[0].regions] == [r.text for r in sync.regions]

    def test_submits_every_page_in_one_go(self, no_preprocess: None) -> None:
        """묶음의 요청이 한 번에 들어가야 페이지 사이에서 GPU 가 놀지 않는다."""
        import asyncio

        pipe = build_async([{"findings": []}] * 4, [])
        pipe.config.batch.pages = 4
        results = asyncio.run(pipe.run_batch_async(["a.png", "b.png", "c.png", "d.png"]))

        assert len(results) == 4
        assert pipe.async_llm.calls == 4  # type: ignore[attr-defined]
        # 4장이 동시에 떠 있었다 — 순차라면 1 이 최대다
        assert pipe.async_llm.peak > 1  # type: ignore[attr-defined]

    def test_tags_identify_the_page(self, no_preprocess: None) -> None:
        """순서에만 의존하지 않도록 페이지 표식이 요청에 실려야 한다."""
        import asyncio

        pipe = build_async([{"findings": []}] * 3, [])
        pipe.config.batch.pages = 3
        asyncio.run(pipe.run_batch_async(["a.png", "b.png", "c.png"]))
        pages = [tag[0] for tag in pipe.async_llm.seen_tags]  # type: ignore[attr-defined]
        assert pages == [0, 1, 2]

    def test_windows_do_not_lose_pages(self, no_preprocess: None) -> None:
        """묶음 크기보다 페이지가 많으면 여러 묶음으로 나뉜다 — 하나도 빠지면 안 된다."""
        import asyncio

        pipe = build_async([{"findings": []}] * 5, [])
        pipe.config.batch.pages = 2
        results = asyncio.run(
            pipe.run_batch_async(["a.png", "b.png", "c.png", "d.png", "e.png"])
        )
        assert len(results) == 5
        assert pipe.async_llm.calls == 5  # type: ignore[attr-defined]

    def test_reports_progress(self, no_preprocess: None) -> None:
        import asyncio

        seen: list[tuple[int, int]] = []
        pipe = build_async([{"findings": []}] * 4, [])
        pipe.config.batch.pages = 2
        asyncio.run(
            pipe.run_batch_async(
                ["a.png", "b.png", "c.png", "d.png"],
                on_progress=lambda done, total: seen.append((done, total)),
            )
        )
        assert seen == [(2, 2), (4, 4)]

    def test_records_both_shared_and_per_page_detect_time(
        self, no_preprocess: None
    ) -> None:
        """페이지별 VLM 시간은 존재하지 않는다 — 그 사실이 숫자에 드러나야 한다."""
        import asyncio

        pipe = build_async([{"findings": []}] * 2, [])
        pipe.config.batch.pages = 2
        results = asyncio.run(pipe.run_batch_async(["a.png", "b.png"]))
        for result in results:
            assert result.timings["detect_window"] >= result.timings["detect"]
        assert results[0].timings["detect_window"] == results[1].timings["detect_window"]

    def test_a_failed_page_does_not_sink_the_batch(
        self, no_preprocess: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """한 장이 죽어도 나머지는 처리하고, 죽은 장도 자리를 지킨다."""
        import asyncio

        pipe = build_async([{"findings": []}] * 3, [])
        pipe.config.batch.pages = 3

        real = pipe.preprocess_page
        calls = {"n": 0}

        def flaky(path: str, **kw: Any):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("깨진 이미지")
            return real(path, **kw)

        monkeypatch.setattr(pipe, "preprocess_page", flaky)
        results = asyncio.run(pipe.run_batch_async(["a.png", "b.png", "c.png"]))

        assert len(results) == 3
        assert any("처리 실패" in w for r in results for w in r.warnings)

    def test_a_failed_page_keeps_its_place(
        self, no_preprocess: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """실패한 페이지가 앞으로 몰리면 입력과 결과를 짝지을 수 없다.

        성공한 것만 모아 처리하고 나중에 이어 붙이면 정확히 그렇게 된다.
        """
        import asyncio

        pipe = build_async([{"findings": []}] * 3, [])
        pipe.config.batch.pages = 3

        real = pipe.preprocess_page

        def flaky(path: str, **kw: Any):
            if path == "b.png":
                raise RuntimeError("깨진 이미지")
            return real(path, **kw)

        monkeypatch.setattr(pipe, "preprocess_page", flaky)
        results = asyncio.run(pipe.run_batch_async(["a.png", "b.png", "c.png"]))

        assert [r.image_path for r in results] == ["a.png", "b.png", "c.png"]
        assert results[1].width == 0  # 가운데가 실패한 자리다
        assert any("처리 실패" in w for w in results[1].warnings)
        assert not any("처리 실패" in w for w in results[0].warnings)
