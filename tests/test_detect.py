"""① VLM 탐지 단계 테스트.

vLLM 서버 없이 배선을 검증한다. 특히 **타일 좌표 환산**이 맞는지 — 모델은
자기가 받은 조각 기준으로만 답하므로, 페이지 좌표로 되돌리는 것은 전부 코드
책임이다. 여기가 틀리면 크롭이 엉뚱한 곳을 잘라 모든 항목이 vlm_coarse 로
떨어지는데, 겉으로는 "OCR 이 못 읽었다" 처럼 보여 진단이 어렵다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline.detect import (
    DetectConfig,
    _dedup,
    _sane_bbox,
    _to_page,
    crop_norm,
    detect,
    infer_scale,  # noqa: E402
    match_key,
    smart_resize,
    tile_rects,
)
from pii_pipeline.llm.client import LlmConfig
from pii_pipeline.schema import VlmFinding

np = pytest.importorskip("numpy", reason="numpy 미설치 환경에서는 건너뛴다")

PAGE_W, PAGE_H = 1748, 2480


def blank_page(w: int = PAGE_W, h: int = PAGE_H) -> Any:
    return np.zeros((h, w, 3), dtype=np.uint8)


class FakeClient:
    """타일별로 미리 정해둔 payload 를 돌려주는 가짜 클라이언트."""

    def __init__(self, payloads: list[dict[str, Any]], metas: list[dict] | None = None) -> None:
        self.payloads = payloads
        self.metas = metas
        self.calls: list[dict[str, Any]] = []

    #: ``detect()`` 가 병렬 실행 전에 지연 초기화를 미리 건드린다.
    client = None
    #: ``detect()`` 가 '모델이 본 크기' 를 계산할 때 image_max_side 를 읽는다.
    config = LlmConfig()

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        image: Any = None,
        temperature: float | None = None,
    ):
        n = len(self.calls)
        self.calls.append(
            {"system": system, "user": user, "image": image, "temperature": temperature}
        )
        payload = self.payloads[n] if n < len(self.payloads) else {"findings": []}
        meta = (self.metas[n] if self.metas and n < len(self.metas) else {}) or {}
        return payload, dict(meta)


def item(text: str, label: str = "NAME", bbox=(0.1, 0.1, 0.3, 0.2), conf: float = 0.9) -> dict:
    """VLM 응답 항목 하나.

    ``bbox`` 는 편의상 0.0~1.0 으로 받지만 **payload 에는 0~1000 정수로 넣는다** —
    그게 Qwen-VL 의 native 형식이고 프롬프트가 요구하는 것이다. 테스트가
    0.0~1.0 소수를 보내면 프로덕션에서 실제로 오는 형식을 한 번도 검증하지
    않게 된다.
    """
    return {
        "text": text,
        "type": label,
        "field": "성명",
        "bbox_2d": [int(round(v * 1000)) for v in bbox],
        "conf": conf,
    }


def cfg(**kw: Any) -> DetectConfig:
    """테스트용 DetectConfig — ``workers=1`` 이 기본이다.

    ``FakeClient`` 가 **호출 순서**로 payload 를 고르기 때문에 병렬 실행에서는
    타일과 payload 의 짝이 어긋난다. 병렬 실행 자체는 ``TestParallel`` 에서
    순서에 의존하지 않는 fake 로 따로 검증한다.
    """
    kw.setdefault("workers", 1)
    return DetectConfig(**kw)


# --------------------------------------------------------------------------
# 타일 기하
# --------------------------------------------------------------------------


class TestTileRects:
    def test_single_tile_is_whole_page(self) -> None:
        assert tile_rects(PAGE_W, PAGE_H, 1, 0.08) == [(0.0, 0.0, 1.0, 1.0)]

    def test_zero_or_negative_collapses_to_one(self) -> None:
        assert tile_rects(PAGE_W, PAGE_H, 0, 0.1) == [(0.0, 0.0, 1.0, 1.0)]

    def test_portrait_splits_vertically(self) -> None:
        rects = tile_rects(1000, 2000, 2, 0.0)
        assert rects == [(0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 1.0)]

    def test_landscape_splits_horizontally(self) -> None:
        """긴 축을 자르면 조각이 정사각형에 가까워져 비전 인코더에 유리하다."""
        rects = tile_rects(2000, 1000, 2, 0.0)
        assert rects == [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)]

    def test_tiles_cover_the_whole_page(self) -> None:
        rects = tile_rects(PAGE_W, PAGE_H, 4, 0.08)
        assert rects[0][1] == 0.0
        assert rects[-1][3] == 1.0

    def test_neighbours_overlap(self) -> None:
        """겹침이 없으면 경계에 걸친 한 줄을 양쪽에서 반씩 잘라 둘 다 못 읽는다."""
        rects = tile_rects(PAGE_W, PAGE_H, 3, 0.1)
        assert rects[1][1] < rects[0][3]
        assert rects[2][1] < rects[1][3]

    def test_overlap_is_clamped(self) -> None:
        for rect in tile_rects(PAGE_W, PAGE_H, 3, 5.0):
            assert 0.0 <= rect[1] <= rect[3] <= 1.0

    def test_boundaries_snap_to_the_patch_grid(self) -> None:
        """조각 크기가 패치 격자의 배수여야 서버가 다시 리샘플하지 않는다.

        안 맞으면 1748x826 이 1760x832 로 보간되어 10px 급 한글이 뭉개진다.
        그건 곧 미탐이다. 페이지가 격자에 맞아 있어야 (preprocess 가 여백을
        붙여 맞춘다) 이 성질이 끝까지 성립한다.
        """
        w, h = 1760, 2464  # preprocess 가 내놓는 형태
        for rect in tile_rects(w, h, 3, 0.08, 32):
            y1, y2 = round(rect[1] * h), round(rect[3] * h)
            assert (y2 - y1) % 32 == 0, f"조각 높이 {y2 - y1} 가 32 의 배수가 아니다"
            assert y1 % 32 == 0

    def test_no_snapping_when_factor_is_one(self) -> None:
        """격자를 모르면 스냅하지 않는다 (이전 동작)."""
        rects = tile_rects(1000, 2000, 2, 0.0, 1)
        assert rects[1][1] == pytest.approx(0.5)

    def test_snapping_never_leaves_a_gap(self) -> None:
        """스냅 때문에 페이지 일부가 어느 조각에도 안 들어가면 그건 미탐이다."""
        w, h = 1760, 2464
        rects = tile_rects(w, h, 4, 0.08, 32)
        assert rects[0][1] == 0.0
        assert rects[-1][3] == 1.0
        for a, b in zip(rects[:-1], rects[1:], strict=True):
            assert b[1] <= a[3], "조각 사이에 빈 구간이 생겼다"


class TestCropNorm:
    def test_shape_matches_rect(self) -> None:
        crop = crop_norm(blank_page(1000, 2000), (0.0, 0.25, 1.0, 0.75))
        assert crop.shape[:2] == (1000, 1000)

    def test_never_returns_empty(self) -> None:
        crop = crop_norm(blank_page(100, 100), (0.5, 0.5, 0.5, 0.5))
        assert crop.size > 0


class TestToPage:
    def test_middle_tile_offsets_are_applied(self) -> None:
        # 두 번째 조각(y 0.5~1.0) 안의 상단 20% 는 페이지 기준 0.5~0.6
        out = _to_page((0.0, 0.0, 1.0, 0.2), (0.0, 0.5, 1.0, 1.0))
        assert out == pytest.approx((0.0, 0.5, 1.0, 0.6))

    def test_full_page_tile_is_identity(self) -> None:
        local = (0.2, 0.3, 0.4, 0.5)
        assert _to_page(local, (0.0, 0.0, 1.0, 1.0)) == pytest.approx(local)


class TestSaneBbox:
    def test_passes_clean_values(self) -> None:
        assert _sane_bbox([0.1, 0.2, 0.3, 0.4]) == pytest.approx((0.1, 0.2, 0.3, 0.4))

    def test_sorts_reversed_coordinates(self) -> None:
        """순서만 틀린 것을 버리면 탐지 하나를 통째로 잃는다."""
        assert _sane_bbox([0.3, 0.4, 0.1, 0.2]) == pytest.approx((0.1, 0.2, 0.3, 0.4))

    def test_clamps_out_of_range(self) -> None:
        x1, y1, x2, y2 = _sane_bbox([-0.5, 0.2, 1.7, 0.4])
        assert (x1, x2) == (0.0, 1.0)

    def test_degenerate_point_gets_minimum_area(self) -> None:
        x1, y1, x2, y2 = _sane_bbox([0.5, 0.5, 0.5, 0.5])
        assert x2 > x1 and y2 > y1

    def test_rejects_malformed(self) -> None:
        assert _sane_bbox(None) is None
        assert _sane_bbox([0.1, 0.2]) is None
        assert _sane_bbox("nope") is None
        assert _sane_bbox([0.1, 0.2, 0.3, "x"]) is None


# --------------------------------------------------------------------------
# 중복 제거
# --------------------------------------------------------------------------


class TestMatchKey:
    def test_folds_evasion_notation(self) -> None:
        assert match_key("공1공-1234-5678") == match_key("010 1234 5678")

    def test_ignores_separators(self) -> None:
        assert match_key("901231-1234563") == match_key("901231 1234563")


class TestDedup:
    def _f(self, text: str, bbox, conf: float = 0.9, label: str = "NAME") -> VlmFinding:
        return VlmFinding(text=text, type=label, bbox_norm=bbox, conf=conf)

    def test_same_value_overlapping_is_merged(self) -> None:
        out = _dedup([
            self._f("홍길동", (0.1, 0.5, 0.3, 0.55), conf=0.8),
            self._f("홍길동", (0.11, 0.5, 0.31, 0.55), conf=0.95),
        ])
        assert len(out) == 1
        assert out[0].conf == 0.95  # 확신도 높은 쪽이 남는다

    def test_same_value_elsewhere_is_kept(self) -> None:
        """같은 이름이 문서 여러 곳에 나오는 것은 정상이고 각각 마스킹해야 한다."""
        out = _dedup([
            self._f("홍길동", (0.1, 0.1, 0.3, 0.15)),
            self._f("홍길동", (0.1, 0.8, 0.3, 0.85)),
        ])
        assert len(out) == 2

    def test_different_type_same_spot_is_kept(self) -> None:
        """한 칸에 이름과 번호가 같이 있는 서식이 흔하다."""
        out = _dedup([
            self._f("홍길동", (0.1, 0.1, 0.5, 0.15), label="NAME"),
            self._f("홍길동", (0.1, 0.1, 0.5, 0.15), label="OTHER"),
        ])
        assert len(out) == 2

    def test_evasion_variant_is_treated_as_duplicate(self) -> None:
        out = _dedup([
            self._f("010-1234-5678", (0.1, 0.5, 0.4, 0.55), label="PHONE"),
            self._f("공1공-1234-5678", (0.1, 0.5, 0.4, 0.55), label="PHONE"),
        ])
        assert len(out) == 1

    def test_output_is_in_reading_order(self) -> None:
        out = _dedup([
            self._f("c", (0.1, 0.8, 0.2, 0.85)),
            self._f("a", (0.1, 0.1, 0.2, 0.15)),
            self._f("b", (0.6, 0.1, 0.7, 0.15)),
        ])
        assert [f.text for f in out] == ["a", "b", "c"]


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


class TestDetect:
    def test_calls_once_per_tile(self) -> None:
        client = FakeClient([{"findings": []}] * 3)
        detect(blank_page(), client, cfg(tiles=3))
        assert len(client.calls) == 3

    def test_system_prompt_is_identical_across_tiles(self) -> None:
        """이게 깨지면 prefix caching 이 안 걸려 타일 수만큼 비용이 곱해진다."""
        client = FakeClient([{"findings": []}] * 3)
        detect(blank_page(), client, cfg(tiles=3))
        systems = {c["system"] for c in client.calls}
        users = {c["user"] for c in client.calls}
        assert len(systems) == 1
        assert len(users) == 1

    def test_tile_local_coords_become_page_coords(self) -> None:
        """조각 안의 y=0 은 페이지의 y=0 이 아니다.

        기준값을 0.5 로 박아 두지 않는다. 조각 경계는 **패치 격자(32px)로
        스냅되므로** 정확히 절반이 아니다 (2480/2 = 1240 은 32 의 배수가 아니다).
        기대값을 ``tile_rects`` 에서 가져와야 환산 자체를 검증하게 된다.
        """
        client = FakeClient([
            {"findings": []},
            {"findings": [item("홍길동", bbox=(0.2, 0.0, 0.4, 1.0))]},
        ])
        rects = tile_rects(PAGE_W, PAGE_H, 2, 0.0, 32)
        findings, _ = detect(blank_page(), client, cfg(tiles=2, overlap=0.0))
        assert len(findings) == 1
        # 두 번째 조각의 상단이 곧 이 항목의 상단이다
        assert findings[0].bbox_norm[1] == pytest.approx(rects[1][1])
        assert findings[0].bbox_norm[3] == pytest.approx(1.0)
        # x 는 세로 분할이므로 그대로다
        assert findings[0].bbox_norm[0] == pytest.approx(0.2)

    def test_records_tile_number(self) -> None:
        client = FakeClient([{"findings": []}, {"findings": [item("홍길동")]}])
        findings, _ = detect(blank_page(), client, cfg(tiles=2))
        assert findings[0].tile == 1

    def test_sends_an_image_every_call(self) -> None:
        client = FakeClient([{"findings": []}] * 2)
        detect(blank_page(), client, cfg(tiles=2))
        assert all(c["image"] is not None for c in client.calls)

    def test_single_tile_sends_the_untouched_page(self) -> None:
        page = blank_page(400, 600)
        client = FakeClient([{"findings": []}])
        detect(page, client, cfg(tiles=1))
        assert client.calls[0]["image"] is page

    def test_one_failed_tile_does_not_lose_the_others(self) -> None:
        warnings: list[str] = []
        client = FakeClient(
            [{"findings": [item("홍길동")]}, {}, {"findings": [item("김철수")]}],
            metas=[{}, {"error": "timeout"}, {}],
        )
        findings, metas = detect(blank_page(), client, cfg(tiles=3), warnings)
        assert {f.text for f in findings} == {"홍길동", "김철수"}
        assert any("타일 1 실패" in w for w in warnings)
        assert len(metas) == 3

    def test_salvaged_response_raises_a_warning(self) -> None:
        """형식 이탈은 프롬프트가 예산을 넘겼다는 신호다. 조용히 넘기지 않는다."""
        warnings: list[str] = []
        client = FakeClient([{"findings": [item("홍길동")]}], metas=[{"salvaged": "Extra data"}])
        detect(blank_page(), client, cfg(tiles=1), warnings)
        assert any("형식 이탈" in w for w in warnings)

    def test_item_without_bbox_is_dropped_with_a_warning(self) -> None:
        warnings: list[str] = []
        client = FakeClient([
            {"findings": [{"text": "홍길동", "type": "NAME", "field": "", "conf": 0.9}]}
        ])
        findings, _ = detect(blank_page(), client, cfg(tiles=1), warnings)
        assert findings == []
        assert any("bbox_2d" in w for w in warnings)

    def test_item_without_type_is_skipped(self) -> None:
        client = FakeClient([{"findings": [{"text": "x", "bbox_2d": [0, 0, 1, 1]}]}])
        findings, _ = detect(blank_page(), client, cfg(tiles=1))
        assert findings == []

    def test_non_dict_items_are_skipped(self) -> None:
        client = FakeClient([{"findings": ["not a dict", None, item("홍길동")]}])
        findings, _ = detect(blank_page(), client, cfg(tiles=1))
        assert [f.text for f in findings] == ["홍길동"]

    def test_missing_findings_key_is_treated_as_empty(self) -> None:
        client = FakeClient([{}])
        findings, _ = detect(blank_page(), client, cfg(tiles=1))
        assert findings == []

    def test_empty_text_survives_for_signature(self) -> None:
        """서명·인영은 읽을 글자가 없다. 버리면 도장 영역을 놓친다."""
        client = FakeClient([
            {"findings": [item("", label="SIGNATURE", bbox=(0.6, 0.8, 0.8, 0.9))]}
        ])
        findings, _ = detect(blank_page(), client, cfg(tiles=1))
        assert len(findings) == 1
        assert findings[0].type == "SIGNATURE"

    def test_dedup_can_be_disabled(self) -> None:
        payload = {"findings": [item("홍길동", bbox=(0.1, 0.1, 0.3, 0.2))]}
        client = FakeClient([payload, payload])
        findings, _ = detect(
            blank_page(), client, cfg(tiles=2, overlap=0.5, dedup=False)
        )
        assert len(findings) == 2

    def test_hint_reaches_the_user_message(self) -> None:
        client = FakeClient([{"findings": []}])
        detect(blank_page(), client, cfg(tiles=1, hint="가족관계증명서다"))
        assert "가족관계증명서다" in client.calls[0]["user"]


# --------------------------------------------------------------------------
# 다중 샘플 합집합 — 학습 없이 미탐을 줄이는 손잡이
# --------------------------------------------------------------------------


class TestSamples:
    def test_default_is_one_call_per_tile_at_temperature_zero(self) -> None:
        """기본값은 결정론이어야 한다.

        ``samples==1`` 에서 온도를 건드리면 같은 문서를 두 번 돌렸을 때 결과가
        달라진다 — 감사 대응이 불가능해진다. 그래서 ``temperature=None`` 을
        넘겨 설정값(0)이 쓰이게 한다.
        """
        client = FakeClient([{"findings": []}] * 3)
        detect(blank_page(), client, cfg(tiles=3))
        assert len(client.calls) == 3
        assert all(c["temperature"] is None for c in client.calls)

    def test_samples_multiplies_calls_and_raises_temperature(self) -> None:
        client = FakeClient([{"findings": []}] * 6)
        detect(blank_page(), client, cfg(tiles=3, samples=2, sample_temperature=0.4))
        assert len(client.calls) == 6  # tiles x samples
        assert all(c["temperature"] == 0.4 for c in client.calls)

    def test_union_recovers_a_value_only_one_sample_found(self) -> None:
        """이게 이 기능의 존재 이유다.

        한 샘플이 놓친 항목을 다른 샘플이 잡으면 합집합에 남아야 한다. 미탐
        (개인정보 노출)이 과탐(멀쩡한 칸 덧칠)보다 훨씬 비싸므로 맞는 거래다.
        """
        client = FakeClient([
            {"findings": [item("홍길동", bbox=(0.1, 0.1, 0.3, 0.2))]},
            {"findings": [
                item("홍길동", bbox=(0.1, 0.1, 0.3, 0.2)),
                item("901231-1234567", label="RRN", bbox=(0.5, 0.1, 0.9, 0.2)),
            ]},
        ])
        findings, _ = detect(blank_page(), client, cfg(tiles=1, samples=2))
        assert {f.text for f in findings} == {"홍길동", "901231-1234567"}

    def test_same_value_in_both_samples_is_merged_once(self) -> None:
        payload = {"findings": [item("홍길동", bbox=(0.1, 0.1, 0.3, 0.2))]}
        client = FakeClient([payload, payload, payload])
        findings, _ = detect(blank_page(), client, cfg(tiles=1, samples=3))
        assert len(findings) == 1

    def test_sample_number_is_recorded_in_meta(self) -> None:
        client = FakeClient([{"findings": []}] * 4)
        _, metas = detect(blank_page(), client, cfg(tiles=2, samples=2))
        assert sorted((m["tile"], m["sample"]) for m in metas) == [
            (0, 0), (0, 1), (1, 0), (1, 1)
        ]

    def test_a_failed_sample_names_which_one(self) -> None:
        warnings: list[str] = []
        client = FakeClient(
            [{"findings": []}, {}], metas=[{}, {"error": "timeout"}]
        )
        detect(blank_page(), client, cfg(tiles=1, samples=2), warnings)
        assert any("샘플 1" in w for w in warnings)


# --------------------------------------------------------------------------
# 병렬 호출
# --------------------------------------------------------------------------


def banded_page(tiles: int) -> Any:
    """타일마다 픽셀값이 다른 페이지.

    병렬 실행에서는 ``complete_json`` 의 **호출 순서가 보장되지 않는다.** 그래서
    호출 카운터로 payload 를 고르는 fake 로는 타일↔payload 짝을 검증할 수 없다.
    이 페이지는 밴드마다 값이 달라서 fake 가 **받은 이미지로** 타일을 식별할 수
    있다 — 순서에 의존하지 않는 검증이 된다.
    """
    page = np.zeros((PAGE_H, PAGE_W, 3), dtype=np.uint8)
    band = PAGE_H // tiles
    for i in range(tiles):
        page[i * band : (i + 1) * band] = (i + 1) * 40
    return page


class ByImageClient:
    """이미지의 대표 픽셀값으로 타일을 식별하는 fake. 순서에 의존하지 않는다."""

    def __init__(self, tiles: int) -> None:
        self.tiles = tiles
        self.seen: list[int] = []

    client = None
    config = LlmConfig()

    def complete_json(self, system, user, schema, image=None, temperature=None):
        # 밴드 중앙을 보고 어느 타일인지 알아낸다 (겹침 구간을 피한다).
        mid = image[image.shape[0] // 2, 0, 0]
        tile_no = max(0, min(self.tiles - 1, round(int(mid) / 40) - 1))
        self.seen.append(tile_no)
        # 타일마다 고유한 값 → 어느 타일의 payload 인지 결과에서 확인할 수 있다
        return {"findings": [item(f"사람{tile_no}", bbox=(0.1, 0.4, 0.3, 0.6))]}, {}


class TestParallel:
    def test_every_tile_is_called_exactly_once(self) -> None:
        client = ByImageClient(3)
        detect(banded_page(3), client, DetectConfig(tiles=3, overlap=0.0, workers=3))
        assert sorted(client.seen) == [0, 1, 2]

    def test_results_are_the_same_as_sequential(self) -> None:
        """병렬화가 정답을 바꾸면 안 된다.

        ``workers`` 만 다르게 두 번 돌려 결과가 같은지 본다. 값과 좌표가 모두
        같아야 한다 — 좌표가 다르면 타일↔payload 짝이 어긋난 것이다.
        """
        page = banded_page(3)
        seq, _ = detect(page, ByImageClient(3), DetectConfig(tiles=3, overlap=0.0, workers=1))
        par, _ = detect(page, ByImageClient(3), DetectConfig(tiles=3, overlap=0.0, workers=3))
        assert [(f.text, f.bbox_norm) for f in seq] == [
            (f.text, f.bbox_norm) for f in par
        ]

    def test_metas_stay_in_tile_order(self) -> None:
        """메타는 호출 완료 순서가 아니라 타일 순서로 나와야 한다 (감사 추적)."""
        _, metas = detect(
            banded_page(4), ByImageClient(4), DetectConfig(tiles=4, overlap=0.0, workers=4)
        )
        assert [m["tile"] for m in metas] == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# 좌표 규약 — 조용히 망가지던 자리
# --------------------------------------------------------------------------


class TestInferScale:
    """모델이 0~1 이 아닌 규약으로 답할 때.

    Qwen-VL 계열의 native grounding 형식은 **0~1000 스케일**이다. 프롬프트가
    0.0~1.0 을 요구해도 학습된 습관대로 답하는 일이 있고, JSON Schema 의
    ``maximum: 1`` 은 문법 기반 guided decoding 이 강제해주지 않는다.
    그걸 0~1 로 알고 잘라내면 **모든 항목이 우하단 한 점으로 뭉친다.**
    """

    def test_unit_scale_is_left_alone(self) -> None:
        assert infer_scale([[0.1, 0.2, 0.3, 0.4]], 1748, 826) == (1.0, 1.0, "unit")

    def test_thousand_scale_is_detected(self) -> None:
        sx, sy, name = infer_scale([[310, 420, 440, 450]], 1748, 826)
        assert (sx, sy, name) == (1000.0, 1000.0, "per-mille")

    def test_pixel_scale_is_detected(self) -> None:
        sx, sy, name = infer_scale([[310, 420, 1500, 700]], 1748, 826)
        assert (sx, sy, name) == (1748.0, 826.0, "pixel")

    def test_decision_uses_the_whole_response_not_one_item(self) -> None:
        """항목 하나만 보면 작은 값이 0~1 인지 0~1000 인지 알 수 없다.

        같은 응답 안에 큰 값이 하나라도 있으면 전부 같은 규약이다.
        """
        raws = [[0.1, 0.2, 0.3, 0.4], [310, 420, 440, 450]]
        assert infer_scale(raws, 1748, 826)[2] == "per-mille"

    def test_empty_response_does_not_crash(self) -> None:
        assert infer_scale([], 1748, 826) == (1.0, 1.0, "unit")

    def test_garbage_entries_are_skipped(self) -> None:
        assert infer_scale([None, "x", [1, 2]], 1748, 826) == (1.0, 1.0, "unit")


class TestSmartResize:
    """``qwen-vl-utils`` 의 ``smart_resize`` 와 같은 값을 내야 한다.

    이 값이 틀리면 절대 픽셀 좌표를 잘못된 값으로 나눠서, 오차가 위치에 비례해
    커지는 밀림이 생긴다 — 정확히 진단하기 어려운 모양의 고장이다.
    """

    def test_dimensions_become_multiples_of_the_factor(self) -> None:
        h, w = smart_resize(2480, 1748, 32)
        assert h % 32 == 0 and w % 32 == 0

    def test_a4_page_is_not_downscaled_by_default(self) -> None:
        """기본 상한(16384 토큰)은 A4 300dpi 를 줄이지 않는다."""
        assert smart_resize(2480, 1748, 32) == (2496, 1760)

    def test_qwen25_factor_differs(self) -> None:
        """Qwen2/2.5-VL 은 14x2=28 이다. 모델을 바꾸면 이 값도 바뀐다."""
        h, w = smart_resize(2480, 1748, 28)
        assert h % 28 == 0 and w % 28 == 0
        assert (h, w) == (2492, 1736)

    def test_max_pixels_shrinks_and_keeps_the_aspect_ratio(self) -> None:
        h, w = smart_resize(2480, 1748, 32, max_pixels=256 * 32 * 32)
        assert h * w <= 256 * 32 * 32
        assert abs((w / h) - (1748 / 2480)) < 0.05

    def test_min_pixels_grows_a_tiny_crop(self) -> None:
        h, w = smart_resize(10, 8, 32)
        assert h >= 32 and w >= 32

    def test_never_returns_zero(self) -> None:
        h, w = smart_resize(1, 1, 32)
        assert h >= 32 and w >= 32


class TestCoordConventionEndToEnd:
    """세 규약이 모두 제자리에 떨어져야 한다.

    Qwen 계열의 grounding 좌표계는 버전마다 다르다 — Qwen2-VL 0~1000,
    Qwen2.5-VL 리사이즈 이미지의 절대 픽셀, Qwen3-VL 다시 0~1000. 모델을 바꿨을
    때 조용히 어긋나는 것이 최악이므로 세 경로를 다 고정해 둔다.
    """

    def test_per_mille_is_the_expected_native_format(self) -> None:
        """[310,420,440,450] 은 페이지의 왼쪽 중간이다. 이게 native 형식이다."""
        client = FakeClient([{"findings": [
            {"text": "김수현", "type": "NAME", "field": "신청인",
             "bbox_2d": [310, 420, 440, 450], "conf": 0.9}
        ]}])
        findings, metas = detect(blank_page(), client, cfg(tiles=1))
        assert len(findings) == 1
        x1, y1, x2, y2 = findings[0].bbox_norm
        assert 0.30 < x1 < 0.32 and 0.41 < y1 < 0.43
        assert 0.43 < x2 < 0.45 and 0.44 < y2 < 0.46
        assert metas[0]["coord_convention"] == "per-mille"

    def test_the_expected_format_is_not_warned_about(self) -> None:
        """기대한 형식으로 왔으면 조용해야 한다. 안 그러면 경고가 소음이 된다."""
        warnings: list[str] = []
        client = FakeClient([{"findings": [item("김수현")]}])
        _, metas = detect(blank_page(1000, 1400), client, cfg(tiles=1), warnings)
        assert warnings == []
        assert metas[0]["coord_convention"] == "per-mille"

    def test_decimals_still_work(self) -> None:
        """모델이 0.0~1.0 소수로 답해도 같은 자리에 떨어져야 한다."""
        client = FakeClient([{"findings": [
            {"text": "김수현", "type": "NAME", "field": "",
             "bbox_2d": [0.31, 0.42, 0.44, 0.45], "conf": 0.9}
        ]}])
        findings, metas = detect(blank_page(), client, cfg(tiles=1))
        x1, y1, _, _ = findings[0].bbox_norm
        assert 0.30 < x1 < 0.32 and 0.41 < y1 < 0.43
        assert metas[0]["coord_convention"] == "unit"

    def test_absolute_pixels_divide_by_what_the_model_saw(self) -> None:
        """Qwen2.5-VL 식 절대 픽셀. **우리가 보낸 크기가 아니라 모델이 본 크기**다.

        1000x1400 은 클라이언트 축소를 거치지 않고 factor 32 로 992x1408 이 되어
        인코더에 들어간다. 보낸 크기로 나누면 그 비율만큼 어긋나고, 오차가
        위치에 비례해 커져서 "박스가 전체적으로 밀렸다" 로 보인다.
        """
        seen_w, seen_h = 992, 1408
        client = FakeClient([{"findings": [
            {"text": "김수현", "type": "NAME", "field": "",
             "bbox_2d": [
                 int(0.31 * seen_w), int(0.80 * seen_h),
                 int(0.44 * seen_w), int(0.83 * seen_h),
             ], "conf": 0.9}
        ]}])
        warnings: list[str] = []
        findings, metas = detect(blank_page(1000, 1400), client, cfg(tiles=1), warnings)
        x1, y1, x2, y2 = findings[0].bbox_norm
        assert 0.305 < x1 < 0.315 and 0.795 < y1 < 0.805
        assert 0.435 < x2 < 0.445 and 0.825 < y2 < 0.835
        assert metas[0]["coord_convention"] == "pixel"
        assert any("절대 픽셀" in w for w in warnings)

    def test_auto_detection_has_a_real_blind_spot(self) -> None:
        """**자동 추론으로는 못 가르는 경우가 있다.** 이게 고정 옵션이 있는 이유다.

        절대 픽셀인데 그 값들이 우연히 모두 1000 미만이면 (페이지 위쪽에만
        항목이 있는 경우) per-mille 로 오판한다. 결과는 위치에 비례해 커지는
        밀림이고, 눈으로는 그냥 "박스가 밀렸다" 로만 보인다.
        """
        seen_w, seen_h = 992, 1408
        payload = {"findings": [
            {"text": "김수현", "type": "NAME", "field": "",
             "bbox_2d": [
                 int(0.31 * seen_w), int(0.42 * seen_h),
                 int(0.44 * seen_w), int(0.45 * seen_h),
             ], "conf": 0.9}
        ]}
        page = blank_page(1000, 1400)

        auto, metas = detect(page, FakeClient([payload]), cfg(tiles=1))
        assert metas[0]["coord_convention"] == "per-mille"
        assert auto[0].bbox_norm[1] > 0.5          # 실제 0.42 인데 0.59 로 밀렸다

        pinned, metas = detect(
            page, FakeClient([payload]), cfg(tiles=1, coord_convention="pixel")
        )
        assert metas[0]["coord_convention"] == "pixel"
        assert 0.415 < pinned[0].bbox_norm[1] < 0.425

    def test_a_typo_in_the_pinned_convention_is_not_silently_ignored(self) -> None:
        """조용히 auto 로 넘기면 고정한 줄 알고 쓴다."""
        with pytest.raises(ValueError, match="coord_convention"):
            detect(
                blank_page(1000, 1400),
                FakeClient([{"findings": [item("김수현")]}]),
                cfg(tiles=1, coord_convention="permille"),
            )

    def test_max_pixels_downscale_is_warned(self) -> None:
        """축소는 작은 한글을 뭉개고 그건 미탐이 된다. 조용히 넘기면 안 된다."""
        warnings: list[str] = []
        client = FakeClient([{"findings": []}])
        detect(blank_page(), client, cfg(tiles=1, max_pixels=256 * 32 * 32), warnings)
        assert any("축소되어 인코더에" in w for w in warnings)

    def test_whole_page_in_one_tile_is_warned_as_downscaled(self) -> None:
        """타일 1개로 A4 를 보내면 image_max_side 에서 0.64배로 줄어든다.

        타일링을 하는 이유가 바로 이것이다. 예전에는 이 축소가 설정 요약에만
        나왔고 실행 중에는 조용했다 — 미탐의 가장 흔한 원인인데.
        """
        warnings: list[str] = []
        detect(blank_page(), FakeClient([{"findings": []}]), cfg(tiles=1), warnings)
        assert any("축소되어 인코더에" in w for w in warnings)

    def test_tiled_page_is_not_downscaled(self) -> None:
        """타일 3개면 긴 변이 1748 이라 image_max_side(2000) 에 걸리지 않는다."""
        warnings: list[str] = []
        detect(blank_page(), FakeClient([{"findings": []}] * 3), cfg(tiles=3), warnings)
        assert not any("축소" in w for w in warnings)
