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
    match_key,
    tile_rects,
)
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
    return {"text": text, "type": label, "field": "성명", "bbox_norm": list(bbox), "conf": conf}


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
        """조각 안의 y=0.5 는 페이지의 y=0.5 가 아니다."""
        client = FakeClient([
            {"findings": []},
            {"findings": [item("홍길동", bbox=(0.2, 0.0, 0.4, 1.0))]},
        ])
        findings, _ = detect(blank_page(), client, cfg(tiles=2, overlap=0.0))
        assert len(findings) == 1
        # 두 번째 조각은 페이지 y 0.5~1.0 구간이다
        assert findings[0].bbox_norm[1] == pytest.approx(0.5)
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
        assert any("bbox_norm" in w for w in warnings)

    def test_item_without_type_is_skipped(self) -> None:
        client = FakeClient([{"findings": [{"text": "x", "bbox_norm": [0, 0, 1, 1]}]}])
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
