"""``scripts/measure.py`` 의 대조 로직 테스트.

이 스크립트는 vLLM 서버가 있어야 실행되지만, **틀리기 쉬운 부분은 서버가 필요
없는 쪽에 있다** — 설정 A 가 찾은 것과 설정 B 가 찾은 것이 같은 항목인지
판정하는 대조 규칙이다. 여기가 틀리면 상대 재현율이 조용히 잘못 나오고, 그
숫자로 프로젝트의 다음 단계를 정하게 된다. 그래서 대조 규칙만 따로 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pii_pipeline.schema import (
    Agreement,
    OcrStatus,
    PageResult,
    PiiRegion,
    Source,
    VlmFinding,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from measure import (  # noqa: E402
    OVERLAP_MIN,
    Item,
    _overlap_ratio,
    _same,
    collect,
    render,
    stats_for,
)


def region(
    type_: str = "NAME",
    text: str | None = "홍길동",
    bbox: tuple[int, int, int, int] = (100, 100, 300, 140),
    source: Source = Source.OCR_REFINED,
    agreement: Agreement = Agreement.EXACT,
    needs_review: bool = False,
    vlm_text: str | None = None,
) -> PiiRegion:
    return PiiRegion(
        id="r001",
        type=type_,
        bbox=bbox,
        source=source,
        confidence=0.9,
        text=text,
        vlm_text=vlm_text,
        ocr_status=OcrStatus.OK,
        needs_review=needs_review,
        agreement=agreement,
    )


def page(*regions: PiiRegion, findings: int = 0, warnings: int = 0) -> PageResult:
    return PageResult(
        image_path="p.png",
        width=1748,
        height=2480,
        regions=list(regions),
        findings=[VlmFinding(text="x", type="NAME") for _ in range(findings)],
        warnings=[f"w{i}" for i in range(warnings)],
    )


# --------------------------------------------------------------------------
# 겹침 비율
# --------------------------------------------------------------------------


class TestOverlapRatio:
    def test_identical_boxes_are_one(self) -> None:
        assert _overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_is_zero(self) -> None:
        assert _overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_touching_edges_is_zero(self) -> None:
        assert _overlap_ratio((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_contained_box_is_one_not_iou(self) -> None:
        """작은 쪽 면적 기준이라 포함 관계면 1.0 이다.

        IoU 였다면 0.04 로 나와 "다른 항목" 이 된다. 설정마다 박스 크기가 다른
        것(기하 선택은 넓고 텍스트 매칭은 좁다)이 정상이므로 IoU 로 보면 안 된다.
        """
        assert _overlap_ratio((0, 0, 100, 100), (40, 40, 60, 60)) == 1.0


# --------------------------------------------------------------------------
# 같은 항목인가
# --------------------------------------------------------------------------


class TestSame:
    def test_same_value_and_place(self) -> None:
        item = Item(key="홍길동", bbox=(100, 100, 300, 140), type="NAME")
        assert _same(item, "홍길동", (105, 102, 295, 138), "NAME")

    def test_same_value_far_apart_is_a_different_item(self) -> None:
        """같은 이름이 문서 여러 곳에 나오는 것은 정상이고 각각 세야 한다.

        하나로 합치면 두 번째 위치를 놓친 설정이 놓치지 않은 것처럼 보인다.
        """
        item = Item(key="홍길동", bbox=(100, 100, 300, 140), type="NAME")
        assert not _same(item, "홍길동", (100, 900, 300, 940), "NAME")

    def test_different_type_at_the_same_place_is_a_different_item(self) -> None:
        """유형 오분류를 합집합에서 지워버리면 그 실패가 안 보인다."""
        item = Item(key="9012311234567", bbox=(100, 100, 300, 140), type="RRN")
        assert not _same(item, "9012311234567", (100, 100, 300, 140), "DRIVER_LICENSE")

    def test_empty_value_falls_back_to_position(self) -> None:
        """서명·도장은 읽을 값이 없다. 위치와 종류로만 대조한다."""
        item = Item(key="", bbox=(600, 800, 700, 900), type="SIGNATURE")
        assert _same(item, "", (605, 805, 695, 895), "SIGNATURE")

    def test_one_side_missing_the_value_still_matches(self) -> None:
        """한 설정은 값을 읽고 다른 설정은 못 읽은 경우 (손글씨).

        같은 자리를 둘 다 찾았는데 값 유무로 갈라놓으면 재현율이 낮게 나온다.
        """
        item = Item(key="", bbox=(100, 100, 300, 140), type="NAME")
        assert _same(item, "김서연", (100, 100, 300, 140), "NAME")

    def test_different_values_at_the_same_place_are_different(self) -> None:
        item = Item(key="홍길동", bbox=(100, 100, 300, 140), type="NAME")
        assert not _same(item, "김철수", (100, 100, 300, 140), "NAME")

    def test_barely_overlapping_is_not_the_same(self) -> None:
        item = Item(key="홍길동", bbox=(0, 0, 100, 100), type="NAME")
        # 작은 쪽(10x100) 면적의 10% 만 겹친다 — 임계값 미만
        assert OVERLAP_MIN > 0.1
        assert not _same(item, "홍길동", (99, 0, 199, 100), "NAME")


# --------------------------------------------------------------------------
# 합집합 기준셋
# --------------------------------------------------------------------------


class TestCollect:
    def test_union_counts_each_item_once(self) -> None:
        r = region()
        items = collect({"base": page(r), "x3": page(region())})
        assert len(items) == 1
        assert items[0].found_by == {"base", "x3"}

    def test_item_found_by_only_one_variant(self) -> None:
        items = collect({
            "base": page(region()),
            "x3": page(region(), region(text="김철수", bbox=(100, 400, 300, 440))),
        })
        assert len(items) == 2
        only = [i for i in items if i.found_by == {"x3"}]
        assert len(only) == 1
        assert only[0].key == "김철수"

    def test_items_come_back_in_reading_order(self) -> None:
        items = collect({"base": page(
            region(text="아래", bbox=(100, 900, 300, 940)),
            region(text="위", bbox=(100, 100, 300, 140)),
        )})
        assert [i.key for i in items] == ["위", "아래"]

    def test_vlm_text_is_used_when_ocr_read_nothing(self) -> None:
        """손글씨 칸은 ``text`` 가 None 이고 ``vlm_text`` 만 있다."""
        items = collect({"base": page(region(text=None, vlm_text="김서연"))})
        assert items[0].key == "김서연"

    def test_separators_do_not_split_one_item_in_two(self) -> None:
        """설정마다 구분자 전사가 달라도 같은 값이면 같은 항목이다.

        갈라지면 합집합이 부풀고 두 설정 모두 상대재현율 50% 로 찍힌다 —
        실제로는 둘 다 같은 것을 찾았는데도.
        """
        items = collect({
            "base": page(region(type_="PHONE", text="010-1234-5678")),
            "think": page(region(type_="PHONE", text="010 1234 5678")),
        })
        assert len(items) == 1

    def test_evasion_notation_folds_to_the_same_key(self) -> None:
        """회피 표기와 정상 표기는 같은 항목이다 (``match_key`` 가 접는다)."""
        items = collect({
            "base": page(region(type_="PHONE", text="010-1234-5678")),
            "think": page(region(type_="PHONE", text="공1공-1234-5678")),
        })
        assert len(items) == 1


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------


class TestStats:
    def test_localized_rate_counts_refined_only(self) -> None:
        s = stats_for({"p": page(
            region(),
            region(source=Source.VLM_COARSE, bbox=(100, 400, 300, 440)),
        )})
        assert s["regions"] == 2
        assert s["localized"] == 0.5

    def test_disagreement_counts_non_exact(self) -> None:
        s = stats_for({"p": page(
            region(agreement=Agreement.EXACT),
            region(agreement=Agreement.NONE, bbox=(100, 400, 300, 440)),
        )})
        assert s["disagree"] == 1

    def test_empty_page_does_not_divide_by_zero(self) -> None:
        assert stats_for({"p": page()})["localized"] == 0.0


# --------------------------------------------------------------------------
# 판정 문구 — 이 스크립트의 결론
# --------------------------------------------------------------------------


def _runs(**found: list[PiiRegion]) -> dict[str, dict]:
    return {n: {"pages": {"p": page(*rs)}, "seconds": 1.0} for n, rs in found.items()}


class TestVerdict:
    def _regions(self, n: int) -> list[PiiRegion]:
        return [
            region(text=f"사람{i}", bbox=(100, 100 + i * 100, 300, 140 + i * 100))
            for i in range(n)
        ]

    def test_large_gap_says_the_capability_is_there(self) -> None:
        """think 가 훨씬 많이 찾으면 증류로 옮길 것이 있다는 뜻이다."""
        all_ten = self._regions(10)
        runs = _runs(base=all_ten[:5], think=all_ten)
        out = render(runs, collect({n: r["pages"]["p"] for n, r in runs.items()}), False)
        assert "능력은 가중치에 있다" in out

    def test_no_gap_says_there_is_nothing_to_recover(self) -> None:
        all_ten = self._regions(10)
        runs = _runs(base=all_ten, think=all_ten)
        out = render(runs, collect({n: r["pages"]["p"] for n, r in runs.items()}), False)
        assert "끌어낼 여유가 없다" in out

    def test_middling_gap_asks_for_more_documents(self) -> None:
        all_twenty = self._regions(20)
        runs = _runs(base=all_twenty[:19], think=all_twenty)
        out = render(runs, collect({n: r["pages"]["p"] for n, r in runs.items()}), False)
        assert "애매하다" in out

    def test_no_verdict_without_the_think_variant(self) -> None:
        runs = _runs(base=self._regions(3))
        out = render(runs, collect({n: r["pages"]["p"] for n, r in runs.items()}), False)
        assert "판정:" not in out

    def test_relative_recall_is_never_sold_as_absolute(self) -> None:
        """이 경고가 빠지면 100% 를 '다 찾았다' 로 읽게 된다."""
        runs = _runs(base=self._regions(3))
        out = render(runs, collect({n: r["pages"]["p"] for n, r in runs.items()}), False)
        assert "상한이 아니다" in out

    def test_per_item_table_marks_who_missed_what(self) -> None:
        all_four = self._regions(4)
        runs = _runs(base=all_four[:2], x3=all_four)
        out = render(runs, collect({n: r["pages"]["p"] for n, r in runs.items()}), True)
        assert "항목별" in out
        assert "o" in out and "." in out
