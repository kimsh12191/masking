"""``scripts/diagnose.py`` 테스트.

두 가지를 지킨다.

1. **출력에 문서 내용이 없다.** 이 진단기의 존재 이유가 폐쇄망에서 화면을 읽어
   나오는 것이므로, 값이 한 글자라도 섞이면 도구가 위험물이 된다. 그래서 값을
   심어 놓고 출력에 없는지 확인한다.
2. **판정이 원인을 맞게 가른다.** 규약 오류 / 배율 오차 / 체계적 밀림 / 무작위
   오차의 처방이 서로 다르다. 여기가 틀리면 엉뚱한 곳을 고치게 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pii_pipeline.schema import (
    Agreement,
    BBox,
    OcrStatus,
    PageResult,
    PiiRegion,
    Source,
    VlmFinding,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from diagnose import (  # noqa: E402
    best_scale,
    conventions,
    offsets,
    render,
    spread,
    tally,
)

W, H = 1000, 2000

#: 출력에 절대 나와서는 안 되는 문자열들.
SECRETS = ("901112-2846261", "홍길동", "서울시 강남구", "test_set/secret.png")


def region(
    *,
    vlm: BBox | None = (100, 100, 300, 140),
    ocr: BBox | None = (100, 100, 300, 140),
    source: Source = Source.OCR_REFINED,
    agreement: Agreement = Agreement.EXACT,
    union_vlm: bool = False,
) -> PiiRegion:
    """진단에 쓰이는 필드만 채운 영역.

    ``union_vlm=True`` 면 ``nearby`` 로 고른 영역을 흉내낸다 — ``member_boxes`` 가
    ``member_index`` 보다 하나 길다는 것이 그 표식이다 (뒤에 VLM bbox 가 붙는다).
    """
    members = [b for b in (ocr,) if b is not None]
    index = list(range(len(members)))
    if union_vlm and vlm is not None:
        members = [*members, vlm]
    return PiiRegion(
        id="r001",
        type="RRN",
        bbox=members[0] if members else (0, 0, 1, 1),
        source=source,
        confidence=0.9,
        text="901112-2846261",
        vlm_text="901112-2846261",
        vlm_bbox=vlm,
        field="주민등록번호",
        member_boxes=members,
        member_index=index,
        ocr_status=OcrStatus.OK,
        agreement=agreement,
    )


def page(*regions: PiiRegion, convention: str = "unit", n_findings: int = 0) -> PageResult:
    return PageResult(
        image_path="test_set/secret.png",
        width=W,
        height=H,
        regions=list(regions),
        findings=[VlmFinding(text="홍길동", type="NAME") for _ in range(n_findings)],
        raw_llm={"vlm": [{"coord_convention": convention}]},
    )


def shifted(n: int, dx: float, dy: float) -> PageResult:
    """``n`` 건을 페이지 전체에 흩뿌리고 OCR 박스를 ``(dx, dy)`` 만큼 옮긴다.

    ``dx``/``dy`` 는 페이지 비율이다. x 를 넓게 흩뿌려야 ``best_scale`` 이
    배율과 평행이동을 구분할 수 있다.
    """
    out: list[PiiRegion] = []
    for i in range(n):
        fx = 0.05 + 0.9 * i / max(1, n - 1)
        fy = 0.05 + 0.9 * i / max(1, n - 1)
        vx, vy = int(fx * W), int(fy * H)
        ox, oy = int((fx + dx) * W), int((fy + dy) * H)
        out.append(region(vlm=(vx, vy, vx + 100, vy + 40), ocr=(ox, oy, ox + 100, oy + 40)))
    return page(*out)


def scaled(n: int, k: float) -> PageResult:
    """OCR 박스가 VLM 좌표의 ``k`` 배 위치에 있는 페이지."""
    out: list[PiiRegion] = []
    for i in range(n):
        fx = 0.05 + 0.9 * i / max(1, n - 1)
        vx, vy = int(fx * W), int(fx * H)
        ox, oy = int(fx * k * W), int(fx * k * H)
        out.append(region(vlm=(vx, vy, vx + 100, vy + 40), ocr=(ox, oy, ox + 100, oy + 40)))
    return page(*out)


# --------------------------------------------------------------------------
# 내용이 새지 않는다 — 이 도구의 전제
# --------------------------------------------------------------------------


class TestNoLeak:
    def test_output_contains_no_document_content(self) -> None:
        out = render([shifted(10, 0.03, -0.02), page(region(), region(union_vlm=True))])
        for secret in SECRETS:
            assert secret not in out, f"출력에 문서 내용이 섞였다: {secret!r}"

    def test_output_has_no_hangul_values_even_when_all_paths_fire(self) -> None:
        """모든 분기를 한 번에 태워도 새지 않아야 한다."""
        pages = [
            shifted(8, 0.05, 0.0),
            scaled(8, 1.2),
            page(region(source=Source.VLM_COARSE, agreement=Agreement.NONE),
                 convention="per-mille"),
            page(region(agreement=Agreement.NONE, union_vlm=True)),
            page(),
        ]
        out = render(pages)
        for secret in SECRETS:
            assert secret not in out


# --------------------------------------------------------------------------
# 오프셋 수집
# --------------------------------------------------------------------------


class TestOffsets:
    def test_coarse_regions_are_excluded(self) -> None:
        """``vlm_coarse`` 는 최종 좌표가 VLM 것이라 오차가 정의상 0 이다.

        넣으면 밀림이 있는데도 0 으로 희석되어 판정이 '문제 없음' 으로 뒤집힌다.
        """
        p = page(region(source=Source.VLM_COARSE))
        assert offsets([p]) == []

    def test_regions_without_a_vlm_bbox_are_skipped(self) -> None:
        assert offsets([page(region(vlm=None))]) == []

    def test_unioned_vlm_box_does_not_pull_the_center(self) -> None:
        """``nearby`` 건은 bbox 가 VLM 좌표까지 덮는다. OCR 쪽만 봐야 한다."""
        p = page(region(vlm=(0, 0, 100, 40), ocr=(500, 1000, 600, 1040), union_vlm=True))
        (vx, vy, ox, oy) = offsets([p])[0]
        assert ox == 550 / W and oy == 1020 / H

    def test_zero_size_page_is_ignored(self) -> None:
        p = page(region())
        p.width, p.height = 0, 0
        assert offsets([p]) == []


class TestBestScale:
    def test_narrow_spread_refuses_to_judge(self) -> None:
        """페이지 중앙에만 항목이 있으면 배율과 평행이동을 구분할 수 없다."""
        assert best_scale([(0.5, 0.55), (0.51, 0.56), (0.52, 0.57), (0.53, 0.58)]) is None

    def test_recovers_a_known_scale(self) -> None:
        fit = best_scale([(0.1, 0.12), (0.4, 0.48), (0.7, 0.84), (0.9, 1.08)])
        assert fit is not None and abs(fit[0] - 1.2) < 0.01

    def test_pure_translation_is_not_read_as_a_scale(self) -> None:
        """원점 통과 회귀를 쓰면 0.05 평행이동이 k≈1.08 로 나와 오진한다."""
        fit = best_scale([(a, a + 0.05) for a in (0.1, 0.4, 0.7, 0.95)])
        assert fit is not None and abs(fit[0] - 1.0) < 0.001

    def test_noise_inflates_the_standard_error(self) -> None:
        """흔들림이 크면 기울기 이탈을 배율 문제로 부르지 못하게 막아야 한다."""
        clean = best_scale([(a, a) for a in (0.1, 0.4, 0.7, 0.95)])
        noisy = best_scale([(0.1, 0.3), (0.4, 0.2), (0.7, 0.9), (0.95, 0.7)])
        assert clean is not None and noisy is not None
        assert noisy[1] > clean[1]

    def test_too_few_points_refuses(self) -> None:
        assert best_scale([(0.1, 0.1), (0.9, 0.9)]) is None


class TestSpread:
    def test_empty_is_zero(self) -> None:
        assert spread([]) == (0.0, 0.0, 0.0)

    def test_median_and_tails(self) -> None:
        m, lo, hi = spread([float(i) for i in range(100)])
        assert lo < m < hi


# --------------------------------------------------------------------------
# 집계
# --------------------------------------------------------------------------


class TestTally:
    def test_counts_each_basis_separately(self) -> None:
        t = tally([page(
            region(),                                                    # text
            region(agreement=Agreement.NONE),                            # geometry
            region(agreement=Agreement.NONE, union_vlm=True),             # nearby
            region(source=Source.VLM_COARSE, agreement=Agreement.NONE),   # coarse
        )])
        assert t["regions"] == 4
        assert t["text"] == 1 and t["geometry"] == 1
        assert t["nearby"] == 1 and t["coarse"] == 1

    def test_conventions_are_counted_per_call(self) -> None:
        c = conventions([page(convention="per-mille"), page(convention="unit")])
        assert c["per-mille"] == 1 and c["unit"] == 1


# --------------------------------------------------------------------------
# 판정 — 이 도구의 결론
# --------------------------------------------------------------------------


class TestVerdict:
    def test_bad_convention_is_reported_first(self) -> None:
        out = render([page(region(), convention="per-mille")])
        assert "좌표 규약 오류" in out

    def test_mostly_coarse_blames_ocr_not_coordinates(self) -> None:
        """빨간 박스가 대부분이면 좌표 이야기를 하기 전에 OCR 을 봐야 한다."""
        out = render([page(*[region(source=Source.VLM_COARSE) for _ in range(5)])])
        assert "크롭 OCR 이 대부분 빈손" in out

    def test_scale_error_is_distinguished_from_translation(self) -> None:
        out = render([scaled(10, 1.2)])
        assert "배율이 어긋난다" in out
        assert "체계적으로 밀렸다" not in out

    def test_systematic_shift_is_called_correctable(self) -> None:
        out = render([shifted(10, 0.05, 0.05)])
        assert "체계적으로 밀렸다" in out

    def test_perfect_alignment_says_coordinates_are_not_the_problem(self) -> None:
        out = render([shifted(10, 0.0, 0.0)])
        assert "밀림이 없다" in out

    def test_scattered_error_is_called_random_grounding_error(self) -> None:
        """쏠림 없이 흩어지면 코드로 고칠 것이 없다 — 그게 학습이 필요하다는 결론이다.

        여기서 배율 오류라고 판정하면 안 된다. 흔들림이 기울기를 흔든 것뿐이다.
        """
        p = shifted(12, 0.0, 0.0)
        for i, r in enumerate(p.regions):
            v = r.vlm_bbox
            assert v is not None
            dx = (30, -40, 15, -25, 45, -10, 20, -35, 5, -45, 35, -20)[i]
            dy = (-50, 40, -20, 55, -35, 25, -55, 10, 45, -15, -40, 30)[i]
            r.member_boxes = [(v[0] + dx, v[1] + dy, v[2] + dx, v[3] + dy)]
        out = render([p])
        assert "무작위 grounding 오차" in out
        assert "배율이 어긋난다" not in out

    def test_tiny_sample_refuses_to_conclude(self) -> None:
        out = render([page(region())])
        assert "표본이 4건 미만" in out

    def test_empty_input_does_not_crash(self) -> None:
        out = render([page()])
        assert "영역 없음" in out
