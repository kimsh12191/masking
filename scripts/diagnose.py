#!/usr/bin/env python3
"""VLM grounding 오차 진단기 — **폐쇄망에서 읽어 나올 수 있는 형태로만 출력한다.**

문서 내용을 한 글자도 출력하지 않는다. 값, 필드명, 파일명, OCR 텍스트 모두
찍지 않는다. 나가는 것은 **건수·비율·기하 통계·판정 문구**뿐이다. 그래서 화면을
그대로 읽어 주거나 사진으로 찍어 공유해도 개인정보가 나가지 않는다.

무엇을 답하는가 — "박스가 밀렸다" 를 다음 셋 중 하나로 확정한다:

    1. 좌표 규약 오류      모델이 0~1 이 아니라 0~1000/픽셀로 답한다
    2. 체계적 밀림          한 방향으로 일정하게 밀린다 (코드로 교정 가능)
    3. 무작위 grounding 오차 방향이 없고 흩어진다 (공차 확대 + 학습이 답)

셋의 처방이 완전히 다르므로 눈으로 구분하려 해선 안 된다. 판정 근거는 두 수치다.

    중앙 오프셋 (dx, dy)   VLM bbox 중심 -> 확정된 OCR 박스 중심. 페이지 비율.
                          한쪽으로 쏠려 있으면 체계적이다.
    최적 배율 (kx, ky)     "VLM 좌표에 k 를 곱하면 맞는다" 의 최소제곱 해.
                          1 에서 벗어나면 규약/리사이즈 문제다.

사용법::

    python scripts/diagnose.py test_set/*.png
    python scripts/diagnose.py test_set/*.pdf --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pii_pipeline.config import load_config  # noqa: E402
from pii_pipeline.pipeline import PiiPipeline  # noqa: E402
from pii_pipeline.schema import Agreement, PageResult, Source  # noqa: E402

#: 이 비율(페이지 긴 변 대비) 이상 한쪽으로 쏠리면 체계적 밀림으로 본다.
SYSTEMATIC_MIN = 0.015

#: 최적 배율이 이 범위를 벗어나면 좌표 규약/리사이즈 문제로 본다.
SCALE_TOL = 0.03


# --------------------------------------------------------------------------
# 집계
# --------------------------------------------------------------------------


def offsets(pages: list[PageResult]) -> list[tuple[float, float, float, float]]:
    """``(vlm 중심x, vlm 중심y, ocr 중심x, ocr 중심y)`` 를 정규화 좌표로 모은다.

    좌표가 OCR 로 확정된 영역만 쓴다 — ``vlm_coarse`` 는 최종 좌표가 VLM 것이라
        오차가 정의상 0 이고, 넣으면 밀림이 있는데도 0 으로 희석된다.

    ``nearby`` 로 고른 건은 ``bbox`` 가 VLM 좌표까지 합집합으로 덮고 있어 중심이
    중간으로 끌려간다. 그래서 ``member_index`` 가 가리키는 앞쪽 박스들 —
    **OCR 이 실제로 검출한 것** — 만 쓴다. 좌표가 같은지로 골라내면 밀림이 0인
    건이 통째로 빠져서 "밀림 없음" 을 측정할 수 없게 된다.
    """
    out: list[tuple[float, float, float, float]] = []
    for page in pages:
        if page.width <= 0 or page.height <= 0:
            continue
        for r in page.regions:
            if r.source is not Source.OCR_REFINED or r.vlm_bbox is None:
                continue
            ocr_boxes = r.member_boxes[: len(r.member_index)]
            if not ocr_boxes:
                continue
            ox1 = min(b[0] for b in ocr_boxes)
            oy1 = min(b[1] for b in ocr_boxes)
            ox2 = max(b[2] for b in ocr_boxes)
            oy2 = max(b[3] for b in ocr_boxes)
            v = r.vlm_bbox
            out.append((
                (v[0] + v[2]) / 2.0 / page.width,
                (v[1] + v[3]) / 2.0 / page.height,
                (ox1 + ox2) / 2.0 / page.width,
                (oy1 + oy2) / 2.0 / page.height,
            ))
    return out


def best_scale(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """``ocr ≈ k * vlm + b`` 의 최소제곱 기울기.

    **절편을 함께 추정해야 한다.** 원점을 지나는 직선으로 맞추면 순수한
    평행이동도 기울기로 흡수되어 (0.05 만 밀려도 k≈1.08) 배율 오차로 오진한다.
    두 원인의 처방이 정반대이므로 — 배율은 규약/리사이즈, 평행이동은 타일 환산 —
    섞이면 안 된다. 절편을 두면 기울기는 배율만, 중앙 오프셋은 평행이동만 본다.

    **표준오차를 함께 돌려준다.** grounding 이 무작위로 흩어져 있으면 기울기도
    흔들려서 1.02~1.03 이 우연히 나온다. 그걸 배율 오류로 부르면 아무 문제 없는
    좌표 환산 코드를 고치러 가게 된다. 오차보다 뚜렷하게 큰 이탈만 배율 문제다.

    페이지 중앙 근처 항목만 있으면 배율을 결정할 수 없으므로, 입력이 좁은 범위에
    몰려 있으면 ``None`` 을 돌려준다 (판정을 하지 않는다).

    Returns:
        ``(기울기, 기울기의 표준오차)``. 판정할 수 없으면 ``None``.
    """
    xs = [a for a, _ in pairs]
    n = len(pairs)
    if n < 4 or (max(xs) - min(xs)) < 0.25:
        return None
    mx = sum(xs) / n
    my = sum(b for _, b in pairs) / n
    denom = sum((a - mx) ** 2 for a in xs)
    if denom <= 0:
        return None
    k = sum((a - mx) * (b - my) for a, b in pairs) / denom
    b0 = my - k * mx
    resid = sum((b - (k * a + b0)) ** 2 for a, b in pairs)
    se = (resid / (n - 2) / denom) ** 0.5 if n > 2 else 0.0
    return (k, se)


def spread(values: list[float]) -> tuple[float, float, float]:
    """``(중앙값, 10퍼센타일, 90퍼센타일)``."""
    if not values:
        return (0.0, 0.0, 0.0)
    s = sorted(values)
    lo = s[max(0, int(len(s) * 0.10) - 1)]
    hi = s[min(len(s) - 1, int(len(s) * 0.90))]
    return (median(s), lo, hi)


def tally(pages: list[PageResult]) -> dict[str, int]:
    """영역을 확정 근거별로 센다. ``reason`` 문구가 아니라 필드로 판정한다."""
    c = Counter()
    for page in pages:
        for r in page.regions:
            c["regions"] += 1
            if r.source is not Source.OCR_REFINED:
                c["coarse"] += 1
            elif r.agreement is Agreement.EXACT:
                c["text"] += 1
            elif len(r.member_boxes) > len(r.member_index):
                # nearby 만 VLM bbox 를 member_boxes 뒤에 덧붙인다 (locate.py 참조).
                # 좌표가 같은지로 판정하면 우연히 일치한 기하 선택을 오분류한다.
                c["nearby"] += 1
            else:
                c["geometry"] += 1
    return dict(c)


def conventions(pages: list[PageResult]) -> Counter:
    """VLM 호출별 좌표 규약. ``unit`` 이 아닌 것이 하나라도 있으면 그게 원인이다."""
    c: Counter = Counter()
    for page in pages:
        for meta in page.raw_llm.get("vlm") or []:
            if isinstance(meta, dict) and meta.get("coord_convention"):
                c[meta["coord_convention"]] += 1
    return c


# --------------------------------------------------------------------------
# 출력
# --------------------------------------------------------------------------


def render(pages: list[PageResult]) -> str:
    lines: list[str] = []
    add = lines.append

    n_pages = len(pages)
    n_findings = sum(len(p.findings) for p in pages)
    t = tally(pages)
    n_regions = t.get("regions", 0)

    add("=" * 62)
    add(f" VLM grounding 진단  —  페이지 {n_pages}장 / VLM 탐지 {n_findings}건")
    add("=" * 62)
    add("")

    # ── 좌표 규약 ────────────────────────────────────────────
    conv = conventions(pages)
    add("[1] 좌표 규약 (VLM 호출 단위)")
    if not conv:
        add("    측정값 없음 — VLM 호출이 전부 실패했거나 탐지가 0건이다")
    else:
        for name, count in conv.most_common():
            mark = "" if name == "unit" else "   <-- 프롬프트가 요구한 형식이 아니다"
            add(f"    {name:<10} {count:4d}회{mark}")
    add("")

    # ── 좌표 확정 근거 ───────────────────────────────────────
    add(f"[2] 좌표 확정 근거 (영역 {n_regions}건)")
    if n_regions:
        # 숫자를 앞에 둔다 — 한글 라벨은 표시 폭이 2라서 뒤에 두면 자리가 안 맞는다.
        for key, label in (
            ("text", "텍스트 일치 (초록 / 신뢰할 수 있다)"),
            ("geometry", "위치 겹침 (초록 / 다른 셀일 수 있다)"),
            ("nearby", "근접 선택 (초록 / VLM 좌표가 밀렸다)"),
            ("coarse", "VLM 좌표 그대로 (빨강 / OCR 검출 실패)"),
        ):
            v = t.get(key, 0)
            add(f"    {v:4d}건  {v / n_regions * 100:5.1f}%   {label}")
    else:
        add("    영역 없음")
    add("")

    # ── 밀림 ─────────────────────────────────────────────────
    pts = offsets(pages)
    add(f"[3] 밀림 (OCR 로 확정된 {len(pts)}건 기준, 페이지 크기 대비 %)")
    if len(pts) < 4:
        add("    표본이 4건 미만이라 통계를 내지 않는다")
        add("")
        add(_verdict(conv, t, None))
        return "\n".join(lines)

    dxs = [ox - vx for vx, _, ox, _ in pts]
    dys = [oy - vy for _, vy, _, oy in pts]
    mdx, lodx, hidx = spread(dxs)
    mdy, lody, hidy = spread(dys)
    add(f"    dx  중앙 {mdx * 100:+6.2f}%   범위 {lodx * 100:+6.2f} ~ {hidx * 100:+6.2f}")
    add(f"    dy  중앙 {mdy * 100:+6.2f}%   범위 {lody * 100:+6.2f} ~ {hidy * 100:+6.2f}")

    fx = best_scale([(vx, ox) for vx, _, ox, _ in pts])
    fy = best_scale([(vy, oy) for _, vy, _, oy in pts])
    add(
        "    최적 배율  kx "
        + (f"{fx[0]:.3f} (±{fx[1]:.3f})" if fx else "(표본 분포가 좁아 판정 불가)")
        + "   ky "
        + (f"{fy[0]:.3f} (±{fy[1]:.3f})" if fy else "(표본 분포가 좁아 판정 불가)")
    )
    add("")
    add(_verdict(conv, t, (mdx, mdy, hidx - lodx, hidy - lody, fx, fy)))
    return "\n".join(lines)


def _off_scale(fit: tuple[float, float] | None) -> bool:
    """배율 이탈이 **표본 오차보다 뚜렷한가.** 무작위 흔들림을 배제한다."""
    if fit is None:
        return False
    k, se = fit
    return abs(k - 1.0) > max(SCALE_TOL, 2.0 * se)


def _verdict(conv: Counter, t: dict[str, int], geo: tuple | None) -> str:
    """세 원인 중 하나로 판정한다. 처방이 서로 다르므로 하나만 고른다."""
    out = ["판정"]

    bad_conv = sum(v for k, v in conv.items() if k != "unit")
    if bad_conv:
        out.append(
            f"  (1) **좌표 규약 오류** — {bad_conv}회 호출이 0~1 이 아닌 값을 냈다.\n"
            "      환산은 detect.infer_scale 이 하고 있지만 경계값에서 틀릴 수 있다.\n"
            "      처방: 프롬프트를 모델의 native 형식(0~1000)에 맞추는 편이 안전하다."
        )

    n_regions = t.get("regions", 0)
    if n_regions and t.get("coarse", 0) / n_regions > 0.5:
        out.append(
            f"  (2) **크롭 OCR 이 대부분 빈손이다** ({t['coarse']}/{n_regions}).\n"
            "      좌표 문제가 아니라 OCR 검출 문제다. paddleocr 버전/모델 경로,\n"
            "      locate 의 upscale, det_db_thresh 를 먼저 본다."
        )

    if geo is None:
        if len(out) == 1:
            out.append("  표본이 부족하다. 페이지를 더 넣고 다시 돌려라.")
        return "\n".join(out)

    mdx, mdy, sx, sy, fx, fy = geo
    scale_off = _off_scale(fx) or _off_scale(fy)
    shifted = abs(mdx) > SYSTEMATIC_MIN or abs(mdy) > SYSTEMATIC_MIN

    if scale_off:
        kx = f"{fx[0]:.3f}" if fx else "?"
        ky = f"{fy[0]:.3f}" if fy else "?"
        out.append(
            f"  (3) **배율이 어긋난다** (kx={kx}, ky={ky}).\n"
            "      평행이동이 아니라 스케일 오차다 — 좌표 규약이나 이미지\n"
            "      리사이즈 환산이 원인이다. 모델이 리사이즈된 픽셀 좌표를 냈고\n"
            "      infer_scale 이 per-mille 로 오판했을 가능성이 가장 크다."
        )
    elif shifted:
        out.append(
            f"  (4) **한 방향으로 체계적으로 밀렸다** (dx={mdx * 100:+.2f}%, "
            f"dy={mdy * 100:+.2f}%).\n"
            "      흩어짐보다 쏠림이 크다 = 코드로 교정할 수 있는 오차다.\n"
            "      타일 경계 환산(detect._to_page)과 전처리 회전을 확인하라."
        )
    elif abs(sx) < 0.005 and abs(sy) < 0.005:
        out.append(
            "  (5) **밀림이 없다.** 쏠림도 흩어짐도 0에 가깝다. 좌표는 문제가 아니다.\n"
            "      그래도 결과가 나쁘다면 탐지 자체(미탐)를 봐야 한다."
        )
    else:
        out.append(
            f"  (6) **무작위 grounding 오차다** (쏠림 dx={mdx * 100:+.2f}% "
            f"dy={mdy * 100:+.2f}%, 흩어짐 {sx * 100:.1f}%/{sy * 100:.1f}%).\n"
            "      방향이 없으므로 코드로 교정할 수 없다. 9B VLM 의 grounding\n"
            "      한계 그 자체다. 학습 없이 할 수 있는 것은 여기까지이며,\n"
            "      크롭 공차를 넓혀 OCR 이 좌표를 확정하게 하는 것이 맞는 대응이다.\n"
            "      [2] 의 '근접 선택' 비율이 이 오차를 얼마나 회수했는지 보여준다."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="VLM grounding 오차 진단 (문서 내용을 출력하지 않는다)"
    )
    ap.add_argument("inputs", nargs="+", help="이미지 또는 PDF 경로")
    ap.add_argument("--config", "-c", default=None, help="YAML 설정 경로")
    ap.add_argument("--pages", default=None, help="PDF 페이지 범위 (예: 1-3)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else None
    pipeline = PiiPipeline(cfg)

    pages: list[PageResult] = []
    for path in args.inputs:
        try:
            pages.extend(pipeline.run_any(path, pages=args.pages))
        except Exception as exc:  # noqa: BLE001 - 한 파일로 진단을 포기하지 않는다
            # 경로도 내용이 될 수 있으므로 파일명을 찍지 않는다.
            print(f"입력 1건 처리 실패: {type(exc).__name__}", file=sys.stderr)

    if not pages:
        print("처리된 페이지가 없다.", file=sys.stderr)
        return 1

    print(render(pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
