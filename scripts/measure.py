#!/usr/bin/env python3
"""설정을 바꿔가며 돌리고 **숫자로 비교한다.**

이 스크립트가 답해야 하는 질문은 하나다.

    지금 못 하는 것이, 가중치에 능력이 없어서인가
    아니면 끌어내지 못해서인가?

같은 9B 에 추론 모드(think)를 켜서 잘 된다면 능력은 이미 있는 것이고, 그 격차는
증류로 옮길 수 있다. think 도 안 되면 후속학습으로 회수할 것이 없다는 뜻이고,
외부 감독(더 큰 교사·OCR 라벨러)이 필요하다. **어느 쪽이냐로 프로젝트의 다음
단계가 갈린다.** 그래서 학습에 손대기 전에 이 숫자부터 본다.

사람 라벨 없이 재는 법
----------------------

정답이 없으니 **절대 재현율은 못 잰다.** 대신 두 가지를 잰다.

``상대 재현율``
    모든 설정이 찾아낸 것의 **합집합**을 기준으로 삼고, 각 설정이 그중 몇 %를
    찾았는지 본다. 어떤 설정이 다른 설정보다 더 찾는지는 이걸로 충분히 갈린다.
    합집합 자체가 놓친 것은 여전히 안 보인다 — 이 값은 상한이 아니다.

``단독 발견``
    그 설정만 찾은 항목 수. 이게 0 이 아니면 다른 설정들이 그만큼 놓치고 있다는
    뜻이다. 합집합을 늘리는 데 기여하는 설정이 무엇인지 보여준다.

나머지(``좌표확정률``, ``불일치``, ``검토필요``)는 정답 없이도 그대로 의미가 있다.

사용 예::

    # 기본 3종 비교 (no-think / no-think x3샘플 / think)
    python scripts/measure.py docs/*.png

    # 특정 설정만
    python scripts/measure.py docs/*.png --only base,think

    # 항목별 표까지 (어느 설정이 무엇을 놓쳤는지)
    python scripts/measure.py docs/*.png --per-item

주의: ``think`` 설정은 guided decoding 을 **끈다.** 스키마를 강제하면 모델이 첫
토큰부터 JSON 을 뱉어야 해서 사고 토큰이 나올 자리가 없다 — 켠 줄 알고 재면
no-think 와 같은 것을 재게 된다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pii_pipeline import PiiPipeline  # noqa: E402
from pii_pipeline.config import load_config  # noqa: E402
from pii_pipeline.detect import match_key  # noqa: E402
from pii_pipeline.schema import PageResult, Source  # noqa: E402

log = logging.getLogger("measure")

#: 두 영역이 같은 항목인지 볼 때의 최소 겹침 비율 (작은 쪽 면적 기준).
#: IoU 가 아니라 포함률을 쓴다 — 설정마다 박스 크기가 달라서 (기하 선택은 넓고
#: 텍스트 매칭은 좁다) IoU 로 보면 같은 항목이 다른 것으로 갈린다.
OVERLAP_MIN = 0.3


# --------------------------------------------------------------------------
# 비교할 설정들
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Variant:
    """비교 대상 하나.

    Attributes:
        name: 표에 찍힐 이름.
        why: 이 설정으로 무엇을 알아보려는 것인지. 표 아래 범례로 나간다.
        detect: ``DetectConfig`` 에 덮어쓸 값.
        llm: ``LlmConfig`` 에 덮어쓸 값.
    """

    name: str
    why: str
    detect: dict[str, Any] = dataclasses.field(default_factory=dict)
    llm: dict[str, Any] = dataclasses.field(default_factory=dict)


VARIANTS: list[Variant] = [
    Variant(
        name="base",
        why="현재 운영 설정. 나머지는 전부 이것과 비교한다",
    ),
    Variant(
        name="x3",
        why="같은 타일을 3번 뽑아 합집합. 학습 없이 미탐을 줄이는 손잡이",
        detect={"samples": 3},
    ),
    Variant(
        name="think",
        why="가중치에 능력이 있는지 본다. 여기서 잘 되면 격차는 증류로 옮길 수 있다",
        detect={"samples": 1},
        # guided 를 끄지 않으면 추론 모드가 조용히 무력화된다 (모듈 docstring).
        # 사고 토큰이 출력 예산을 먹으므로 상한도 함께 올린다.
        llm={
            "enable_thinking": True,
            "guided": False,
            "max_tokens": 8192,
            "max_tokens_on_truncation": 16384,
            "timeout": 300.0,
        },
    ),
]


# --------------------------------------------------------------------------
# 항목 대조
# --------------------------------------------------------------------------


def _overlap_ratio(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """겹친 넓이 / **작은 쪽** 넓이."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / smaller if smaller > 0 else 0.0


@dataclasses.dataclass
class Item:
    """설정 간 대조 단위. 같은 개인정보 1건."""

    key: str  # 정규화된 값 (없으면 "")
    bbox: tuple[int, int, int, int]
    type: str
    found_by: set[str] = dataclasses.field(default_factory=set)

    def label(self) -> str:
        return f"{self.type} '{self.key or '(값없음)'}' @{self.bbox}"


def _same(item: Item, key: str, bbox: tuple[int, int, int, int], type_: str) -> bool:
    """같은 항목인가.

    값이 같고 위치가 겹치면 같은 것으로 본다. **값이 같아도 위치가 멀면 다른
    항목이다** — 같은 이름이 문서 여러 곳에 나오는 것은 정상이고 각각 세야 한다.

    값이 비어 있는 경우(서명·도장, 또는 설정마다 전사가 갈린 경우)는 위치와
    종류만으로 본다. 종류가 다르면 다른 항목으로 센다 — 유형 오분류를 합집합에서
    지워버리면 그 실패가 보이지 않게 된다.
    """
    if item.type != type_:
        return False
    if _overlap_ratio(item.bbox, bbox) < OVERLAP_MIN:
        return False
    return not (item.key and key) or item.key == key


def collect(results: dict[str, PageResult]) -> list[Item]:
    """설정별 결과를 하나의 항목 목록으로 합친다 (= 합집합 기준셋)."""
    items: list[Item] = []
    for name, page in results.items():
        for region in page.regions:
            key = match_key(region.text or region.vlm_text or "")
            hit = next((i for i in items if _same(i, key, region.bbox, region.type)), None)
            if hit is None:
                items.append(Item(key=key, bbox=region.bbox, type=region.type,
                                  found_by={name}))
            else:
                hit.found_by.add(name)
                if not hit.key:
                    hit.key = key
    items.sort(key=lambda i: (i.bbox[1], i.bbox[0]))
    return items


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------


def run_variant(variant: Variant, base_cfg: Any, paths: list[str]) -> dict[str, Any]:
    """한 설정으로 전체 문서를 돌린다."""
    cfg = dataclasses.replace(base_cfg)
    cfg.detect = dataclasses.replace(base_cfg.detect, **variant.detect)
    cfg.llm = dataclasses.replace(base_cfg.llm, **variant.llm)

    pipeline = PiiPipeline(cfg)
    pages: dict[str, PageResult] = {}
    started = time.perf_counter()
    for path in paths:
        try:
            for n, result in enumerate(pipeline.run_any(path)):
                pages[f"{Path(path).stem}#{n}"] = result
        except Exception as exc:  # noqa: BLE001 - 한 문서 실패로 측정을 접지 않는다
            log.error("[%s] %s 처리 실패: %s", variant.name, path, exc)
    return {"pages": pages, "seconds": time.perf_counter() - started}


def stats_for(pages: dict[str, PageResult]) -> dict[str, float]:
    """정답 없이도 의미가 있는 지표들."""
    regions = [r for p in pages.values() for r in p.regions]
    findings = sum(len(p.findings) for p in pages.values())
    refined = sum(1 for r in regions if r.source is Source.OCR_REFINED)
    n = len(regions)
    return {
        "findings": findings,
        "regions": n,
        "localized": refined / n if n else 0.0,
        "disagree": sum(1 for r in regions if r.agreement.value != "exact"),
        "review": sum(1 for r in regions if r.needs_review),
        "warnings": sum(len(p.warnings) for p in pages.values()),
    }


def render(
    runs: dict[str, dict[str, Any]], items: list[Item], per_item: bool
) -> str:
    """비교표."""
    names = list(runs)
    total = len(items)
    lines: list[str] = []

    head = f"{'설정':<8}{'탐지':>6}{'영역':>6}{'좌표확정':>9}{'불일치':>7}{'검토':>6}"
    head += f"{'상대재현율':>11}{'단독발견':>9}{'초':>8}"
    lines += ["", head, "-" * len(head) * 2]

    for name in names:
        s = stats_for(runs[name]["pages"])
        found = sum(1 for i in items if name in i.found_by)
        only = sum(1 for i in items if i.found_by == {name})
        lines.append(
            f"{name:<8}{s['findings']:>6}{s['regions']:>6}"
            f"{s['localized']:>8.0%} {s['disagree']:>7}{s['review']:>6}"
            f"{(found / total if total else 0):>10.0%} {only:>9}"
            f"{runs[name]['seconds']:>8.1f}"
        )

    lines += [
        "",
        f"합집합 기준셋 {total}건 — 어느 설정이든 한 번이라도 찾은 항목의 수.",
        "**상대재현율은 상한이 아니다.** 모든 설정이 함께 놓친 것은 여기에 없다.",
    ]

    if total and "think" in names:
        think = sum(1 for i in items if "think" in i.found_by)
        base = sum(1 for i in items if "base" in i.found_by)
        gap = (think - base) / total
        lines += ["", "판정:"]
        if gap >= 0.10:
            lines.append(
                f"  think 가 base 보다 {gap:.0%} 더 찾았다 → **능력은 가중치에 있다.**"
                "\n  격차를 증류로 옮기는 것이 다음 단계다 (think 트레이스를 학습 데이터로)."
            )
        elif gap <= 0.02:
            lines.append(
                f"  think 와 base 의 차이가 {gap:.0%} 다 → **끌어낼 여유가 없다.**"
                "\n  증류로 회수할 것이 없다. 외부 감독(더 큰 교사·OCR 라벨러)이 필요하다."
            )
        else:
            lines.append(
                f"  격차 {gap:.0%} — 애매하다. 문서를 늘려 다시 재라 "
                "(장수가 적으면 이 값은 흔들린다)."
            )

    if per_item:
        lines += ["", "항목별 (o=찾음, .=놓침)", ""]
        lines.append(f"{'':<{max(1, max((len(i.label()) for i in items), default=1))}}  "
                     + " ".join(f"{n[:5]:>5}" for n in names))
        for i in items:
            marks = " ".join(f"{'o' if n in i.found_by else '.':>5}" for n in names)
            lines.append(f"{i.label():<60}  {marks}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="설정을 바꿔가며 돌리고 숫자로 비교한다 (사람 라벨 불필요)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {v.name:<7} {v.why}" for v in VARIANTS),
    )
    p.add_argument("inputs", nargs="+", help="이미지 또는 PDF")
    p.add_argument("-c", "--config", default=None, help="기준 설정 파일")
    p.add_argument("--only", default=None,
                   help=f"쉼표로 구분한 설정 이름 (기본: 전부). "
                        f"가능: {','.join(v.name for v in VARIANTS)}")
    p.add_argument("--per-item", action="store_true",
                   help="어느 설정이 무엇을 놓쳤는지 항목별로 출력")
    p.add_argument("--json", default=None, help="결과를 JSON 으로도 저장할 경로")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    wanted = set(args.only.split(",")) if args.only else {v.name for v in VARIANTS}
    unknown = wanted - {v.name for v in VARIANTS}
    if unknown:
        print(f"모르는 설정: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    variants = [v for v in VARIANTS if v.name in wanted]

    base_cfg = load_config(args.config).pipeline

    runs: dict[str, dict[str, Any]] = {}
    for variant in variants:
        print(f"[{variant.name}] {len(args.inputs)}개 문서 처리 중...", file=sys.stderr)
        runs[variant.name] = run_variant(variant, base_cfg, args.inputs)

    # **페이지별로** 대조한다. 문서 전체를 한 번에 합치면 다른 페이지의 같은
    # 좌표가 서로 겹친 것으로 잡힌다.
    items: list[Item] = []
    page_names = sorted({k for r in runs.values() for k in r["pages"]})
    for page_name in page_names:
        per_page = {n: r["pages"][page_name] for n, r in runs.items()
                    if page_name in r["pages"]}
        items.extend(collect(per_page))

    report = render(runs, items, args.per_item)
    print(report)

    if args.json:
        payload = {
            "variants": {
                n: {"seconds": r["seconds"], **stats_for(r["pages"])}
                for n, r in runs.items()
            },
            "union_total": len(items),
            "items": [
                {"type": i.type, "text": i.key, "bbox": list(i.bbox),
                 "found_by": sorted(i.found_by)}
                for i in items
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
