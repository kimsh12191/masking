#!/usr/bin/env python3
"""VLM 좌표 학습 데이터 생성기.

문서 이미지 → 타일 이미지 + 정답 JSON. **사람 라벨 0건.**

    이미지/PDF
      └─ preprocess          추론과 **같은** 전처리 (해상도·기울기·격자정렬)
      └─ 페이지 전역 OCR     한 번만. 타일마다 돌리면 경계에서 정답이 어긋난다
      └─ tile_rects          추론과 **같은** 타일 분할
      └─ 타일별 정답 생성    OCR 박스 → 타일 기준 0~1000 정수
      └─ 왕복 검산           추론 디코딩으로 되돌려 원래 박스가 나오는가
      └─ tiles/*.png + data.jsonl

이 스크립트가 실제로 지키는 계약은 하나다. **모델이 학습에서 보는 이미지와
추론에서 보는 이미지가 같고, 좌표 규약도 같다.** 그래서 전처리·타일링을
재구현하지 않고 ``pii_pipeline`` 의 함수를 그대로 호출한다. 여기서 한 번
어긋나면 틀린 좌표를 학습시키게 되고, 그 오차는 결과만 보고는 찾을 수 없다.

사용법::

    # 데이터 생성 (+ 검수용 오버레이 20장)
    python scripts/build_grounding_data.py docs/*.png -o data/grounding --overlay 20

    # 설정을 서빙과 맞춰서
    python scripts/build_grounding_data.py scans/*.pdf -o data/grounding -c config/default.yaml

**만든 뒤 반드시 오버레이를 눈으로 확인할 것.** 검산은 좌표 변환이 일관된지만
보증하고, OCR 이 애초에 엉뚱한 곳을 잡았는지는 못 본다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pii_pipeline.config import load_config  # noqa: E402
from pii_pipeline.detect import crop_norm, tile_rects  # noqa: E402
from pii_pipeline.llm.client import fit_max_side  # noqa: E402
from pii_pipeline.ocr.paddle_runner import PaddleOcrRunner  # noqa: E402
from pii_pipeline.pdf import is_pdf, render_pages  # noqa: E402
from pii_pipeline.pipeline import PipelineConfig  # noqa: E402
from pii_pipeline.preprocess import preprocess, preprocess_array  # noqa: E402
from pii_pipeline.train.dataset import (  # noqa: E402
    GroundingConfig,
    TileSample,
    build_tile_sample,
    quantization_limit,
    roundtrip,
)

log = logging.getLogger("build_grounding_data")


# --------------------------------------------------------------------------
# 페이지 처리
# --------------------------------------------------------------------------


def page_samples(
    image,
    cfg: PipelineConfig,
    gcfg: GroundingConfig,
    ocr: PaddleOcrRunner,
) -> tuple[list[tuple[TileSample, object]], list[str]]:
    """전처리된 페이지에서 타일 샘플들을 만든다.

    Returns:
        ``([(샘플, 타일이미지)], 문제 목록)``. 문제 목록이 비어 있지 않으면
        그 페이지는 학습에 쓰면 안 된다.
    """
    page_h, page_w = image.shape[:2]
    problems: list[str] = []

    boxes = ocr.run(image)
    if not boxes:
        return [], ["페이지 OCR 이 박스를 하나도 못 찾았다"]

    rects = tile_rects(
        page_w, page_h, cfg.detect.tiles, cfg.detect.overlap, cfg.detect.image_factor
    )

    out: list[tuple[TileSample, object]] = []
    for i, rect in enumerate(rects):
        sample = build_tile_sample(boxes, i, rect, page_w, page_h, gcfg)
        if sample is None:
            continue

        # ── 왕복 검산 ────────────────────────────────────────
        # 남아도 되는 오차는 per-mille 양자화뿐이다. 그보다 크면 전처리·타일
        # 분할·좌표 환산 중 어딘가가 추론 경로와 달라졌다는 뜻이고, 그 상태로
        # 만든 데이터는 **틀린 좌표를 가르친다.**
        limit = quantization_limit(rect, page_w, page_h) + 1.0
        if sample.max_error_px > limit:
            problems.append(
                f"타일 {i}: 왕복 오차 {sample.max_error_px:.1f}px "
                f"> 허용 {limit:.1f}px — 좌표 변환이 추론 경로와 어긋났다"
            )
            continue

        tile_img = crop_norm(image, rect) if len(rects) > 1 else image
        out.append((sample, as_model_sees(tile_img, cfg.llm.image_max_side)))

    return out, problems


def as_model_sees(tile, max_side: int):
    """추론에서 vLLM 으로 보내는 것과 **같은 픽셀**로 만든다.

    ``LlmClient.encode_image`` 가 보내기 직전에 긴 변을 ``image_max_side`` 로
    한 번 더 줄인다 (PIL LANCZOS). 학습에서 이 단계를 빼면 모델이 학습 때와
    추론 때 **다른 이미지**를 보게 된다.

    지금 기본 설정에서는 타일 긴 변이 1760, 상한이 1984 라 축소가 일어나지
    않는다. 그래서 빼먹어도 당장은 티가 안 나고, ``target_long_side`` 를 올리는
    순간 조용히 갈라진다. 우연히 맞는 것에 기대지 않는다.

    **좌표는 손대지 않는다.** per-mille 은 이미지 크기에 불변이라 리사이즈해도
    같은 값이다.

    Returns:
        BGR numpy 배열. 축소가 없으면 입력을 그대로 돌려준다.
    """
    h, w = tile.shape[:2]
    new_h, new_w = fit_max_side(h, w, max_side)
    if (new_h, new_w) == (h, w):
        return tile

    # 축소가 실제로 필요할 때만 PIL 을 부른다. 기본 설정에서는 여기까지 오지
    # 않으므로 --overlay 0 이면 Pillow 없이도 데이터를 만들 수 있다.
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    # 추론과 같은 보간을 쓴다. cv2.INTER_AREA 로 대신하면 미세하게 다른
    # 픽셀이 나오고, 작은 글씨에서는 그 차이가 무의미하지 않다.
    rgb = Image.fromarray(tile[:, :, ::-1]).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(rgb)[:, :, ::-1]


def load_pages(path: str, cfg: PipelineConfig):
    """이미지/PDF 를 전처리된 페이지 배열로 내놓는다 (추론과 같은 전처리)."""
    if is_pdf(path):
        for page in render_pages(path, target_long_side=cfg.target_long_side):
            pre = preprocess_array(
                page.image,
                target_long_side=cfg.target_long_side,
                deskew=cfg.deskew,
                align=cfg.detect.image_factor,
            )
            yield f"{Path(path).stem}_p{page.page_no:03d}", pre.image
    else:
        pre = preprocess(
            path,
            target_long_side=cfg.target_long_side,
            deskew=cfg.deskew,
            align=cfg.detect.image_factor,
        )
        yield Path(path).stem, pre.image


# --------------------------------------------------------------------------
# 검수용 오버레이
# --------------------------------------------------------------------------


def save_overlay(tile_img, sample: TileSample, path: Path, page_w: int, page_h: int) -> None:
    """정답 박스를 타일 위에 그린다. **눈으로 확인하는 것이 최종 검증이다.**

    좌표는 저장된 ``bbox_2d`` 에서 **추론 디코딩을 거쳐** 되돌린 값을 쓴다.
    OCR 박스를 그대로 그리면 변환이 틀려도 그림은 멀쩡해서, 정작 검증하려는
    것을 검증하지 못한다.
    """
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    arr = np.asarray(tile_img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]
    canvas = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    ox, oy = sample.rect[0] * page_w, sample.rect[1] * page_h
    # 타일이 image_max_side 로 축소됐을 수 있다 (as_model_sees). 그림도 같은
    # 배율로 줄여야 박스가 글자에 맞는다. 축소가 없으면 1.0 이다.
    scale = canvas.width / max(1.0, (sample.rect[2] - sample.rect[0]) * page_w)

    for item in sample.items:
        x1, y1, x2, y2 = roundtrip(item["bbox_2d"], sample.rect, page_w, page_h)
        color = (0, 168, 89) if item["text"] else (214, 45, 32)  # 초록: 글자 / 빨강: 좌표만
        draw.rectangle(
            (
                (x1 - ox) * scale,
                (y1 - oy) * scale,
                (x2 - ox) * scale,
                (y2 - oy) * scale,
            ),
            outline=color,
            width=2,
        )

    canvas.save(path)


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="VLM 좌표 학습 데이터 생성 (OCR 을 정답으로 사용)"
    )
    ap.add_argument("inputs", nargs="+", help="문서 이미지 또는 PDF 경로")
    ap.add_argument("-o", "--out", required=True, help="출력 디렉터리")
    ap.add_argument("-c", "--config", default=None, help="YAML 설정 (서빙과 같은 것)")
    ap.add_argument(
        "--overlay",
        type=int,
        default=20,
        help="검수용 오버레이를 앞에서 N개 저장 (0 이면 끔)",
    )
    ap.add_argument("--text-min-conf", type=float, default=0.9)
    ap.add_argument(
        "--no-textless",
        action="store_true",
        help="FAILED/저신뢰 박스를 빼고 인쇄 텍스트만 (손글씨·도장을 못 배운다)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = load_config(args.config).pipeline if args.config else PipelineConfig()
    gcfg = GroundingConfig(
        text_min_conf=args.text_min_conf,
        keep_textless=not args.no_textless,
    )

    out_dir = Path(args.out)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        (out_dir / "overlay").mkdir(exist_ok=True)

    ocr = PaddleOcrRunner(cfg.ocr)

    n_pages = n_samples = n_items = n_textless = 0
    n_overlay = 0
    worst_error = 0.0
    problems: list[str] = []

    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as fp:
        for path in args.inputs:
            try:
                pages = list(load_pages(path, cfg))
            except Exception as exc:  # noqa: BLE001 - 한 파일로 전체를 버리지 않는다
                problems.append(f"{Path(path).name}: 읽기 실패 ({type(exc).__name__})")
                continue

            for stem, image in pages:
                n_pages += 1
                page_h, page_w = image.shape[:2]
                samples, page_problems = page_samples(image, cfg, gcfg, ocr)
                problems.extend(f"{stem} {p}" for p in page_problems)

                for sample, tile_img in samples:
                    name = f"{stem}_t{sample.tile}"
                    tile_path = tiles_dir / f"{name}.png"

                    import cv2  # type: ignore[import-not-found]

                    cv2.imwrite(str(tile_path), tile_img)

                    fp.write(
                        json.dumps(
                            {
                                "image": str(tile_path.relative_to(out_dir)),
                                "target": {"findings": sample.items},
                                # 재현·디버깅용. 학습에는 쓰지 않는다.
                                "meta": {
                                    "page": stem,
                                    "tile": sample.tile,
                                    "rect": [round(v, 6) for v in sample.rect],
                                    "page_size": [page_w, page_h],
                                    "max_error_px": round(sample.max_error_px, 2),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    n_samples += 1
                    n_items += len(sample.items)
                    n_textless += sample.n_textless
                    worst_error = max(worst_error, sample.max_error_px)

                    if n_overlay < args.overlay:
                        save_overlay(
                            tile_img,
                            sample,
                            out_dir / "overlay" / f"{name}.png",
                            page_w,
                            page_h,
                        )
                        n_overlay += 1

    # ── 요약 ──────────────────────────────────────────────────
    print()
    print("=" * 58)
    print(f" 페이지        {n_pages}")
    print(f" 타일 샘플     {n_samples}")
    print(f" 항목          {n_items}  (타일당 평균 {n_items / max(1, n_samples):.1f})")
    print(
        f"   글자+좌표   {n_items - n_textless}"
        f"   좌표만      {n_textless}  (손글씨·도장·저신뢰)"
    )
    print(f" 최대 왕복오차 {worst_error:.2f}px  ← per-mille 양자화 한계 이내여야 한다")
    print("=" * 58)

    if problems:
        print(f"\n문제 {len(problems)}건:")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  ... 외 {len(problems) - 20}건")

    if args.overlay and n_overlay:
        print(f"\n검수: {out_dir / 'overlay'} 의 {n_overlay}장을 **눈으로 확인할 것.**")
        print("  초록 = 글자와 좌표를 함께 가르치는 항목")
        print("  빨강 = 좌표만 가르치는 항목 (손글씨·도장·저신뢰)")
        print("  박스가 글자에 안 붙어 있으면 학습에 쓰지 마라.")

    return 1 if (problems and not n_samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
