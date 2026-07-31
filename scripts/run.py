#!/usr/bin/env python3
"""파이프라인 실행 CLI.

사용 예:

    # 1장 처리 + 시각화 검수 이미지 생성
    python scripts/run.py sample.png -o out/ --overlay

    # 디렉터리 일괄 처리
    python scripts/run.py data/*.png -o out/

    # 텍스트 pass 만 (이미지 pass 끄고 베이스라인 측정)
    python scripts/run.py sample.png -o out/ --no-pass2

    # 앵커링 편향 비교용 blind 모드
    python scripts/run.py sample.png -o out/ --pass2-blind
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pii_pipeline import PiiPipeline, PipelineConfig  # noqa: E402
from pii_pipeline.llm.client import LlmConfig  # noqa: E402
from pii_pipeline.ocr.paddle_runner import OcrConfig  # noqa: E402
from pii_pipeline.viz import draw_overlay, legend_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="금융 문서 개인정보 영역 탐지",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=legend_text(),
    )
    p.add_argument("images", nargs="+", help="입력 이미지 경로")
    p.add_argument("-o", "--out", default="out", help="출력 디렉터리 (기본: out)")
    p.add_argument("--overlay", action="store_true", help="시각화 검수 이미지 생성")
    p.add_argument("--font", default=None, help="오버레이용 한글 폰트 경로")
    p.add_argument("--show-ocr-boxes", action="store_true", help="OCR 박스 전체 표시")

    g = p.add_argument_group("파이프라인")
    g.add_argument("--no-pass2", action="store_true", help="이미지 검수 pass 끄기")
    g.add_argument("--pass2-blind", action="store_true", help="1차 결과를 감추고 독립 판단")
    g.add_argument("--no-deskew", action="store_true", help="기울기 보정 끄기")
    g.add_argument("--long-side", type=int, default=2480, help="전처리 긴 변 길이")

    g = p.add_argument_group("OCR")
    g.add_argument("--det-dir", default=None, help="검출 모델 디렉터리 (폐쇄망 필수)")
    g.add_argument("--rec-dir", default=None, help="인식 모델 디렉터리 (폐쇄망 필수)")
    g.add_argument("--cls-dir", default=None, help="방향분류 모델 디렉터리")
    g.add_argument("--cpu", action="store_true", help="OCR 을 CPU 로 실행")

    g = p.add_argument_group("LLM")
    g.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM 엔드포인트")
    g.add_argument("--model", default="Qwen/Qwen3.5-9B", help="모델 이름")
    g.add_argument("--image-max-side", type=int, default=1800,
                   help="pass2 이미지 긴 변 길이. 너무 줄이면 작은 글씨를 못 읽는다")

    p.add_argument("--include-ocr", action="store_true",
                   help="결과 JSON 에 OCR 박스와 LLM 원시응답 포함 (디버깅)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ocr_cfg = OcrConfig.from_env()
    if args.det_dir:
        ocr_cfg.det_model_dir = args.det_dir
    if args.rec_dir:
        ocr_cfg.rec_model_dir = args.rec_dir
    if args.cls_dir:
        ocr_cfg.cls_model_dir = args.cls_dir
    if args.cpu:
        ocr_cfg.use_gpu = False

    config = PipelineConfig(
        enable_pass2=not args.no_pass2,
        pass2_blind=args.pass2_blind,
        deskew=not args.no_deskew,
        target_long_side=args.long_side,
        ocr=ocr_cfg,
        llm=LlmConfig(
            base_url=args.base_url,
            model=args.model,
            image_max_side=args.image_max_side,
        ),
    )

    problems = ocr_cfg.validate_offline()
    for problem in problems:
        logging.warning("%s", problem)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PiiPipeline(config)
    exit_code = 0

    for image_path in args.images:
        stem = Path(image_path).stem
        try:
            result = pipeline.run(image_path)
        except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
            logging.exception("처리 실패: %s", image_path)
            exit_code = 1
            (out_dir / f"{stem}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            continue

        json_path = out_dir / f"{stem}.json"
        json_path.write_text(
            result.to_json(include_ocr=args.include_ocr), encoding="utf-8"
        )

        stats = result.stats()
        timings = {k: round(v, 2) for k, v in result.timings.items()}
        print(f"\n=== {image_path} ===")
        print(f"  결과      {json_path}")
        print(f"  OCR 박스  {stats['n_ocr_boxes']}")
        print(f"  탐지 영역 {stats['n_regions']}  (검토필요 {stats['n_needs_review']})")
        print(f"  출처별    {json.dumps(stats['by_source'], ensure_ascii=False)}")
        print(f"  유형별    {json.dumps(stats['by_type'], ensure_ascii=False)}")
        print(f"  소요시간  {json.dumps(timings)}")
        for warning in result.warnings:
            print(f"  ! {warning}")

        if args.overlay:
            from pii_pipeline.preprocess import preprocess

            pre = preprocess(
                image_path,
                target_long_side=config.target_long_side,
                deskew=config.deskew,
            )
            overlay = draw_overlay(
                pre.image,
                result,
                font_path=args.font,
                show_ocr_boxes=args.show_ocr_boxes,
            )
            overlay_path = out_dir / f"{stem}.overlay.png"
            overlay.save(overlay_path)
            print(f"  검수 이미지 {overlay_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
