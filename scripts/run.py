#!/usr/bin/env python3
"""파이프라인 실행 CLI.

이미지를 입력하면 페이지당 두 파일을 남긴다.

    {이름}.boxes.png   박스 영역이 표시된 이미지
    {이름}.json        박스 위치 + 개인정보 유형

사용 예:

    # 1장 처리 (이미지 + JSON 둘 다 저장)
    python scripts/run.py sample.png -o out/

    # 디렉터리 일괄 처리
    python scripts/run.py data/*.png -o out/

    # JSON 만 (이미지 저장 생략)
    python scripts/run.py sample.png -o out/ --no-image

    # 텍스트 pass 만 (이미지 pass 끄고 베이스라인 측정)
    python scripts/run.py sample.png -o out/ --no-pass2

    # 앵커링 편향 비교용 blind 모드
    python scripts/run.py sample.png -o out/ --pass2-blind
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pii_pipeline import PiiPipeline, PipelineConfig  # noqa: E402
from pii_pipeline.llm.client import LlmConfig  # noqa: E402
from pii_pipeline.ocr.paddle_runner import OcrConfig  # noqa: E402
from pii_pipeline.output import format_summary, save_result  # noqa: E402
from pii_pipeline.viz import legend_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="금융 문서 개인정보 영역 탐지 — 박스 표시 이미지 + 위치 JSON 을 저장한다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=legend_text(),
    )
    p.add_argument("images", nargs="+", help="입력 이미지 경로")
    p.add_argument("-o", "--out", default="out", help="출력 디렉터리 (기본: out)")

    g = p.add_argument_group("출력")
    g.add_argument("--no-image", action="store_true",
                   help="박스 표시 이미지를 저장하지 않고 JSON 만 남긴다")
    g.add_argument("--font", default=None,
                   help="한글 폰트 경로 (없어도 라벨은 ASCII 로 표시된다)")
    g.add_argument("--show-ocr-boxes", action="store_true",
                   help="OCR 박스 전체를 회색으로 함께 표시 (디버깅)")
    g.add_argument("--include-ocr", action="store_true",
                   help="JSON 에 OCR 박스와 LLM 원시응답 포함 (디버깅)")

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

        # 박스 표시 이미지와 위치 JSON 을 함께 저장한다.
        # 전처리 이미지는 result 가 들고 있으므로 재처리하지 않는다.
        written = save_result(
            result,
            out_dir,
            stem=stem,
            write_image=not args.no_image,
            include_ocr=args.include_ocr,
            font_path=args.font,
            show_ocr_boxes=args.show_ocr_boxes,
        )

        print()
        print(format_summary(result, written))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
