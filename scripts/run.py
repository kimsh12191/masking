#!/usr/bin/env python3
"""파이프라인 실행 CLI.

이미지를 입력하면 페이지당 두 파일을 남긴다.

    {이름}.boxes.png   박스 영역이 표시된 이미지
    {이름}.json        박스 위치 + 개인정보 유형

설정은 네 단계로 겹쳐 적용된다 (뒤가 앞을 덮는다):

    코드 기본값 → config.yaml → 환경변수(PII_*) → CLI 플래그

사용 예:

    # 설정 파일만으로 실행 (config.yaml 또는 config/default.yaml 자동 탐색)
    python scripts/run.py sample.png

    # 설정 파일 지정
    python scripts/run.py sample.png --config config/prod.yaml

    # 디렉터리 일괄 처리
    python scripts/run.py data/*.png -o out/

    # 일회성 변경 — 설정 파일을 고치지 않고 덮어쓰기
    python scripts/run.py sample.png --model Qwen/Qwen3.5-9B --no-pass2

    # 적용된 설정만 확인하고 종료
    python scripts/run.py --print-config
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pii_pipeline import PiiPipeline  # noqa: E402
from pii_pipeline.config import describe, load_config  # noqa: E402
from pii_pipeline.output import format_summary, save_result  # noqa: E402
from pii_pipeline.viz import legend_text  # noqa: E402

BOOL = argparse.BooleanOptionalAction


def build_parser() -> argparse.ArgumentParser:
    """CLI 파서.

    모든 기본값이 ``None`` 이다. 설정 파일 값을 덮어쓸지 판단하려면
    "지정되지 않음"과 "false 로 지정됨"을 구분해야 한다.
    """
    p = argparse.ArgumentParser(
        description="금융 문서 개인정보 영역 탐지 — 박스 표시 이미지 + 위치 JSON 을 저장한다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=legend_text(),
    )
    p.add_argument("images", nargs="*", help="입력 이미지 경로")
    p.add_argument("-c", "--config", default=None,
                   help="설정 파일 경로 "
                        "(기본: $PII_CONFIG > ./config.yaml > ./config/default.yaml)")
    p.add_argument("--print-config", action="store_true",
                   help="적용된 설정을 출력하고 종료")
    p.add_argument("-v", "--verbose", action="store_true")

    g = p.add_argument_group("출력")
    g.add_argument("-o", "--out", default=None, help="출력 디렉터리")
    g.add_argument("--image", action=BOOL, default=None,
                   help="박스 표시 이미지 저장 여부 (--no-image 로 끈다)")
    g.add_argument("--font", default=None, help="한글 폰트 경로")
    g.add_argument("--show-ocr-boxes", action=BOOL, default=None,
                   help="OCR 박스 전체를 회색으로 표시 (디버깅)")
    g.add_argument("--include-ocr", action=BOOL, default=None,
                   help="JSON 에 OCR 박스와 LLM 원시응답 포함 (디버깅)")

    g = p.add_argument_group("파이프라인")
    g.add_argument("--pass2", action=BOOL, default=None,
                   help="이미지 검수 pass (--no-pass2 로 끈다)")
    g.add_argument("--pass2-blind", action=BOOL, default=None,
                   help="1차 결과를 감추고 독립 판단")
    g.add_argument("--deskew", action=BOOL, default=None, help="기울기 보정")
    g.add_argument("--long-side", type=int, default=None, help="전처리 긴 변 길이")

    g = p.add_argument_group("OCR")
    g.add_argument("--det-dir", default=None, help="검출 모델 디렉터리 (폐쇄망 필수)")
    g.add_argument("--rec-dir", default=None, help="인식 모델 디렉터리 (폐쇄망 필수)")
    g.add_argument("--cls-dir", default=None, help="방향분류 모델 디렉터리")
    g.add_argument("--cpu", action="store_true", help="OCR 을 CPU 로 실행")

    g = p.add_argument_group("LLM")
    g.add_argument("--base-url", default=None, help="vLLM 엔드포인트")
    g.add_argument("--model", default=None, help="모델 이름")
    g.add_argument("--image-max-side", type=int, default=None,
                   help="pass2 이미지 긴 변 길이. 1500 이상 유지할 것")
    return p


def apply_cli_overrides(config, args: argparse.Namespace) -> None:
    """``None`` 이 아닌 CLI 인자만 설정에 덮어쓴다 (④ 단계)."""
    pipe, ocr, llm, out = config.pipeline, config.pipeline.ocr, config.pipeline.llm, config.output

    for value, target, attr in (
        (args.out, out, "out_dir"),
        (args.image, out, "write_image"),
        (args.font, out, "font_path"),
        (args.show_ocr_boxes, out, "show_ocr_boxes"),
        (args.include_ocr, out, "include_ocr"),
        (args.pass2, pipe, "enable_pass2"),
        (args.pass2_blind, pipe, "pass2_blind"),
        (args.deskew, pipe, "deskew"),
        (args.long_side, pipe, "target_long_side"),
        (args.det_dir, ocr, "det_model_dir"),
        (args.rec_dir, ocr, "rec_model_dir"),
        (args.cls_dir, ocr, "cls_model_dir"),
        (args.base_url, llm, "base_url"),
        (args.model, llm, "model"),
        (args.image_max_side, llm, "image_max_side"),
    ):
        if value is not None:
            setattr(target, attr, value)

    if args.cpu:
        ocr.use_gpu = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2

    apply_cli_overrides(config, args)

    if args.print_config:
        print(describe(config))
        return 0

    if not args.images:
        print("입력 이미지를 지정하십시오. (--help 로 사용법 확인)", file=sys.stderr)
        return 2

    print(describe(config))
    for problem in config.pipeline.ocr.validate_offline():
        logging.warning("%s", problem)

    out_dir = Path(config.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PiiPipeline(config.pipeline)
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
            write_image=config.output.write_image,
            include_ocr=config.output.include_ocr,
            font_path=config.output.font_path,
            show_ocr_boxes=config.output.show_ocr_boxes,
        )

        print()
        print(format_summary(result, written))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
