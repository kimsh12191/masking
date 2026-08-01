#!/usr/bin/env python3
"""파이프라인 실행 CLI.

이미지 또는 PDF 를 입력하면 페이지당 두 파일을 남긴다.

    단일 이미지   {이름}.boxes.png       {이름}.json
    PDF           {이름}_p001.boxes.png  {이름}_p001.json
                  {이름}_p002.boxes.png  {이름}_p002.json  ...

설정은 네 단계로 겹쳐 적용된다 (뒤가 앞을 덮는다):

    코드 기본값 → config.yaml → 환경변수(PII_*) → CLI 플래그

사용 예:

    # 설정 파일만으로 실행 (config.yaml 또는 config/default.yaml 자동 탐색)
    python scripts/run.py sample.png

    # 설정 파일 지정
    python scripts/run.py sample.png --config config/prod.yaml

    # PDF — 페이지별로 png + json 이 나온다
    python scripts/run.py 계약서.pdf -o out/

    # PDF 페이지 범위 지정
    python scripts/run.py 계약서.pdf -o out/ --pages 1-3,7

    # 이미지와 PDF 를 섞어서 일괄 처리
    python scripts/run.py data/*.png data/*.pdf -o out/

    # 일회성 변경 — 설정 파일을 고치지 않고 덮어쓰기
    python scripts/run.py sample.png --model Qwen/Qwen3.5-9B --tiles 4

    # 적용된 설정만 확인하고 종료 (타일 수와 image_max_side 의 조합을 검산해준다)
    python scripts/run.py --print-config

    # 디버깅 — VLM 원본 판단과 크롭 OCR 박스를 JSON 에 함께 남긴다
    python scripts/run.py sample.png -o out/ --include-ocr --show-ocr-boxes
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pii_pipeline import PiiPipeline  # noqa: E402
from pii_pipeline.config import describe, load_config  # noqa: E402
from pii_pipeline.output import (  # noqa: E402
    IMAGE_SUFFIX,
    JSON_SUFFIX,
    default_stem,
    format_summary,
)
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
    p.add_argument("inputs", nargs="*", help="입력 이미지 또는 PDF 경로")
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

    g = p.add_argument_group("PDF 입력")
    g.add_argument("--pages", default=None,
                   help='처리할 페이지 범위. 예: "1-3,7", "5-" (기본: 전체)')
    g.add_argument("--pdf-password", default=None,
                   help="암호화된 PDF 의 열기 암호 "
                        "(셸 히스토리에 남으므로 PII_PDF_PASSWORD 환경변수를 권장)")

    g = p.add_argument_group("파이프라인")
    g.add_argument("--deskew", action=BOOL, default=None, help="기울기 보정")
    g.add_argument("--long-side", type=int, default=None, help="전처리 긴 변 길이")

    g = p.add_argument_group("② VLM 탐지")
    g.add_argument("--tiles", type=int, default=None,
                   help="페이지를 몇 조각으로 나눠 VLM 을 호출할지 (기본 3). "
                        "밀집한 표 문서는 3~4. 1 이면 전체를 한 번에 (뒤쪽 항목이 잘린다)")
    g.add_argument("--tile-overlap", type=float, default=None,
                   help="조각 간 겹침 비율 (기본 0.08). 경계에 걸린 줄을 보호한다")
    g.add_argument("--hint", default=None,
                   help="문서 종류 힌트. 비워 두는 것이 기본이다 (prefix caching 유지)")

    g = p.add_argument_group("③ 좌표 확정")
    g.add_argument("--crop-pad", type=float, default=None,
                   help="크롭 여유 비율 (기본 0.35). VLM 좌표가 어긋나므로 넉넉히")
    g.add_argument("--crop-upscale", type=float, default=None,
                   help="크롭 업샘플 배율 (기본 2.0). 작은 글씨 대응. 1.0 이면 끈다")
    g.add_argument("--similarity", type=float, default=None,
                   help="유사 매칭 임계값 (기본 0.65). 1.0 이면 완전일치만")
    g.add_argument("--geometry-fallback", action=BOOL, default=None,
                   help="텍스트가 안 맞을 때 위치 겹침으로 박스를 고른다 (기본 ON). "
                        "--no-geometry-fallback 이면 텍스트 불일치 = 좌표 포기. A/B 비교용")

    g = p.add_argument_group("OCR")
    g.add_argument("--det-dir", default=None, help="검출 모델 디렉터리 (폐쇄망 필수)")
    g.add_argument("--rec-dir", default=None, help="인식 모델 디렉터리 (폐쇄망 필수)")
    g.add_argument("--cls-dir", default=None, help="방향분류 모델 디렉터리")
    g.add_argument("--gpu-id", type=int, default=None,
                   help="OCR 에 쓸 GPU 번호 (기본 0). vLLM 이 0 번을 쓰면 1 등으로 옮긴다. "
                        "CUDA_VISIBLE_DEVICES 가 있으면 그 목록 안의 상대 번호")
    g.add_argument("--cpu", action="store_true", help="OCR 을 CPU 로 실행")

    g = p.add_argument_group("LLM")
    g.add_argument("--base-url", default=None, help="vLLM 엔드포인트")
    g.add_argument("--model", default=None, help="모델 이름")
    g.add_argument("--image-max-side", type=int, default=None,
                   help="VLM 에 보낼 이미지 긴 변 길이. --tiles 와 함께 볼 것 "
                        "(--print-config 가 조합을 검산해준다)")
    return p


def written_paths(out_dir: Path, result) -> dict[str, Path]:
    """이미 저장된 파일 경로를 요약 출력용으로 재구성한다.

    저장은 ``run_any(out_dir=...)`` 안에서 끝났으므로 실제 존재하는 것만 담는다.
    """
    stem = default_stem(result)
    found: dict[str, Path] = {}
    for key, suffix in (("image", IMAGE_SUFFIX), ("json", JSON_SUFFIX)):
        path = out_dir / f"{stem}{suffix}"
        if path.exists():
            found[key] = path
    return found


def apply_cli_overrides(config, args: argparse.Namespace) -> None:
    """``None`` 이 아닌 CLI 인자만 설정에 덮어쓴다 (④ 단계)."""
    pipe = config.pipeline
    ocr, llm, out = pipe.ocr, pipe.llm, config.output
    det, loc = pipe.detect, pipe.locate

    for value, target, attr in (
        (args.out, out, "out_dir"),
        (args.image, out, "write_image"),
        (args.font, out, "font_path"),
        (args.show_ocr_boxes, out, "show_ocr_boxes"),
        (args.include_ocr, out, "include_ocr"),
        (args.deskew, pipe, "deskew"),
        (args.long_side, pipe, "target_long_side"),
        (args.det_dir, ocr, "det_model_dir"),
        (args.rec_dir, ocr, "rec_model_dir"),
        (args.cls_dir, ocr, "cls_model_dir"),
        (args.gpu_id, ocr, "gpu_id"),
        (args.base_url, llm, "base_url"),
        (args.model, llm, "model"),
        (args.image_max_side, llm, "image_max_side"),
        (args.tiles, det, "tiles"),
        (args.tile_overlap, det, "overlap"),
        (args.hint, det, "hint"),
        (args.crop_pad, loc, "pad_ratio"),
        (args.crop_upscale, loc, "upscale"),
        (args.similarity, loc, "similarity"),
        (args.geometry_fallback, loc, "geometry_fallback"),
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

    if not args.inputs:
        print("입력 이미지 또는 PDF 를 지정하십시오. (--help 로 사용법 확인)",
              file=sys.stderr)
        return 2

    print(describe(config))
    for problem in config.pipeline.ocr.validate_offline():
        logging.warning("%s", problem)

    out_dir = Path(config.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PiiPipeline(config.pipeline)
    exit_code = 0

    password = args.pdf_password or os.getenv("PII_PDF_PASSWORD")

    for input_path in args.inputs:
        try:
            results = pipeline.run_any(
                input_path,
                out_dir=str(out_dir),
                pages=args.pages,
                password=password,
                write_image=config.output.write_image,
                include_ocr=config.output.include_ocr,
                font_path=config.output.font_path,
                show_ocr_boxes=config.output.show_ocr_boxes,
            )
        except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
            logging.exception("처리 실패: %s", input_path)
            exit_code = 1
            (out_dir / f"{Path(input_path).stem}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            continue

        for result in results:
            if any(w.startswith("처리 실패") for w in result.warnings):
                exit_code = 1
            print()
            print(format_summary(result, written_paths(out_dir, result)))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
