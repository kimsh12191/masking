#!/usr/bin/env python3
"""폐쇄망 반입용 PaddleOCR 모델 사전 다운로드.

PaddleOCR 은 첫 실행 시 모델을 자동 다운로드한다. 폐쇄망에서는 여기서 실패하므로
**외부망 장비에서 이 스크립트를 먼저 실행**하고 결과 디렉터리를 반입해야 한다.

절차:

    # ① 외부망 장비에서
    python scripts/download_models.py --out ./ocr_models
    tar czf ocr_models.tar.gz ocr_models/

    # ② 폐쇄망으로 ocr_models.tar.gz 반입 후
    tar xzf ocr_models.tar.gz
    export PII_OCR_DET_DIR=$PWD/ocr_models/det
    export PII_OCR_REC_DIR=$PWD/ocr_models/rec
    export PII_OCR_CLS_DIR=$PWD/ocr_models/cls

모델 URL 을 하드코딩하지 않고 PaddleOCR 자체의 다운로드 경로를 재사용한다.
버전에 따라 URL 이 바뀌므로 이 방식이 더 안전하다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_cached_dirs() -> dict[str, Path]:
    """PaddleOCR 이 모델을 내려받은 캐시 디렉터리를 찾는다."""
    roots = [
        Path.home() / ".paddleocr",
        Path.home() / ".paddlex",  # PaddleOCR 3.x
    ]
    found: dict[str, Path] = {}

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            # 추론 모델 디렉터리는 inference 파일 쌍을 갖는다
            has_model = any(path.glob("inference.pdmodel")) or any(
                path.glob("inference.json")
            )
            if not has_model:
                continue
            name = path.name.lower()
            parent = path.parent.name.lower()
            key = None
            if "det" in name or "det" in parent:
                key = "det"
            elif "rec" in name or "rec" in parent:
                key = "rec"
            elif "cls" in name or "cls" in parent:
                key = "cls"
            if key and key not in found:
                found[key] = path
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="ocr_models", help="모델을 모을 디렉터리")
    parser.add_argument("--lang", default="korean", help="인식 언어 (기본: korean)")
    args = parser.parse_args(argv)

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print(
            "paddleocr 가 설치되어 있지 않습니다.\n"
            "  pip install -r requirements.txt\n"
            "버전을 반드시 핀 고정하십시오 (2.x 와 3.x 의 API 가 다릅니다).",
            file=sys.stderr,
        )
        return 1

    print(f"PaddleOCR 초기화 중 (lang={args.lang}) — 모델을 다운로드합니다...")
    # 초기화만 해도 det/rec/cls 모델이 캐시로 내려온다
    PaddleOCR(lang=args.lang, use_angle_cls=True, use_gpu=False, show_log=False)
    print("다운로드 완료. 캐시 디렉터리를 탐색합니다.")

    found = find_cached_dirs()
    if not found:
        print(
            "캐시에서 모델 디렉터리를 찾지 못했습니다.\n"
            "  ~/.paddleocr 또는 ~/.paddlex 를 직접 확인해 수동으로 복사하십시오.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, src in sorted(found.items()):
        dst = out_dir / key
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"  {key:4s} <- {src}  ({size_mb:.1f} MB)")

    missing = {"det", "rec"} - set(found)
    if missing:
        print(f"\n경고: {', '.join(sorted(missing))} 모델을 찾지 못했습니다.", file=sys.stderr)

    print(f"\n반입 준비 완료: {out_dir}")
    print("\n폐쇄망에서 다음 환경변수를 설정하십시오:")
    for key in sorted(found):
        print(f"  export PII_OCR_{key.upper()}_DIR={out_dir.resolve()}/{key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
