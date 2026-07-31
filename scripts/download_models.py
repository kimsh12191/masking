#!/usr/bin/env python3
"""폐쇄망 반입용 PaddleOCR 가중치 다운로드.

**인터넷이 되는 장비에서 실행하십시오.** 가중치는 전부 바이두 BOS
(``*.bj.bcebos.com``)에 있으며, 사내망/프록시 환경에서는 이 호스트가 막혀 있는
경우가 많습니다.

    # ① 외부망 장비에서
    python scripts/download_models.py --out ./ocr_models
    tar czf ocr_models.tar.gz ocr_models/

    # ② 폐쇄망에 반입 후 — 무결성 확인
    tar xzf ocr_models.tar.gz
    python scripts/download_models.py --verify ./ocr_models

    # ③ 경로 지정
    export PII_OCR_DET_DIR=$PWD/ocr_models/det
    export PII_OCR_REC_DIR=$PWD/ocr_models/rec
    export PII_OCR_CLS_DIR=$PWD/ocr_models/cls

다운로드 없이 목록만 보려면 (반입 신청서용):

    python scripts/download_models.py --list

가중치 합계는 약 122MB 다. 실제 반입 부담은 paddlepaddle 휠(~1GB)이 더 크다.
``--list`` 가 그 명령도 함께 출력한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "manifest.json"


@dataclass
class ModelSpec:
    """반입 대상 모델 하나.

    Attributes:
        key: 배치 디렉터리 이름 (det / rec / cls). ``OcrConfig`` 의 경로와 대응.
        title: 사람이 읽는 이름.
        approx_mb: 공식 문서 기준 대략 용량 (신청서 작성용).
        urls: 다운로드 URL 후보. 앞에서부터 시도한다.
            공식 문서의 링크가 틀린 경우가 있어 후보를 여러 개 둔다.
        note: 선택 이유 / 주의사항.
    """

    key: str
    title: str
    approx_mb: float
    urls: list[str]
    note: str = ""


#: 기본 조합 — 금융 서식 기준.
#:
#: det 는 **server 판**을 쓴다. 금융 서식은 작은 글씨와 촘촘한 표가 많고
#: 이 파이프라인에서 좌표 정확도가 근간이다. mobile 판(4.7MB)은 검출 누락이
#: 늘어나고, 검출이 놓치면 이미지 pass 도 회수할 수 없다.
MODELS: dict[str, ModelSpec] = {
    "det": ModelSpec(
        key="det",
        title="텍스트 검출 PP-OCRv4 server",
        approx_mb=110.0,
        urls=[
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
            "paddle3.0.0/PP-OCRv4_server_det_infer.tar",
        ],
        note="좌표 정확도가 파이프라인의 근간이므로 server 판 권장",
    ),
    "rec": ModelSpec(
        key="rec",
        title="한국어 인식 korean PP-OCRv3",
        approx_mb=11.0,
        urls=[
            "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar",
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
            "paddle3.0.0/korean_PP-OCRv3_mobile_rec_infer.tar",
        ],
        note="한국어는 v3 까지만 사전학습 추론 모델이 제공된다 (v4 는 train 모델만)",
    ),
    "cls": ModelSpec(
        key="cls",
        title="방향 분류 ch_ppocr_mobile_v2.0_cls",
        approx_mb=1.4,
        urls=[
            "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
        ],
        note="회전된 스캔 문서 보정용",
    ),
}

#: det 를 경량판으로 바꿀 때 쓰는 대안 (--mobile-det)
MOBILE_DET = ModelSpec(
    key="det",
    title="텍스트 검출 PP-OCRv4 mobile",
    approx_mb=4.7,
    urls=[
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
        "paddle3.0.0/PP-OCRv4_mobile_det_infer.tar",
    ],
    note="용량 우선. 작은 글씨 검출 누락이 늘어난다",
)

#: 추론 모델 디렉터리에 반드시 있어야 하는 파일 (형식은 Paddle 버전에 따라 다름)
MODEL_FILE_SETS = (
    ("inference.pdmodel", "inference.pdiparams"),
    ("inference.json", "inference.pdiparams"),
)

PIP_COMMANDS = """\
# paddlepaddle 휠 — CUDA 버전에 맞는 것을 받아야 한다 (가중치보다 훨씬 크다)
#   CUDA 11.8
pip download paddlepaddle-gpu==2.6.2.post118 -d wheels/ \\
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
#   CUDA 12.x
pip download paddlepaddle-gpu==2.6.2 -d wheels/ \\
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 파이프라인 의존성 (플랫폼이 다르면 --platform/--python-version 을 명시할 것)
pip download -r requirements.txt -d wheels/
"""


# --------------------------------------------------------------------------
# 헬퍼
# --------------------------------------------------------------------------


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def dir_digest(directory: Path) -> dict[str, str]:
    """디렉터리 내 파일별 sha256 (반입 후 무결성 확인용)."""
    return {
        f.relative_to(directory).as_posix(): sha256_of(f)
        for f in sorted(directory.rglob("*"))
        if f.is_file()
    }


def looks_like_model_dir(directory: Path) -> bool:
    names = {f.name for f in directory.iterdir() if f.is_file()}
    return any(set(required) <= names for required in MODEL_FILE_SETS)


def find_model_root(extracted: Path) -> Path:
    """tar 안에서 실제 모델 파일이 있는 디렉터리를 찾는다.

    tar 구조가 버전마다 달라(루트 직접 / 한 겹 감싸짐) 재귀 탐색한다.
    """
    if looks_like_model_dir(extracted):
        return extracted
    for candidate in sorted(p for p in extracted.rglob("*") if p.is_dir()):
        if looks_like_model_dir(candidate):
            return candidate
    raise RuntimeError(
        f"추출 결과에서 추론 모델 파일을 찾지 못했습니다: {extracted}\n"
        f"  기대한 파일 조합: {MODEL_FILE_SETS}"
    )


def download(url: str, dest: Path, timeout: float = 120.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def fetch_spec(spec: ModelSpec, out_dir: Path) -> dict[str, object]:
    """모델 하나를 받아 ``out_dir/<key>`` 에 배치하고 메타를 돌려준다."""
    target = out_dir / spec.key
    errors: list[str] = []

    for url in spec.urls:
        print(f"  [{spec.key}] {url}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "model.tar"
            try:
                download(url, archive)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                errors.append(f"{url} -> {type(exc).__name__}: {exc}")
                print(f"       실패, 다음 후보 시도: {exc}")
                continue

            archive_sha = sha256_of(archive)
            archive_mb = archive.stat().st_size / 1e6

            extracted = tmp_path / "x"
            extracted.mkdir()
            with tarfile.open(archive) as tf:
                # 경로 탈출 방어 (신뢰할 수 없는 아카이브 대비)
                for member in tf.getmembers():
                    resolved = (extracted / member.name).resolve()
                    if not str(resolved).startswith(str(extracted.resolve())):
                        raise RuntimeError(f"아카이브에 비정상 경로가 있습니다: {member.name}")
                tf.extractall(extracted)  # noqa: S202 - 위에서 검증함

            root = find_model_root(extracted)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(root, target)

        print(f"       OK  {archive_mb:.1f} MB -> {target}")
        return {
            "key": spec.key,
            "title": spec.title,
            "source_url": url,
            "archive_sha256": archive_sha,
            "archive_mb": round(archive_mb, 2),
            "files": dir_digest(target),
            "note": spec.note,
        }

    raise RuntimeError(
        f"[{spec.key}] 모든 후보 URL 이 실패했습니다.\n  " + "\n  ".join(errors) + "\n\n"
        "이 호스트(*.bj.bcebos.com)가 막혀 있으면 인터넷이 되는 다른 장비에서 "
        "실행하거나, 위 URL 을 브라우저로 직접 받아 --out 디렉터리에 풀어 넣으십시오."
    )


# --------------------------------------------------------------------------
# 명령
# --------------------------------------------------------------------------


def cmd_list(specs: dict[str, ModelSpec]) -> int:
    total = sum(s.approx_mb for s in specs.values())
    print("반입 대상 — PaddleOCR 가중치\n")
    print(f"{'구분':<6}{'모델':<34}{'용량':>9}")
    print("-" * 52)
    for spec in specs.values():
        print(f"{spec.key:<6}{spec.title:<34}{spec.approx_mb:>7.1f}MB")
    print("-" * 52)
    print(f"{'':<6}{'합계':<34}{total:>7.1f}MB\n")

    print("다운로드 URL")
    for spec in specs.values():
        print(f"  [{spec.key}] {spec.urls[0]}")
        for alt in spec.urls[1:]:
            print(f"        (대안) {alt}")
    print()
    print("주의사항")
    for spec in specs.values():
        if spec.note:
            print(f"  [{spec.key}] {spec.note}")
    print()
    print(PIP_COMMANDS)
    return 0


def cmd_download(specs: dict[str, ModelSpec], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"다운로드 시작 -> {out_dir}\n")

    entries: list[dict[str, object]] = []
    for spec in specs.values():
        try:
            entries.append(fetch_spec(spec, out_dir))
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1

    manifest = {
        "models": entries,
        "total_mb": round(sum(float(e["archive_mb"]) for e in entries), 2),
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n완료. 합계 {manifest['total_mb']} MB")
    print(f"매니페스트: {manifest_path}  (반입 후 --verify 로 무결성 확인)")
    print("\n폐쇄망에서 설정할 환경변수:")
    for spec in specs.values():
        print(f"  export PII_OCR_{spec.key.upper()}_DIR={out_dir.resolve()}/{spec.key}")
    return 0


def cmd_verify(out_dir: Path) -> int:
    """반입된 디렉터리를 매니페스트와 대조한다."""
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"매니페스트가 없습니다: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    for entry in manifest["models"]:
        key = str(entry["key"])
        target = out_dir / key
        print(f"[{key}] {target}")
        if not target.is_dir():
            failures.append(f"{key}: 디렉터리가 없습니다")
            continue

        expected: dict[str, str] = entry["files"]  # type: ignore[assignment]
        actual = dir_digest(target)

        for name, digest in expected.items():
            if name not in actual:
                failures.append(f"{key}/{name}: 파일 누락")
            elif actual[name] != digest:
                failures.append(f"{key}/{name}: 체크섬 불일치")
        for name in actual.keys() - expected.keys():
            print(f"       (매니페스트에 없는 파일: {name})")

        if not looks_like_model_dir(target):
            failures.append(f"{key}: 추론 모델 파일 조합이 없습니다")

    if failures:
        print("\n검증 실패:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("\n검증 통과. 모든 파일의 체크섬이 일치합니다.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="폐쇄망 반입용 PaddleOCR 가중치 다운로드/검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out", default="ocr_models", help="출력 디렉터리")
    parser.add_argument("--list", action="store_true",
                       help="다운로드하지 않고 목록·URL·용량만 출력 (반입 신청서용)")
    parser.add_argument("--verify", metavar="DIR", default=None,
                       help="반입된 디렉터리를 매니페스트와 대조")
    parser.add_argument("--mobile-det", action="store_true",
                       help="검출 모델을 경량판(4.7MB)으로 받는다. 작은 글씨 누락 증가")
    args = parser.parse_args(argv)

    specs = dict(MODELS)
    if args.mobile_det:
        specs["det"] = MOBILE_DET

    if args.verify:
        return cmd_verify(Path(args.verify))
    if args.list:
        return cmd_list(specs)
    return cmd_download(specs, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
