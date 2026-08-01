"""설정 파일 로딩.

설정은 네 단계로 겹쳐 적용된다. 뒤쪽이 앞쪽을 덮는다.

    ① 코드 기본값  (각 dataclass 의 default)
    ② 설정 파일    (config.yaml)
    ③ 환경변수     (PII_* )
    ④ CLI 플래그   (scripts/run.py)

폐쇄망 배포에서는 ②에 환경 고유값(모델 경로, 엔드포인트)을 적어 반입하고,
일회성 변경만 ③④로 처리하는 것을 권한다.

**오타난 설정 키는 조용히 무시하지 않고 예외를 던진다.** 폐쇄망에서 설정이
반영되지 않은 채 도는 것이 가장 찾기 어려운 실패다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .detect import DetectConfig
from .llm.client import LlmConfig
from .locate import LocateConfig
from .ocr.paddle_runner import OcrConfig, parse_gpu_id
from .pipeline import PipelineConfig
from .verify import VerifyConfig

log = logging.getLogger(__name__)

#: 설정 파일 탐색 순서 (앞에서 먼저 찾은 것을 쓴다)
SEARCH_PATHS = ("config.yaml", "config.yml", "config/default.yaml")

#: 환경변수 이름 -> (섹션, 필드)
ENV_MAP: dict[str, tuple[str, str]] = {
    "PII_OCR_DET_DIR": ("ocr", "det_model_dir"),
    "PII_OCR_REC_DIR": ("ocr", "rec_model_dir"),
    "PII_OCR_CLS_DIR": ("ocr", "cls_model_dir"),
    "PII_OCR_GPU_ID": ("ocr", "gpu_id"),
    "PII_LLM_BASE_URL": ("llm", "base_url"),
    "PII_LLM_MODEL": ("llm", "model"),
    "PII_LLM_API_KEY": ("llm", "api_key"),
}


@dataclass
class OutputConfig:
    """산출물 설정.

    Attributes:
        out_dir: 출력 디렉터리.
        write_image: 박스 표시 이미지를 남길지 여부.
        include_ocr: JSON 에 OCR 박스와 LLM 원시응답까지 포함할지 (디버깅).
        font_path: 오버레이 한글 폰트 경로. 없어도 라벨(ASCII)은 표시된다.
        show_ocr_boxes: OCR 박스 전체를 회색으로 함께 표시.
    """

    out_dir: str = "out"
    write_image: bool = True
    include_ocr: bool = False
    font_path: str | None = None
    show_ocr_boxes: bool = False


@dataclass
class AppConfig:
    """파이프라인 + 산출물 설정 묶음."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    #: 실제로 읽은 설정 파일 경로 (없으면 None). 로그/감사용.
    source_path: Path | None = None


# --------------------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------------------


def _field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _apply(target: Any, values: dict[str, Any], section: str) -> None:
    """dataclass 인스턴스에 dict 값을 덮어쓴다. 모르는 키는 예외."""
    allowed = _field_names(type(target))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"설정 섹션 '{section}' 에 알 수 없는 키가 있습니다: {', '.join(unknown)}\n"
            f"  사용 가능한 키: {', '.join(sorted(allowed))}"
        )
    for key, value in values.items():
        setattr(target, key, value)


def find_config(explicit: str | Path | None = None) -> Path | None:
    """설정 파일 경로를 결정한다.

    Args:
        explicit: 명시적으로 지정된 경로. 이 파일이 없으면 예외를 던진다
            (사용자가 지정한 파일이 없는 것은 조용히 넘길 일이 아니다).

    Returns:
        찾은 경로, 또는 없으면 ``None``.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
        return path

    env_path = os.getenv("PII_CONFIG")
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"PII_CONFIG 가 가리키는 설정 파일이 없습니다: {path}"
            )
        return path

    for candidate in SEARCH_PATHS:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def read_yaml(path: Path) -> dict[str, Any]:
    """YAML 설정 파일을 읽는다."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "설정 파일을 읽으려면 PyYAML 이 필요합니다 (pip install PyYAML). "
            "설정 파일 없이 CLI 플래그만으로도 실행할 수 있습니다."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일 최상위는 매핑이어야 합니다: {path}")
    return data


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------

#: YAML 최상위 섹션. ``ocr``/``llm``/``detect``/``locate``/``verify`` 는 코드상
#: ``PipelineConfig`` 안에 있지만, 설정 파일에서는 평평하게 두는 편이 읽기 쉬워
#: 최상위로 뺀다. 섹션 이름이 파이프라인 단계 이름과 1:1 이다.
SECTIONS = ("pipeline", "ocr", "llm", "detect", "locate", "verify", "output")


def load_config(path: str | Path | None = None, use_env: bool = True) -> AppConfig:
    """설정 파일과 환경변수를 읽어 ``AppConfig`` 를 만든다.

    Args:
        path: 설정 파일 경로. 생략하면 ``SEARCH_PATHS`` 순서로 탐색한다.
        use_env: 환경변수 적용 여부.

    Returns:
        조립된 설정. 설정 파일이 없어도 코드 기본값으로 동작한다.

    Raises:
        FileNotFoundError: 명시한 설정 파일이 없을 때.
        ValueError: 알 수 없는 설정 키가 있을 때.
    """
    config = AppConfig(
        pipeline=PipelineConfig(
            ocr=OcrConfig(),
            llm=LlmConfig(),
            detect=DetectConfig(),
            locate=LocateConfig(),
            verify=VerifyConfig(),
        ),
        output=OutputConfig(),
    )

    # ② 설정 파일
    config_path = find_config(path)
    if config_path is not None:
        data = read_yaml(config_path)
        unknown = sorted(set(data) - set(SECTIONS))
        if unknown:
            raise ValueError(
                f"설정 파일에 알 수 없는 최상위 섹션이 있습니다: {', '.join(unknown)}\n"
                f"  사용 가능한 섹션: {', '.join(SECTIONS)}"
            )
        targets = {
            "pipeline": config.pipeline,
            "ocr": config.pipeline.ocr,
            "llm": config.pipeline.llm,
            "detect": config.pipeline.detect,
            "locate": config.pipeline.locate,
            "verify": config.pipeline.verify,
            "output": config.output,
        }
        for section in SECTIONS:
            values = data.get(section) or {}
            if not isinstance(values, dict):
                raise ValueError(f"설정 섹션 '{section}' 은 매핑이어야 합니다")
            _apply(targets[section], values, section)
        config.source_path = config_path
        log.info("설정 파일 적용: %s", config_path)

    # ③ 환경변수
    if use_env:
        targets = {"ocr": config.pipeline.ocr, "llm": config.pipeline.llm}
        for env_name, (section, field_name) in ENV_MAP.items():
            value = os.getenv(env_name)
            if value:
                setattr(targets[section], field_name, value)
                log.debug("환경변수 적용: %s -> %s.%s", env_name, section, field_name)

    # 환경변수와 YAML 은 "1" 같은 문자열을 줄 수 있다. GPU 번호는 정수로
    # 넘겨야 PaddleOCR 이 받으므로 마지막에 한 번 정규화한다.
    ocr = config.pipeline.ocr
    try:
        ocr.gpu_id = parse_gpu_id(ocr.gpu_id)
    except ValueError as exc:
        raise ValueError(f"ocr.gpu_id 설정이 잘못되었습니다 ({exc})") from exc

    return config


def grid_report(config: AppConfig) -> str:
    """패치 격자와 어긋난 설정값을 찾아 한 줄로 알려준다.

    **격자에 맞지 않으면 서버가 조각을 다시 리샘플한다.** 10px 급 한글이 보간으로
    뭉개지고 그건 곧 미탐이다. 그런데 이 어긋남은 실행해도 아무 에러가 없고
    결과만 조금 나빠지므로, 설정을 찍을 때 검산해 주지 않으면 아무도 모른다.

    ``target_long_side`` 가 배수가 아니어도 전처리가 여백을 붙여 맞추기는 한다.
    다만 그건 보정이고, 처음부터 배수로 두는 것이 여백 없이 깔끔하다.

    Args:
        config: 검사할 설정.

    Returns:
        전부 맞으면 ``"OK ..."``, 아니면 어긋난 항목 목록.
    """
    f = config.pipeline.detect.image_factor
    if f <= 1:
        return "image_factor=1 — 격자 정렬을 쓰지 않는다"

    bad: list[str] = []
    canvas = config.pipeline.canvas
    if canvas:
        # 캔버스는 전처리가 여백으로 보정해 주지 않는다 (크기가 고정이라 붙일
        # 자리가 없다). 여기서 안 맞으면 조각도 격자에서 벗어난다.
        for name, value in (("폭", int(canvas[0])), ("높이", int(canvas[1]))):
            if value % f:
                bad.append(f"canvas {name}={value} (÷{f} 아님, {(value // f) * f} 권장)")
    else:
        long_side = config.pipeline.target_long_side
        if long_side and long_side % f:
            near = (long_side // f) * f
            bad.append(f"target_long_side={long_side} (÷{f} 아님, {near} 권장)")
    if config.pipeline.llm.image_max_side % f:
        near = (config.pipeline.llm.image_max_side // f) * f
        bad.append(f"image_max_side={config.pipeline.llm.image_max_side} (÷{f} 아님, {near} 권장)")
    if config.pipeline.locate.min_pad_px < f:
        bad.append(
            f"min_pad_px={config.pipeline.locate.min_pad_px} < {f} "
            f"(모델 grounding 하한보다 작다, {f * 2} 권장)"
        )
    if bad:
        return "어긋남 — " + "; ".join(bad)
    return f"OK — 모두 {f}px 격자에 맞음 (조각 리샘플 없음)"


def describe(config: AppConfig) -> str:
    """현재 적용된 설정 요약 (실행 로그에 남길 용도).

    ``image_max_side`` 와 타일 수를 **같은 줄에** 적는다. 둘의 조합이 VLM 이
    실제로 보는 글자 크기를 결정하는데, 따로 적어 두면 한쪽만 바꿔 놓고
    "왜 작은 글씨를 못 읽지" 로 헤매게 된다.
    """
    llm, ocr, pipe, out = (
        config.pipeline.llm,
        config.pipeline.ocr,
        config.pipeline,
        config.output,
    )
    # canvas 를 쓰면 페이지 크기가 고정이므로 타일 크기를 **추정이 아니라
    # 실측**할 수 있다. tile_rects 를 그대로 불러서 검산한다 — 여기서 손으로
    # 나눈 값을 보여주면 실제와 다른 숫자를 확인하고 넘어가게 된다.
    if pipe.canvas:
        from .detect import tile_rects

        canvas_w, canvas_h = int(pipe.canvas[0]), int(pipe.canvas[1])
        long_side = max(canvas_w, canvas_h)
        rects = tile_rects(
            canvas_w, canvas_h, pipe.detect.tiles, pipe.detect.overlap,
            pipe.detect.image_factor,
        )
        vertical = canvas_h >= canvas_w
        tile_side = (
            round((rects[0][3] - rects[0][1]) * canvas_h)
            if vertical
            else round((rects[0][2] - rects[0][0]) * canvas_w)
        )
        page_report = f"고정 캔버스 {canvas_w}x{canvas_h}"
    else:
        long_side = pipe.target_long_side or 0
        tile_side = int(long_side / max(1, pipe.detect.tiles)) if long_side else 0
        page_report = f"긴 변 {pipe.target_long_side or '원본'} (페이지 크기 가변)"
    return "\n".join(
        [
            f"설정 파일   {config.source_path or '(없음 — 기본값 사용)'}",
            f"모델        {llm.model} @ {llm.base_url}",
            f"OCR         lang={ocr.lang} det={ocr.det_model_dir or '(자동)'} "
            f"rec={ocr.rec_model_dir or '(자동)'}",
            f"OCR 장치    {f'GPU {ocr.gpu_id} (상한 {ocr.gpu_mem}MB)' if ocr.use_gpu else 'CPU'}"
            + (
                f"  CUDA_VISIBLE_DEVICES={visible}"
                if ocr.use_gpu and (visible := os.getenv('CUDA_VISIBLE_DEVICES'))
                else ""
            ),
            f"② VLM 탐지  타일 {pipe.detect.tiles}개 (겹침 {pipe.detect.overlap:.0%})"
            + (
                f" × 샘플 {pipe.detect.samples}회"
                f" (T={pipe.detect.sample_temperature}, 합집합)"
                if pipe.detect.samples > 1
                else ""
            )
            + f"  호출 {pipe.detect.tiles * max(1, pipe.detect.samples)}회"
            f"  image_max_side={llm.image_max_side}"
            + (
                f"  타일 {tile_side}px"
                + (" — 축소 없음" if tile_side and tile_side <= llm.image_max_side
                   else " — 축소 발생, 타일을 늘리거나 image_max_side 를 올릴 것")
                if tile_side
                else ""
            ),
            "   좌표 규약  bbox_2d "
            + (
                "0~1000 native (응답마다 자동 판정)"
                if pipe.detect.coord_convention == "auto"
                else f"{pipe.detect.coord_convention} 고정"
            )
            + f"  image_factor={pipe.detect.image_factor}"
            + (
                f"  max_pixels={pipe.detect.max_pixels}"
                if pipe.detect.max_pixels
                else "  max_pixels=모델 기본값"
            ),
            "   격자 정합  " + grid_report(config),
            f"③ 좌표 확정  크롭 패딩 {pipe.locate.pad_ratio:.0%}"
            f" (최소 {pipe.locate.min_pad_px}px)"
            f"  업샘플 x{pipe.locate.upscale}"
            f"  기하 fallback={'ON' if pipe.locate.geometry_fallback else 'OFF'}",
            f"④ 검증      체크섬 교정={'ON' if pipe.verify.retype_on_checksum else 'OFF'}",
            f"전처리      {page_report}"
            f"  deskew={'ON' if pipe.deskew else 'OFF'}",
            f"출력        {out.out_dir}  이미지={'ON' if out.write_image else 'OFF'}",
        ]
    )
