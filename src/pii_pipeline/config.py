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

from .llm.client import LlmConfig
from .ocr.paddle_runner import OcrConfig
from .pipeline import PipelineConfig
from .propagate import PropagateConfig

log = logging.getLogger(__name__)

#: 설정 파일 탐색 순서 (앞에서 먼저 찾은 것을 쓴다)
SEARCH_PATHS = ("config.yaml", "config.yml", "config/default.yaml")

#: 환경변수 이름 -> (섹션, 필드)
ENV_MAP: dict[str, tuple[str, str]] = {
    "PII_OCR_DET_DIR": ("ocr", "det_model_dir"),
    "PII_OCR_REC_DIR": ("ocr", "rec_model_dir"),
    "PII_OCR_CLS_DIR": ("ocr", "cls_model_dir"),
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

#: YAML 최상위 섹션. ``ocr``/``llm``/``propagate`` 는 코드상 ``PipelineConfig``
#: 안에 있지만, 설정 파일에서는 평평하게 두는 편이 읽기 쉬워 최상위로 뺀다.
SECTIONS = ("pipeline", "ocr", "llm", "propagate", "output")


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
            ocr=OcrConfig(), llm=LlmConfig(), propagate=PropagateConfig()
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
            "propagate": config.pipeline.propagate,
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

    return config


def describe(config: AppConfig) -> str:
    """현재 적용된 설정 요약 (실행 로그에 남길 용도)."""
    llm, ocr, pipe, out = (
        config.pipeline.llm,
        config.pipeline.ocr,
        config.pipeline,
        config.output,
    )
    return "\n".join(
        [
            f"설정 파일   {config.source_path or '(없음 — 기본값 사용)'}",
            f"모델        {llm.model} @ {llm.base_url}",
            f"OCR         lang={ocr.lang} det={ocr.det_model_dir or '(자동)'} "
            f"rec={ocr.rec_model_dir or '(자동)'}",
            f"pass2       {'ON' if pipe.enable_pass2 else 'OFF'}"
            f"{' (blind)' if pipe.pass2_blind else ''}"
            f"  image_max_side={llm.image_max_side}",
            f"값 전파     {'ON' if pipe.propagate.enabled else 'OFF'}"
            f"  min_similarity={pipe.propagate.min_similarity}",
            f"출력        {out.out_dir}  이미지={'ON' if out.write_image else 'OFF'}",
        ]
    )
