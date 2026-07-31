"""금융 문서 개인정보 영역 탐지 파이프라인.

산출물은 **영역 좌표 + 개인정보 유형**이다. 마스킹/치환은 별도 모듈에서 수행한다.

    이미지 → ② VLM 전사 → ③ 크롭 OCR 측량 → ④ 검증 → JSON + 오버레이 PNG

설계 원칙:

1. **판단과 좌표를 분리한다.** 개인정보 판단은 의미 문제이므로 VLM 이 하고,
   좌표는 기하 문제이므로 OCR 이 한다. 한 모델에게 둘을 동시에 요구하지 않는다.
2. **VLM 에게는 전사만 시킨다.** OCR 박스 목록을 주고 고르게 하거나, 이전
   단계가 놓친 것만 찾으라고 하지 않는다. 집합 차집합과 좌표 grounding 은
   LLM 이 가장 못하는 일이다.
3. **규칙은 탐지기가 아니라 검증기다.** 자리수·체크섬 지식은 형식표로 프롬프트에
   녹이고, 체크섬 자체는 탐지 후 검사에만 쓴다. 검증기는 틀린 답을 만들 수 없다.
4. **recall 우선, 단 무음 실패를 만들지 않는다.** 좌표를 못 잡은 항목도 버리지
   않고 ``vlm_coarse`` 로 남기고, 왜 못 잡았는지를 ``reason`` 에 적는다.
"""

from .config import AppConfig, OutputConfig, load_config
from .detect import DetectConfig
from .locate import LocateConfig
from .output import save_result
from .pipeline import PiiPipeline, PipelineConfig
from .schema import (
    PII_LABELS,
    TEXTLESS_LABELS,
    Agreement,
    OcrBox,
    OcrStatus,
    PageResult,
    PiiRegion,
    Source,
    VlmFinding,
)
from .verify import VerifyConfig

__all__ = [
    "PiiPipeline",
    "PipelineConfig",
    "DetectConfig",
    "LocateConfig",
    "VerifyConfig",
    "AppConfig",
    "OutputConfig",
    "load_config",
    "save_result",
    "PageResult",
    "PiiRegion",
    "VlmFinding",
    "OcrBox",
    "OcrStatus",
    "Source",
    "Agreement",
    "PII_LABELS",
    "TEXTLESS_LABELS",
]

__version__ = "0.2.0"
