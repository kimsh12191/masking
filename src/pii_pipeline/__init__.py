"""금융 문서 개인정보 영역 탐지 파이프라인.

산출물은 **영역 좌표 + 개인정보 유형**이다. 마스킹/치환은 별도 모듈에서 수행한다.

설계 원칙:

1. **좌표는 OCR 에서, 판단만 LLM 이 한다.** LLM 에 좌표를 물으면 환각한다.
   OCR 박스에 번호를 매겨 넘기고 번호만 회수하므로 좌표는 픽셀 단위로 정확하다.
2. **규칙이 LLM 보다 먼저다.** 주민번호·카드번호는 체크섬으로 확정한다.
   가장 중요한 항목을 모델 운에 맡기지 않는다.
3. **OCR det/rec 을 분리한다.** 읽지 못한 박스도 번호를 부여해 유지하고,
   이미지 pass 에서 회수한다. 이래야 VLM 도 좌표를 발명할 필요가 없다.
4. **recall 우선.** 애매하면 포함시킨다. 제외 규칙은 ``exclusions.py`` 에 추가한다.
"""

from .config import AppConfig, OutputConfig, load_config
from .output import save_result
from .pipeline import PiiPipeline, PipelineConfig
from .schema import (
    CONTEXT_LABELS,
    PII_LABELS,
    RULE_LABELS,
    OcrBox,
    OcrStatus,
    PageResult,
    PiiRegion,
    Source,
)

__all__ = [
    "PiiPipeline",
    "PipelineConfig",
    "AppConfig",
    "OutputConfig",
    "load_config",
    "save_result",
    "PageResult",
    "PiiRegion",
    "OcrBox",
    "OcrStatus",
    "Source",
    "PII_LABELS",
    "RULE_LABELS",
    "CONTEXT_LABELS",
]

__version__ = "0.1.0"
