"""출력 계약 및 라벨 스키마.

이 모듈은 파이프라인 전체가 공유하는 자료구조를 정의한다.
다운스트림(마스킹/치환 모듈, 학습데이터 빌더)은 ``PageResult`` 만 알면 된다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------
# 라벨 스키마 (닫힌 집합)
# --------------------------------------------------------------------------

#: 탐지 대상 개인정보 라벨 (18종).
#:
#: 여기를 고치면 프롬프트와 guided-decoding 스키마가 자동으로 따라간다.
#: 항목을 늘릴 때 정형 식별자라면 ``RULE_LABELS`` 에도 추가하고
#: ``rules/detectors.py`` 에 패턴을 넣어야 한다.
PII_LABELS: tuple[str, ...] = (
    # ── 핵심 9종 ────────────────────────────────────────────────
    "NAME",            # 이름
    "RRN",             # 주민등록번호
    "ADDRESS",         # 주소
    "EMAIL",           # 이메일
    "IP",              # IP 주소
    "ACCOUNT_NO",      # 계좌번호
    "CARD_NO",         # 카드번호
    "PHONE",           # 전화번호
    "PASSPORT",        # 여권번호
    # ── 추가 9종 ────────────────────────────────────────────────
    "FOREIGN_ID",      # 외국인등록번호
    "DRIVER_LICENSE",  # 운전면허번호
    "BIZ_NO",          # 사업자등록번호
    "CORP_NO",         # 법인등록번호
    "BIRTH",           # 생년월일
    "ORG",             # 소속/직장명
    "TITLE",           # 직위/직책
    "SIGNATURE",       # 서명/인영
    "OTHER",           # 위에 없는 개인식별정보
)

#: 규칙 레이어(정규식+체크섬)로 확정 가능한 라벨.
#: LLM 은 이 라벨들을 다시 판단하지 않는다.
RULE_LABELS: frozenset[str] = frozenset(
    {
        "RRN",
        "FOREIGN_ID",
        "PASSPORT",
        "DRIVER_LICENSE",
        "BIZ_NO",
        "CORP_NO",
        "CARD_NO",
        "PHONE",
        "EMAIL",
        "IP",
        "ACCOUNT_NO",
    }
)

# LLM/VLM 이 문맥으로 판단해야 하는 라벨.
CONTEXT_LABELS: tuple[str, ...] = tuple(
    lbl for lbl in PII_LABELS if lbl not in RULE_LABELS
)


class Source(str, Enum):
    """탐지 출처. 학습데이터 구축 시 티어 필터링에 사용한다."""

    RULE = "rule"                    # 정규식 + 체크섬 통과. 신뢰도 최상.
    LLM_PASS1 = "llm_pass1"          # OCR 텍스트 기반 분류.
    VLM_PASS2 = "vlm_pass2"          # 이미지 검수 pass 에서 회수. OCR 박스 좌표 있음.
    VLM_GROUNDING = "vlm_grounding"  # OCR det 도 놓쳐 VLM 좌표를 그대로 쓴 경우. 좌표 부정확.
    PROPAGATED = "propagated"        # 다른 박스에서 확정된 값과 같아서 전파된 경우.


class OcrStatus(str, Enum):
    OK = "ok"              # rec 성공, 신뢰도 충분
    LOW_CONF = "low_conf"  # rec 성공했으나 신뢰도 낮음 (오독 가능)
    FAILED = "failed"      # det 는 됐으나 rec 실패 (손글씨/도장 등)


# --------------------------------------------------------------------------
# OCR 박스
# --------------------------------------------------------------------------

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2) 축정렬, 원본 픽셀 좌표


@dataclass
class OcrBox:
    """OCR 검출 단위. rec 실패 박스도 버리지 않고 여기에 담는다."""

    index: int
    bbox: BBox
    text: str
    status: OcrStatus = OcrStatus.OK
    rec_conf: float = 0.0
    det_conf: float = 0.0
    quad: list[tuple[int, int]] | None = None
    row: int = -1  # 행 클러스터링 결과 (읽기 순서용)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    def norm_xy(self, page_w: int, page_h: int) -> tuple[float, float]:
        """프롬프트에 넣을 정규화 좌상단 좌표."""
        if page_w <= 0 or page_h <= 0:
            return (0.0, 0.0)
        return (round(self.bbox[0] / page_w, 3), round(self.bbox[1] / page_h, 3))

    def norm_bbox(self, page_w: int, page_h: int) -> tuple[float, float, float, float]:
        """프롬프트에 넣을 정규화 박스 전체 좌표 ``(x1,y1,x2,y2)``.

        좌상단만 주면 모델이 박스의 **폭을 알 수 없어** 인접 여부를 판단할 수
        없다 (긴 주소 박스의 오른쪽 끝이 어디인지 모른다). 그룹화 규칙을
        지키게 하려면 범위를 줘야 한다.
        """
        if page_w <= 0 or page_h <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            round(self.bbox[0] / page_w, 3),
            round(self.bbox[1] / page_h, 3),
            round(self.bbox[2] / page_w, 3),
            round(self.bbox[3] / page_h, 3),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# --------------------------------------------------------------------------
# 탐지 결과
# --------------------------------------------------------------------------


@dataclass
class PiiRegion:
    """개인정보 영역 하나. 다운스트림 마스킹 모듈이 소비하는 단위."""

    id: str
    type: str
    bbox: BBox
    source: Source
    confidence: float
    text: str | None = None
    member_boxes: list[BBox] = field(default_factory=list)
    member_index: list[int] = field(default_factory=list)
    #: 규칙 레이어 탐지 시, 박스 텍스트 내 문자 오프셋.
    #: 부분 마스킹(예: ``901231-1******``)을 구현할 다운스트림이 사용한다.
    #: ``bbox`` 는 항상 박스 전체 영역임에 유의.
    char_span: tuple[int, int] | None = None
    ocr_status: OcrStatus = OcrStatus.OK
    coarse: bool = False         # 좌표가 근사치인가 (VLM grounding)
    needs_review: bool = False   # 사람 검토 필요
    low_confidence: bool = False
    reason: str | None = None    # pass2 가 남긴 근거 문구

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        d["ocr_status"] = self.ocr_status.value
        d["bbox"] = list(self.bbox)
        d["member_boxes"] = [list(b) for b in self.member_boxes]
        return d


@dataclass
class PageResult:
    """페이지 1장의 최종 산출물."""

    image_path: str
    width: int
    height: int
    #: 다중 페이지 문서(PDF)의 1-기반 페이지 번호. 단일 이미지면 ``None``.
    page_no: int | None = None
    regions: list[PiiRegion] = field(default_factory=list)
    ocr_boxes: list[OcrBox] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    raw_llm: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    #: 전처리된 페이지 이미지 (BGR numpy). 박스 오버레이를 그릴 때만 쓴다.
    #: **직렬화되지 않으며 일시적이다.** 결과 좌표는 이 이미지 기준이므로
    #: 오버레이를 다시 그리려면 같은 전처리 설정으로 재처리해야 한다.
    #: 배치 처리 시 메모리를 잡아먹으므로 저장 후 ``release_image()`` 로 해제한다.
    image: Any = field(default=None, repr=False, compare=False)

    def release_image(self) -> None:
        """전처리 이미지 참조를 해제한다 (배치 처리 메모리 관리용)."""
        self.image = None

    def to_dict(self, include_ocr: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "image_path": self.image_path,
            "page_no": self.page_no,
            "page": {"width": self.width, "height": self.height},
            "regions": [r.to_dict() for r in self.regions],
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "warnings": self.warnings,
            "stats": self.stats(),
        }
        if include_ocr:
            out["ocr_boxes"] = [b.to_dict() for b in self.ocr_boxes]
            out["raw_llm"] = self.raw_llm
        return out

    def stats(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in self.regions:
            by_source[r.source.value] = by_source.get(r.source.value, 0) + 1
            by_type[r.type] = by_type.get(r.type, 0) + 1
        return {
            "n_ocr_boxes": len(self.ocr_boxes),
            "n_regions": len(self.regions),
            "n_needs_review": sum(1 for r in self.regions if r.needs_review),
            "by_source": by_source,
            "by_type": by_type,
        }

    def to_json(self, include_ocr: bool = False, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(include_ocr=include_ocr), ensure_ascii=False, indent=indent
        )


# --------------------------------------------------------------------------
# LLM guided-decoding 스키마
# --------------------------------------------------------------------------


def _region_item_schema(label_enum: Sequence[str], with_reason: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "idx": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 1,
            "maxItems": 12,
        },
        "type": {"type": "string", "enum": list(label_enum)},
        "conf": {"type": "number", "minimum": 0, "maximum": 1},
    }
    required = ["idx", "type", "conf"]
    if with_reason:
        props["reason"] = {"type": "string", "maxLength": 120}
        required.append("reason")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


#: pass-1 (OCR 텍스트만) 출력 스키마
PASS1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "maxItems": 128,
            "items": _region_item_schema(CONTEXT_LABELS, with_reason=False),
        }
    },
    "required": ["regions"],
    "additionalProperties": False,
}

#: pass-2 (이미지 검수) 출력 스키마.
#: ``idx`` 는 OCR 박스 번호. det 도 놓친 영역은 ``bbox_norm`` 으로 받는다.
#:
#: ``idx`` 와 ``bbox_norm`` 은 둘 다 optional 이다 — 항목마다 하나만 채우기
#: 때문이다. 다만 **둘 다 비면 좌표를 알 수 없어 그 탐지는 버려진다**
#: (``merge.regions_from_pass2``). JSON Schema 로 "정확히 하나"를 강제하려면
#: ``anyOf`` 가 필요한데 guided-decoding 백엔드 지원이 불확실하므로,
#: ``idx`` 에 ``minItems`` 를 걸어 빈 배열만 막고 나머지는 프롬프트로 지시한다.
PASS2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "missed": {
            "type": "array",
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "idx": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 1,
                        "maxItems": 12,
                    },
                    "bbox_norm": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "type": {"type": "string", "enum": list(PII_LABELS)},
                    "conf": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "maxLength": 120},
                },
                "required": ["type", "conf", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["missed"],
    "additionalProperties": False,
}
