"""출력 계약 및 라벨 스키마.

이 모듈은 파이프라인 전체가 공유하는 자료구조를 정의한다.
다운스트림(마스킹/치환 모듈, 학습데이터 빌더)은 ``PageResult`` 만 알면 된다.

파이프라인은 두 단계다. 자료구조도 그 두 단계를 그대로 반영한다.

    ① VLM      이미지를 보고 "무엇이 개인정보인가" 를 판단한다  ->  VlmFinding
    ② 크롭 OCR 그 위치를 크롭해 "정확히 어디인가" 를 확정한다   ->  PiiRegion

**판단과 좌표를 한 모델에게 동시에 요구하지 않는다.** 의미 판단은 VLM 이,
기하 확정은 OCR 이 한다. 그래서 ``VlmFinding.bbox_norm`` 은 처음부터 "대략"
이라고 이름과 문서에 못박아 두고, 정밀 좌표는 ``PiiRegion.bbox`` 에만 있다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from enum import Enum
from typing import Any

# ``field`` 를 별칭으로 가져오는 이유: 아래 ``VlmFinding``/``PiiRegion`` 에
# ``field`` 라는 **속성**이 있다 (VLM 이 돌려주는 필드명 힌트. JSON 키도 같다).
# 클래스 본문에서 ``field: str = ""`` 를 만나는 순간 모듈 전역의
# ``dataclasses.field`` 가 가려져서, 그 아래 줄의 ``field(default_factory=list)``
# 가 ``None(...)`` 호출이 되어 임포트 시점에 죽는다.

# --------------------------------------------------------------------------
# 라벨 스키마 (닫힌 집합)
# --------------------------------------------------------------------------

#: 탐지 대상 개인정보 라벨 (18종).
#:
#: 여기를 고치면 프롬프트의 형식표와 guided-decoding 스키마가 자동으로 따라간다.
#: 정형 식별자를 추가할 때 체크섬이 있으면 ``rules/checksums.py`` 의
#: ``VALIDATORS`` 에도 넣어라 — 탐지가 아니라 **검증**에 쓰인다.
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

#: 텍스트가 없을 수 있는 라벨. VLM 이 ``text`` 를 빈 문자열로 줘도 버리지 않는다.
#: 서명·인영은 읽을 글자가 없고 좌표만 의미가 있다.
TEXTLESS_LABELS: frozenset[str] = frozenset({"SIGNATURE"})


class Source(str, Enum):
    """**좌표를 어떻게 얻었는가.** 탐지 판단은 전부 VLM 이므로 출처가 아니라
    좌표 획득 경로를 구분한다 — 다운스트림이 신뢰할지 결정하는 기준이 이것이다.
    """

    OCR_REFINED = "ocr_refined"  # 크롭 OCR 이 좌표를 확정. 픽셀 단위로 정확하다.
    VLM_COARSE = "vlm_coarse"    # 크롭 OCR 이 못 읽어 VLM 좌표를 그대로 씀. 부정확.


class OcrStatus(str, Enum):
    OK = "ok"              # rec 성공, 신뢰도 충분
    LOW_CONF = "low_conf"  # rec 성공했으나 신뢰도 낮음 (오독 가능)
    FAILED = "failed"      # det 는 됐으나 rec 실패 (손글씨/도장 등)


class Agreement(str, Enum):
    """VLM 이 읽은 값과 크롭 OCR 이 읽은 값이 얼마나 일치하는가.

    **두 엔진의 독립 교차검증이다.** 같은 값을 서로 다른 모델이 읽어냈다면
    그 값은 거의 확실하다. 어긋나면 어느 쪽이 틀렸는지는 알 수 없지만
    "사람이 봐야 한다" 는 것은 확실하다.
    """

    EXACT = "exact"  # 정규화 후 완전일치. 양쪽이 같은 값을 읽었다.
    NONE = "none"    # 값이 일치하지 않았다 (좌표는 확정됐을 수 있다).


# --------------------------------------------------------------------------
# ① VLM 단계 산출물
# --------------------------------------------------------------------------

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2) 축정렬, 페이지 픽셀 좌표


@dataclass
class VlmFinding:
    """VLM 이 이미지에서 직접 읽어낸 개인정보 1건.

    Attributes:
        text: 문서에 **적힌 그대로**의 값. 회피 표기라면 회피 표기가 담긴다
            (감사에서 "문서에 실제로 뭐라고 적혀 있었나" 를 되짚어야 한다).
            ``TEXTLESS_LABELS`` 라벨은 빈 문자열일 수 있다.
        type: ``PII_LABELS`` 중 하나.
        field: 필드명/맥락 힌트 ("가족사항 자녀 성명" 등). 사람 검수용이고,
            같은 값이 여러 곳에 나올 때 구분에도 쓴다.
        bbox_norm: **대략의** 위치 (0.0~1.0 정규화). 이 값으로 크롭만 뜬다.
            최종 좌표로 쓰지 마라 — 그게 ``Source.VLM_COARSE`` 다.
        conf: VLM 자기 확신도.
        tile: 어느 타일에서 나왔는지 (0-기반). 단일 호출이면 0.
    """

    text: str
    type: str
    field: str = ""
    bbox_norm: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    conf: float = 0.0
    tile: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox_norm"] = [round(v, 4) for v in self.bbox_norm]
        return d


# --------------------------------------------------------------------------
# ② 크롭 OCR 단계 산출물
# --------------------------------------------------------------------------


@dataclass
class OcrBox:
    """크롭 OCR 이 검출한 텍스트 줄 1개. 좌표는 **페이지 기준으로 환산된 뒤**다.

    rec 실패 박스도 버리지 않는다 — "여기 글자는 있는데 못 읽었다" 는 정보가
    손글씨·도장 영역을 판별하는 근거이기 때문이다.
    """

    index: int
    bbox: BBox
    text: str
    status: OcrStatus = OcrStatus.OK
    rec_conf: float = 0.0
    #: 어느 크롭에서 나왔는지 (``VlmFinding`` 순번). 디버깅용.
    crop_id: int = -1

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["bbox"] = list(self.bbox)
        return d


@dataclass
class PiiRegion:
    """개인정보 영역 하나. 다운스트림 마스킹 모듈이 소비하는 단위.

    Attributes:
        bbox: **최종 좌표.** ``source`` 가 ``OCR_REFINED`` 면 픽셀 단위로 정확하고,
            ``VLM_COARSE`` 면 근사치다 (``coarse=True``, ``needs_review=True``).
        text: 크롭 OCR 이 읽은 값. 못 읽었으면 ``None``.
        vlm_text: VLM 이 읽은 값. ``text`` 와 비교한 결과가 ``agreement`` 다.
        char_span: ``text`` 내 문자 오프셋. 부분 마스킹(``901231-1******``)을
            구현할 다운스트림이 쓴다. ``bbox`` 는 항상 박스 전체 영역이다.
        verified: 체크섬 검증을 **통과**했다. 체크섬이 없는 라벨은 항상 ``False``
            이므로 "검증 실패" 와 구분하려면 ``checksum`` 을 함께 보라.
        checksum: ``"ok"`` / ``"failed"`` / ``None``(해당 라벨에 체크섬 없음).
        agreement: VLM 값과 OCR 값의 일치 정도.
        vlm_bbox: **VLM 이 원래 지목한 좌표** (픽셀). ``bbox`` 와의 차이가
            grounding 오차 그 자체다. 이 값을 남겨 두지 않으면 "좌표가 밀렸다" 를
            눈으로만 보고할 수 있고 숫자로 못 낸다 — ``scripts/diagnose.py`` 가
            이걸로 밀림의 크기와 방향을 집계한다.
    """

    id: str
    type: str
    bbox: BBox
    source: Source
    confidence: float
    text: str | None = None
    vlm_text: str | None = None
    vlm_bbox: BBox | None = None
    field: str | None = None
    member_boxes: list[BBox] = _dc_field(default_factory=list)
    member_index: list[int] = _dc_field(default_factory=list)
    char_span: tuple[int, int] | None = None
    ocr_status: OcrStatus = OcrStatus.OK
    coarse: bool = False         # 좌표가 근사치인가
    needs_review: bool = False   # 사람 검토 필요
    low_confidence: bool = False
    verified: bool = False
    checksum: str | None = None
    agreement: Agreement = Agreement.NONE
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        d["ocr_status"] = self.ocr_status.value
        d["agreement"] = self.agreement.value
        d["bbox"] = list(self.bbox)
        d["member_boxes"] = [list(b) for b in self.member_boxes]
        d["vlm_bbox"] = list(self.vlm_bbox) if self.vlm_bbox else None
        return d


# --------------------------------------------------------------------------
# 페이지 결과
# --------------------------------------------------------------------------


@dataclass
class PageResult:
    """페이지 1장의 최종 산출물."""

    image_path: str
    width: int
    height: int
    #: 다중 페이지 문서(PDF)의 1-기반 페이지 번호. 단일 이미지면 ``None``.
    page_no: int | None = None
    regions: list[PiiRegion] = _dc_field(default_factory=list)
    #: VLM 이 뱉은 원본 판단 목록. **recall 측정의 분모다** — 좌표를 못 잡아
    #: 버려진 건이 있어도 여기에는 남는다.
    findings: list[VlmFinding] = _dc_field(default_factory=list)
    #: 크롭 OCR 이 검출한 박스 전체 (페이지 좌표). 디버깅/오버레이용.
    ocr_boxes: list[OcrBox] = _dc_field(default_factory=list)
    timings: dict[str, float] = _dc_field(default_factory=dict)
    raw_llm: dict[str, Any] = _dc_field(default_factory=dict)
    warnings: list[str] = _dc_field(default_factory=list)

    #: 전처리된 페이지 이미지 (BGR numpy). 박스 오버레이를 그릴 때만 쓴다.
    #: **직렬화되지 않으며 일시적이다.** 결과 좌표는 이 이미지 기준이다.
    image: Any = _dc_field(default=None, repr=False, compare=False)

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
            out["findings"] = [f.to_dict() for f in self.findings]
            out["ocr_boxes"] = [b.to_dict() for b in self.ocr_boxes]
            out["raw_llm"] = self.raw_llm
        return out

    def stats(self) -> dict[str, Any]:
        """진단 지표.

        지표를 세 개로 좁힌 것은 의도한 것이다. 단계가 두 개뿐이므로 오차를
        어느 단계에 귀속시킬 수 있어야 한다.

            n_findings      ① VLM 이 몇 건을 찾았나        (recall 의 분자)
            localized_rate  ② 그 중 몇 %가 좌표를 확정했나 (기하 성능)
            n_disagreement  ⑤ 두 엔진이 다르게 읽은 건수   (신뢰도)
        """
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in self.regions:
            by_source[r.source.value] = by_source.get(r.source.value, 0) + 1
            by_type[r.type] = by_type.get(r.type, 0) + 1

        n_refined = by_source.get(Source.OCR_REFINED.value, 0)
        n_regions = len(self.regions)
        return {
            "n_findings": len(self.findings),
            "n_regions": n_regions,
            "n_ocr_boxes": len(self.ocr_boxes),
            "localized_rate": round(n_refined / n_regions, 3) if n_regions else 0.0,
            "n_disagreement": sum(
                1 for r in self.regions if r.agreement is not Agreement.EXACT
            ),
            "n_checksum_failed": sum(1 for r in self.regions if r.checksum == "failed"),
            "n_needs_review": sum(1 for r in self.regions if r.needs_review),
            "by_source": by_source,
            "by_type": by_type,
        }

    def to_json(self, include_ocr: bool = False, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(include_ocr=include_ocr), ensure_ascii=False, indent=indent
        )


# --------------------------------------------------------------------------
# VLM guided-decoding 스키마
# --------------------------------------------------------------------------

#: VLM 단계 출력 스키마.
#:
#: **속성 순서가 의미를 갖는다.** guided decoding 은 스키마 순서대로 토큰을
#: 생성하므로, 모델은 ``text`` -> ``type`` -> ``field`` -> ``bbox_norm`` 순으로
#: 답한다. 즉 "무엇을 읽었는지" 를 먼저 확정하고 그 다음에 "어디인지" 를 답한다.
#: 좌표를 먼저 내게 하면 값이 좌표에 끌려간다 (읽기보다 좌표 찍기가 어려우므로
#: 좌표를 먼저 고정하면 그 근처에서 값을 짜맞춘다).
#:
#: ``maxItems`` 는 타일당 상한이다. 페이지를 타일로 쪼개 호출하므로 한 번에
#: 32건을 넘길 일이 없고, 넘긴다면 타일을 더 쪼개야 한다는 신호다.
VLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "maxItems": 32,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 120},
                    "type": {"type": "string", "enum": list(PII_LABELS)},
                    "field": {"type": "string", "maxLength": 40},
                    # 키 이름과 스케일 모두 **Qwen-VL 의 native grounding 형식**이다.
                    # 공식 쿡북(cookbooks/2d_grounding.ipynb)의 출력이
                    # ``{"bbox_2d": [x1,y1,x2,y2], "label": ...}`` 이고 좌표는
                    # 0~1000 정수다. 우리 형식(0.0~1.0 소수 + bbox_norm)을 쓰면
                    # grounding 과제에서 학습 분포와 싸우게 된다 — 9B 급에서
                    # 그 대가는 좌표 정확도로 나온다.
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "conf": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["text", "type", "field", "bbox_2d", "conf"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
