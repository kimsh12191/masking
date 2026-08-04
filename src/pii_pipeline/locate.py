"""② 좌표 확정 단계 — VLM 이 지목한 영역을 크롭해 OCR 로 측량한다.

    VlmFinding (판단은 신뢰, 좌표는 대략, 전사는 참고)
      └─ 크롭 (여유 패딩 + 업샘플)
      └─ 배치 OCR
      └─ 크롭 안에서 대상 박스 고르기          ← 이 단계가 핵심이다
      └─ PiiRegion (좌표 정확)

**이 모듈은 모델을 호출하지 않는다.** 박스 선택은 전부 코드가 결정론적으로
한다. 크롭 OCR 결과를 다시 VLM 에게 보여주고 고르라고 하면 이전 구조의 실패
(시각적 위치를 박스 번호로 표현시키기)로 되돌아가고, 비용도 ``O(타일)`` 에서
``O(항목)`` 으로 등급이 바뀐다.

박스 선택 우선순위 — **좁은 답부터, 확실한 근거부터**:

=====  ============================  ==================  ============================
순위   근거                          좌표                agreement
=====  ============================  ==================  ============================
①      VLM 텍스트와 일치하는 박스    정확 (OCR)          ``EXACT``
②      VLM bbox 와 겹치는 박스       정확 (OCR)          ``NONE`` (교차검증 실패)
③      크롭 안에서 가장 가까운 줄    OCR + VLM 합집합    ``NONE`` (좌표가 밀렸다)
④      크롭에 OCR 박스가 없음        근사 (VLM)          ``NONE``
=====  ============================  ==================  ============================

③이 있어야 하는 이유는 ②의 전제가 깨질 때다. 크롭은 **VLM 좌표가 어긋난다는
전제로** 넉넉히 뜨는데, 한때 선택은 **어긋나지 않았다는 전제로** 패딩 없는 VLM
bbox 와의 겹침만 봤다. 밀림이 bbox 크기를 넘으면 ①②가 모두 빈손이 되어 페이지의
모든 항목이 ④로 떨어진다 — 좌표를 교정하려고 만든 단계가 좌표가 부정확할 때
정확히 멈춘다. 크롭을 뜰 때 인정한 공차는 고를 때도 인정해야 한다.

**유사 매칭 단계는 없다.** 한때 ①과 ②사이에 "글자가 비슷하면 같은 값으로 본다"
는 단계가 있었지만 제거했다. ②가 같은 일을 더 안전하게 한다 — 텍스트가 비슷하고
위치도 겹치면 ②가 같은 박스를 고르고, 위치가 안 겹치면 애초에 다른 값이므로
유사도로 이어붙여선 안 된다. 임계값 하나를 없애는 대신 잃는 것이 없다.

②가 있어야 하는 이유가 이 설계의 핵심 교훈이다.

VLM 이 잘하는 일은 **"여기에 이 종류의 개인정보가 있다"** 는 판단이다. 근거가
중복적이기 때문이다 — 필드 라벨, 표 구조상의 위치, 옆 칸과의 관계, 값의 대략적
형태. 반면 ``901112-2846261`` 을 13자리 모두 정확히 맞히는 것은 중복성이 0인
과제이고, 9B 급 모델에서 신뢰할 수 없다. 숫자에는 오독을 교정해줄 언어모델
prior 가 없다 (``홍길둥`` 은 ``홍길동`` 으로 교정되지만 숫자는 안 된다).

①만 있으면 **잘하는 일의 결과를 못하는 일이 가로막는다.** VLM 이 위치를 정확히
지목했는데 숫자 한 자리를 틀리면 그 판단이 좌표로 이어지지 못한다. 그래서
텍스트 일치는 **필수 조건이 아니라 확인 증거**로 강등하고, 좌표 확정은 기하가
맡는다.

그럼 왜 VLM 에게 ``text`` 를 계속 요구하는가 — 다운스트림이 필요해서가 아니다.
**모델이 픽셀을 실제로 보게 만드는 장치다.** "여기 개인정보 있냐" 만 물으면
모델은 레이아웃 prior 로 답할 수 있다 ("이 서식의 이 위치는 보통 주민번호
칸이니까"). 칸이 비어 있어도 그렇게 답한다. 값을 적게 하면 그게 안 된다 —
뭔가를 적으려면 봐야 한다.

②의 대가는 정직하게 적어 둔다. **VLM bbox 가 어긋나면 엉뚱한 셀이 조용히
선택된다.** ①에는 없던 실패 모드다 (그건 못 찾으면 못 찾았다고 했다).
완충 장치는 둘이다 — 가장 좁은 후보를 우선하고, 텍스트가 일치하지 않은 채
기하로 선택된 건은 ``agreement=NONE`` + ``needs_review`` 로 남기고 ``reason`` 에
VLM 이 읽은 값과 선택된 박스의 OCR 값을 함께 적는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .normalize import canonicalize
from .ocr.layout import denorm_bbox, pad_bbox, union_bbox
from .ocr.paddle_runner import PaddleOcrRunner
from .schema import (
    Agreement,
    BBox,
    OcrBox,
    OcrStatus,
    PiiRegion,
    Source,
    VlmFinding,
)

log = logging.getLogger(__name__)

#: 비교용 정규화에서 제거할 구분자.
_STRIP = frozenset(" \t\n\r-–—_./\\,·:;()[]{}'\"|")

#: 최종 박스 여유 패딩 (px). 경계 글자 잘림 방지.
DEFAULT_PAD = 2

#: VLM 좌표를 그대로 쓸 때의 여유 패딩 (px). 좌표가 부정확하므로 넉넉히.
COARSE_PAD = 12

#: 이 값 미만이면 low_confidence 플래그. 폐기하지는 않는다.
LOW_CONF_THRESHOLD = 0.5


@dataclass
class LocateConfig:
    """좌표 확정 설정.

    Attributes:
        pad_ratio: 크롭 여유 (VLM bbox 크기 대비 비율). VLM 좌표는 어긋나므로
            넉넉히 잡는다. 좁으면 값이 크롭 밖으로 나가 못 찾는다.
        min_pad_px: 크롭 여유의 최솟값 (px). 작은 bbox 에서 비율만으로는 부족하다.

            **이 값은 모델의 grounding 해상도 하한보다 커야 한다.** Qwen-VL 의
            공간 단위는 비전 토큰 1개, 즉 ``image_factor`` × ``image_factor``
            픽셀이다 (Qwen3-VL 은 32×32). 0~1000 좌표의 눈금이 1.76px 여도
            모델이 실제로 구분할 수 있는 최소 단위는 32px 이고, 300dpi 한글
            한 줄 높이가 30~40px 이므로 **모델은 원리적으로 한 줄보다 정확할
            수 없다.**

            한때 이 값이 12 였다. 한 줄짜리 이름 칸(약 100×30px)에서 ``pad_ratio``
            0.35 는 세로 10px 밖에 주지 못해 ``max(12, 10) = 12px`` 이 되고,
            32px 오차를 흡수하지 못한다. 값이 크롭 밖으로 나가면 텍스트 매칭이
            실패하고, 그 다음은 어긋난 자리에서 박스를 고르게 된다. 모델의
            해상도 하한의 **2배(토큰 2칸)** 를 준다 — 격자 단위로 맞추는 것이 일관된다.
        upscale: 크롭을 OCR 에 넣기 전 확대 배율. **작은 글씨에 대한 유일한
            대응책이다.** 원본에 없는 정보를 만들지는 못하지만, rec 모델은
            입력 글자 높이에 민감하므로 실측으로 효과가 있다. 1.0 이면 끈다.
        max_crop_side: 업샘플 후 크롭 긴 변 상한 (px). 넘으면 배율을 줄인다.
        retry_pad_ratio: 1차에서 값을 못 찾았을 때 크롭을 다시 뜰 여유 비율.
            VLM bbox 가 어긋난 경우를 한 번 더 시도한다.
        max_join: 한 값을 이룰 수 있는 최대 박스 수. 텍스트 이어붙이기와 기하
            선택에 함께 적용된다. 기하 선택에서는 **과선택 상한** 역할이다 —
            넉넉한 bbox 가 옆 칸까지 덮었을 때 무한정 딸려오지 않게 막는다.
        geometry_fallback: 텍스트 매칭이 실패했을 때 기하로 박스를 고를지.
            **끄면 이전 동작으로 돌아간다** (텍스트가 안 맞으면 좌표 포기).
            A/B 비교용으로 남겨 둔 스위치다.
        min_cover: 박스 중심이 VLM bbox 밖일 때, 면적의 이 비율 이상이 안에
            들어와야 후보로 본다. 중심 판정이 실패하는 경우(긴 주소 줄이 VLM
            bbox 를 관통)를 받는 2차 그물이다.
    """

    pad_ratio: float = 0.35
    min_pad_px: int = 64
    upscale: float = 2.0
    max_crop_side: int = 1600
    retry_pad_ratio: float = 1.2
    max_join: int = 4
    geometry_fallback: bool = True
    min_cover: float = 0.5


# --------------------------------------------------------------------------
# 비교용 정규화 (원문 오프셋 매핑 포함)
# --------------------------------------------------------------------------


@dataclass
class _Norm:
    """비교용 정규 문자열 + 원문 오프셋 매핑.

    ``char_span`` 은 반드시 **원문 기준**이어야 한다 — 다운스트림이 부분
    마스킹을 구현할 때 원본 텍스트에 적용하기 때문이다.
    """

    text: str
    starts: list[int]
    ends: list[int]

    def span(self, start: int, end: int) -> tuple[int, int]:
        """정규 문자열 구간 ``[start, end)`` 를 원문 오프셋으로 되돌린다."""
        return (self.starts[start], self.ends[end - 1])


def _norm(text: str) -> _Norm:
    """회피 표기를 접고 구분자·공백을 제거한 비교용 형태를 만든다.

    ``"공1공-1234-5678"`` 이 ``"01012345678"`` 이 되어 ``"010 1234 5678"`` 과
    같은 값으로 매칭된다.
    """
    canon = canonicalize(text)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, ch in enumerate(canon.text):
        if ch in _STRIP:
            continue
        chars.append(ch.upper())
        starts.append(canon.starts[i])
        ends.append(canon.ends[i])
    return _Norm("".join(chars), starts, ends)


# --------------------------------------------------------------------------
# 크롭
# --------------------------------------------------------------------------


def crop_rect(
    finding: VlmFinding, page_w: int, page_h: int, pad_ratio: float, min_pad: int
) -> BBox:
    """VLM 의 대략 좌표에서 크롭할 페이지 사각형을 만든다.

    패딩은 **비율과 절대값 중 큰 쪽**을 쓴다. 비율만 쓰면 작은 bbox 에서 패딩이
    거의 0 이 되고, 절대값만 쓰면 큰 주소 블록에서 부족하다.
    """
    x1, y1, x2, y2 = denorm_bbox(finding.bbox_norm, page_w, page_h)
    pad_x = max(min_pad, int((x2 - x1) * pad_ratio))
    pad_y = max(min_pad, int((y2 - y1) * pad_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(page_w, x2 + pad_x),
        min(page_h, y2 + pad_y),
    )


def crop_scale(rect: BBox, cfg: LocateConfig) -> float:
    """``rect`` 크롭에 실제로 적용될 업샘플 배율.

    ``_cut`` 과 테스트가 같은 계산을 쓰도록 따로 뺐다. 배율을 잘못 가정하면
    좌표 환산이 어긋나는데, 그 오차는 "기하 선택이 안 된다" 처럼 보여서
    원인을 찾기 어렵다.
    """
    long_side = max(rect[3] - rect[1], rect[2] - rect[0])
    if long_side <= 0:
        return 1.0
    scale = max(1.0, cfg.upscale)
    if long_side * scale > cfg.max_crop_side:
        scale = max(1.0, cfg.max_crop_side / long_side)
    return scale


def _cut(image: Any, rect: BBox, cfg: LocateConfig) -> tuple[Any, float]:
    """이미지를 잘라 확대한다.

    Returns:
        ``(크롭 배열, 실제 적용된 배율)``. 배율을 함께 돌려주는 것은 좌표를
        되돌리는 데 필요하기 때문이다. 상한 때문에 요청 배율이 깎일 수 있다.
    """
    x1, y1, x2, y2 = rect
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return crop, 1.0

    scale = crop_scale(rect, cfg)
    if scale <= 1.0:
        return crop, 1.0

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - 환경 의존
        return crop, 1.0

    resized = cv2.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, scale


def _to_page(boxes: list[OcrBox], rect: BBox, scale: float, crop_id: int) -> list[OcrBox]:
    """크롭 좌표의 박스들을 페이지 좌표로 옮긴다 (새 객체를 만든다)."""
    ox, oy = rect[0], rect[1]
    out: list[OcrBox] = []
    for box in boxes:
        bx1, by1, bx2, by2 = box.bbox
        out.append(
            OcrBox(
                index=-1,  # 나중에 페이지 전역 번호를 부여한다
                bbox=(
                    ox + int(bx1 / scale),
                    oy + int(by1 / scale),
                    ox + int(round(bx2 / scale)),
                    oy + int(round(by2 / scale)),
                ),
                text=box.text,
                status=box.status,
                rec_conf=box.rec_conf,
                crop_id=crop_id,
            )
        )
    return out


# --------------------------------------------------------------------------
# 크롭 안에서 값 찾기
# --------------------------------------------------------------------------


@dataclass
class _Match:
    """크롭 안에서 대상 박스를 골라낸 결과.

    Attributes:
        how: 무엇을 근거로 골랐나. ``"text"`` (VLM 텍스트와 일치) 또는
            ``"geometry"`` (VLM bbox 와 겹침). ``reason`` 문구와 검수 우선순위가
            여기서 갈린다 — 기하로 고른 것은 엉뚱한 셀일 수 있다.
    """

    boxes: list[OcrBox]
    agreement: Agreement
    char_span: tuple[int, int] | None
    how: str = "text"


def find_value(
    seed_text: str, boxes: list[OcrBox], cfg: LocateConfig
) -> _Match | None:
    """크롭 OCR 결과에서 VLM 이 읽은 값과 같은 박스(들)를 찾는다.

    **완전일치만 본다.** 비슷하면 넘어가는 단계는 두지 않는다 — 텍스트가 안
    맞으면 ``select_by_geometry`` 가 위치로 고른다. 임계값을 하나 없애는 대신
    잃는 것이 없다 (모듈 docstring 참고).

    탐색 순서에 이유가 있다.

    1. **박스 1개 안에서 완전일치** — 가장 좁고 정확한 답. 부분 마스킹을 위한
       ``char_span`` 도 이 경우에만 의미가 있다.
    2. **인접 박스 이어붙여 완전일치** — 값이 쪼개진 경우 (``"010-1234"`` 와
       ``"5678"`` 이 별개 박스). 읽기 순서로 붙인다.

    좁은 답을 먼저 찾는 것이 중요하다. 이어붙이기를 먼저 시도하면 값 하나에
    옆 칸까지 딸려 들어와 박스가 셀 두 개를 덮는다.

    Args:
        seed_text: VLM 이 읽은 값.
        boxes: 크롭 OCR 박스 (읽기 순서).
        cfg: 설정.

    Returns:
        찾았으면 ``_Match``, 못 찾았으면 ``None``.
    """
    seed = _norm(seed_text).text
    if not seed:
        return None

    usable = [b for b in boxes if b.status is not OcrStatus.FAILED and b.text.strip()]
    if not usable:
        return None

    norms = [_norm(b.text) for b in usable]

    # ① 박스 1개 완전일치
    for box, norm in zip(usable, norms, strict=True):
        pos = norm.text.find(seed)
        if pos >= 0:
            return _Match(
                boxes=[box],
                agreement=Agreement.EXACT,
                char_span=norm.span(pos, pos + len(seed)),
            )

    # ② 인접 박스 이어붙이기
    for width in range(2, min(cfg.max_join, len(usable)) + 1):
        for i in range(len(usable) - width + 1):
            joined = "".join(n.text for n in norms[i : i + width])
            if seed in joined:
                return _Match(
                    boxes=list(usable[i : i + width]),
                    agreement=Agreement.EXACT,
                    char_span=None,  # 여러 박스에 걸쳐 원문 오프셋이 하나가 아니다
                )

    return None


def select_by_geometry(
    vlm_bbox: BBox,
    boxes: list[OcrBox],
    cfg: LocateConfig,
    search: BBox | None = None,
) -> _Match | None:
    """VLM 이 지목한 사각형과 겹치는 OCR 박스를 고른다.

    텍스트를 전혀 보지 않는다. **VLM 의 판단(위치와 종류)만 쓰고 전사 정확도는
    쓰지 않는다.** 숫자 한 자리 오독이 좌표 확정을 막지 않게 하는 것이 목적이다.

    ``FAILED`` 박스도 후보에 넣는다. det 는 성공했고 rec 만 실패한 박스는
    "여기 글자는 있는데 못 읽었다" 는 뜻이고, VLM 이 그 자리를 지목했다면
    **그 좌표가 VLM 의 근사 좌표보다 정확하다.** 손글씨 칸이 여기서
    ``vlm_coarse`` 를 벗어난다 — 텍스트 매칭으로는 불가능했던 회수다.

    선택 순서 (좁은 답 우선):

    1. **중심이 VLM bbox 안**에 있는 박스. 보통 정확히 그 셀이다.
    2. 없으면 **면적의 ``min_cover`` 이상이 안에 들어온** 박스. 긴 주소 줄이
       VLM bbox 를 관통해 중심이 밖으로 나간 경우를 받는다.
    3. 없으면 **겹침이 가장 큰 박스 하나**.
    4. 겹치는 것이 하나도 없으면 **``search`` 안에서 VLM 중심에 가장 가까운
       박스와 그 같은 줄**. ``how="nearby"`` 로 표시된다.

    4단계가 있어야 하는 이유가 이 함수의 요점이다. **크롭은 VLM 좌표가 어긋난다는
    전제로 넉넉히 뜨는데, 선택은 어긋나지 않았다는 전제로 하고 있었다.** 밀림이
    bbox 크기를 넘으면 1~3 이 전부 빈손이 되고, 페이지의 모든 항목이
    ``vlm_coarse`` 로 떨어진다 — VLM 좌표를 교정하려고 만든 단계가 VLM 좌표가
    부정확할 때 정확히 작동을 멈춘다. 크롭을 뜰 때 인정한 공차는 고를 때도
    인정해야 한다.

    Args:
        vlm_bbox: VLM 좌표를 픽셀로 환산한 사각형 (패딩 전).
        boxes: 크롭 OCR 박스 (페이지 좌표).
        cfg: 설정. ``max_join`` 이 과선택 상한, ``min_cover`` 가 2단계 임계값이다.
        search: 실제로 잘라낸 크롭 사각형. 4단계의 탐색 범위다. ``None`` 이면
            ``vlm_bbox`` 를 쓰므로 4단계가 사실상 꺼진다.

    Returns:
        고른 박스들을 담은 ``_Match``, 후보가 전혀 없으면 ``None``.
    """
    if not boxes:
        return None

    how = "geometry"
    inside = [b for b in boxes if _center_in(b.bbox, vlm_bbox)]
    if not inside:
        inside = [b for b in boxes if _cover(b.bbox, vlm_bbox) >= cfg.min_cover]
    if not inside:
        best = max(boxes, key=lambda b: _overlap_area(b.bbox, vlm_bbox))
        if _overlap_area(best.bbox, vlm_bbox) > 0:
            inside = [best]
    if not inside:
        inside = _nearby(vlm_bbox, boxes, cfg, search or vlm_bbox)
        if not inside:
            return None
        how = "nearby"

    # 과선택 상한. 넉넉한 bbox 가 옆 칸까지 덮었을 때 무한정 딸려오지 않게
    # 막는다. 자를 때는 VLM 이 지목한 중심에 가까운 것을 남긴다.
    if len(inside) > cfg.max_join:
        cx = (vlm_bbox[0] + vlm_bbox[2]) / 2.0
        cy = (vlm_bbox[1] + vlm_bbox[3]) / 2.0
        inside.sort(key=lambda b: _dist2(b.bbox, cx, cy))
        inside = inside[: cfg.max_join]

    # 읽기 순서로 되돌린다 (위 -> 아래, 왼 -> 오른). 텍스트 재조립 순서가 된다.
    inside.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return _Match(
        boxes=inside,
        agreement=Agreement.NONE,  # 값 교차검증은 실패했다. 좌표만 확정된 것이다.
        char_span=None,
        how=how,
    )


def _nearby(
    vlm_bbox: BBox, boxes: list[OcrBox], cfg: LocateConfig, search: BBox
) -> list[OcrBox]:
    """VLM bbox 와 전혀 겹치지 않을 때, 크롭 안에서 가장 가까운 줄을 고른다.

    가장 가까운 박스 **하나만** 잡으면 쪼개진 값의 뒷부분이 남는다
    (``"010-1234"`` 만 덮고 ``"5678"`` 은 안 덮인다). 가려지지 않은 숫자가
    남는 것이 옆 칸을 덧칠하는 것보다 훨씬 위험하므로, 같은 줄에 있고
    크롭 안에 있는 박스를 함께 잡는다. 범위는 크롭 사각형으로 닫혀 있다.
    """
    near = [b for b in boxes if _overlap_area(b.bbox, search) > 0]
    if not near:
        return []

    cx = (vlm_bbox[0] + vlm_bbox[2]) / 2.0
    cy = (vlm_bbox[1] + vlm_bbox[3]) / 2.0
    anchor = min(near, key=lambda b: _dist2(b.bbox, cx, cy))
    band = max(1.0, (anchor.bbox[3] - anchor.bbox[1]) * 0.6)
    ay = (anchor.bbox[1] + anchor.bbox[3]) / 2.0
    return [b for b in near if abs((b.bbox[1] + b.bbox[3]) / 2.0 - ay) <= band]


def _center_in(box: BBox, outer: BBox) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _overlap_area(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(w * h) if w > 0 and h > 0 else 0.0


def _cover(box: BBox, outer: BBox) -> float:
    """``box`` 면적 중 ``outer`` 안에 들어온 비율."""
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    return _overlap_area(box, outer) / area if area > 0 else 0.0


def _dist2(box: BBox, cx: float, cy: float) -> float:
    bx = (box[0] + box[2]) / 2.0
    by = (box[1] + box[3]) / 2.0
    return (bx - cx) ** 2 + (by - cy) ** 2


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def locate(
    findings: list[VlmFinding],
    image: Any,
    ocr: PaddleOcrRunner,
    config: LocateConfig | None = None,
    warnings: list[str] | None = None,
) -> tuple[list[PiiRegion], list[OcrBox]]:
    """VLM 탐지 목록의 좌표를 크롭 OCR 로 확정한다.

    세 국면으로 돈다. **모델 호출은 없다** — OCR 배치 2회와 코드 판정뿐이다.

    ① 크롭 + OCR + 텍스트 매칭        전부
    ② 크롭을 넓혀 텍스트 매칭 재시도   ①에서 실패한 것만
    ③ 기하 선택                       ②까지 실패한 것 중 크롭에 박스가 있는 것

    ②를 한 번만 하는 이유: 두 번 넓혀서 텍스트가 안 잡히면 크롭 크기 문제가
    아니라 판독 불일치 문제다. 그건 ③이 처리한다.

    ③의 후보는 **①의(더 좁은) 크롭 결과**를 우선 쓴다. ②의 크롭은 넓어서
    옆 칸까지 들어오므로 기하 선택에 불리하다.

    Args:
        findings: VLM 탐지 목록.
        image: 전처리된 BGR numpy 배열. 좌표계의 기준이다.
        ocr: OCR 실행기.
        config: 설정.
        warnings: 경고를 append 할 목록.

    Returns:
        ``(영역 목록, 크롭 OCR 박스 전체)``. 영역은 ``findings`` 와 **1:1** 이다
        (좌표를 못 잡아도 ``VLM_COARSE`` 로 남는다 — 개인정보를 조용히 버리지
        않는다). 박스 목록은 페이지 좌표로 환산되어 있고 오버레이/디버깅용이다.
    """
    cfg = config or LocateConfig()
    warn = warnings if warnings is not None else []
    if not findings:
        return [], []

    page_h, page_w = image.shape[:2]
    n = len(findings)

    # ── ① 전부 크롭 -> OCR -> 텍스트 매칭 ─────────────────────
    tight, tight_rects = _crop_and_ocr(
        findings, image, ocr, cfg, cfg.pad_ratio, range(n), page_w, page_h
    )
    matches: list[_Match | None] = [
        find_value(f.text, tight[i], cfg) for i, f in enumerate(findings)
    ]
    #: 기하 선택과 진단에 쓸 박스 풀. 좁은 크롭 결과를 기본으로 한다.
    pool: list[list[OcrBox]] = [list(b) for b in tight]
    #: 각 풀이 실제로 나온 크롭 사각형. 기하 선택의 탐색 범위가 된다.
    rects: list[BBox] = list(tight_rects)

    # ── ② 텍스트 매칭 실패분만 크롭을 넓혀 재시도 ─────────────
    # 값이 빈 항목은 재시도하지 않는다 — 찾을 문자열이 없으므로 크롭을 넓혀도
    # 매칭될 수 없고, OCR 을 한 번 더 도는 값만 늘어난다.
    retry = [i for i in range(n) if matches[i] is None and findings[i].text.strip()]
    if retry:
        log.debug("텍스트 매칭 재시도 %d건 (크롭 확대)", len(retry))
        wide, wide_rects = _crop_and_ocr(
            findings, image, ocr, cfg, cfg.retry_pad_ratio, retry, page_w, page_h
        )
        for i, boxes, rect in zip(retry, wide, wide_rects, strict=True):
            matches[i] = find_value(findings[i].text, boxes, cfg)
            if matches[i] is None:
                # 텍스트로 못 찾았다 -> 다음은 기하 선택이고, 거기서는 **넓은 쪽이
                # 옳다.** "좁은 크롭이 옆 칸을 덜 물어온다" 는 이점은 VLM 좌표가
                # 정확할 때만 성립하는데, 여기까지 왔다는 것 자체가 그 전제가
                # 깨졌다는 신호다. 좁은 크롭을 지키면 어긋난 자리만 계속 본다.
                pool[i] = boxes
                rects[i] = rect
            elif not pool[i]:
                pool[i] = boxes
                rects[i] = rect

    # ── ③ 기하 선택 (텍스트를 보지 않는다) ────────────────────
    n_geometry = 0
    n_nearby = 0
    if cfg.geometry_fallback:
        for i in range(n):
            if matches[i] is not None or not pool[i]:
                continue
            vlm_bbox = denorm_bbox(findings[i].bbox_norm, page_w, page_h)
            picked = select_by_geometry(vlm_bbox, pool[i], cfg, rects[i])
            if picked is not None:
                matches[i] = picked
                n_geometry += 1
                if picked.how == "nearby":
                    n_nearby += 1

    # ── 박스에 페이지 전역 번호 부여 ──────────────────────────
    all_boxes: list[OcrBox] = []
    for boxes in pool:
        for box in boxes:
            box.index = len(all_boxes)
            all_boxes.append(box)

    # ── 영역 생성 ─────────────────────────────────────────────
    regions: list[PiiRegion] = []
    for finding, match, boxes in zip(findings, matches, pool, strict=True):
        if match is not None:
            regions.append(_refined_region(finding, match, page_w, page_h))
        else:
            regions.append(_coarse_region(finding, boxes, page_w, page_h))

    n_coarse = sum(1 for r in regions if r.coarse)
    if n_geometry:
        # 기하로 고른 것은 엉뚱한 셀일 수 있다. 건수를 드러내야 검수자가
        # 이 부류를 우선 확인한다.
        warn.append(
            f"기하 선택 {n_geometry}/{n}건 — VLM 텍스트와 일치하는 OCR 박스가 없어 "
            f"위치 겹침으로 골랐다. 좌표는 OCR 것이지만 다른 셀일 수 있다."
        )
    if n_nearby:
        # 이 숫자가 크면 VLM grounding 이 전반적으로 밀렸다는 뜻이다. 겹치는
        # 박스가 아예 없어서 "가까운 줄" 로 주워온 건들이다.
        warn.append(
            f"근접 선택 {n_nearby}/{n}건 — VLM bbox 와 겹치는 OCR 박스가 하나도 없어 "
            f"크롭 안에서 가장 가까운 줄을 골랐다. VLM 좌표가 밀린 것으로 보인다 "
            f"(scripts/diagnose.py 로 밀림의 크기·방향을 확인할 것)."
        )
    if n_coarse:
        warn.append(
            f"좌표 확정 실패 {n_coarse}/{n}건 — 크롭에 OCR 박스가 없어 VLM 좌표를 "
            f"그대로 사용 (vlm_coarse). 해당 영역은 사람 검토가 필요하다."
        )
    return regions, all_boxes


def _crop_and_ocr(
    findings: list[VlmFinding],
    image: Any,
    ocr: PaddleOcrRunner,
    cfg: LocateConfig,
    pad_ratio: float,
    targets: Any,
    page_w: int,
    page_h: int,
) -> tuple[list[list[OcrBox]], list[BBox]]:
    """``targets`` 인덱스의 탐지들을 크롭해 배치 OCR 한다.

    Returns:
        ``(박스 목록, 크롭 사각형 목록)``. 둘 다 ``targets`` 와 **같은 순서·같은
        길이**다. 박스 좌표는 페이지 기준으로 환산되어 있다 (크롭 오프셋 +
        업샘플 배율 역산). 크롭 사각형을 함께 돌려주는 것은 기하 선택이
        "우리가 값이 이 안에 있다고 선언한 범위" 를 알아야 하기 때문이다.
    """
    idx_list = list(targets)
    if not idx_list:
        return [], []

    rects: list[BBox] = []
    crops: list[Any] = []
    scales: list[float] = []
    for i in idx_list:
        rect = crop_rect(findings[i], page_w, page_h, pad_ratio, cfg.min_pad_px)
        crop, scale = _cut(image, rect, cfg)
        rects.append(rect)
        crops.append(crop)
        scales.append(scale)

    results = ocr.run_many(crops)
    return [
        _to_page(boxes, rect, scale, crop_id=i)
        for i, rect, scale, boxes in zip(idx_list, rects, scales, results, strict=True)
    ], rects


def _refined_region(
    finding: VlmFinding, match: _Match, page_w: int, page_h: int
) -> PiiRegion:
    """크롭 OCR 이 좌표를 확정한 경우 (텍스트 매칭이든 기하 선택이든).

    ``source`` 는 둘 다 ``OCR_REFINED`` 다 — 좌표가 OCR 것이라는 사실은 같다.
    근거의 차이는 ``agreement`` 와 ``reason`` 이 표현한다.
    """
    vlm_bbox = denorm_bbox(finding.bbox_norm, page_w, page_h)
    member = [b.bbox for b in match.boxes]
    ocr_text = " ".join(b.text for b in match.boxes if b.text)
    worst = _worst_status(match.boxes)

    if match.agreement is Agreement.EXACT:
        # 두 엔진이 같은 값을 읽었다면 VLM 자기확신도보다 높게 볼 근거가 있다.
        # 다만 1.0 으로 올리지 않는다 — 체크섬 통과만이 그 자격이 있다 (verify.py).
        conf = min(0.95, max(finding.conf, 0.9))
        reason = "크롭 OCR 과 완전일치"
    elif match.how == "nearby":
        # 겹치는 박스가 하나도 없었다. 어느 쪽이 맞는지 알 수 없으므로 **둘 다
        # 덮는다** — VLM 이 지목한 자리와 크롭 안에서 가장 가까운 줄. 어느 한쪽만
        # 택하면 틀렸을 때 값이 안 가려진 채 남는다. 범위는 크롭으로 닫혀 있고,
        # 덧칠은 미탐보다 싸다.
        member = [*member, vlm_bbox]
        conf = finding.conf
        reason = (
            f"VLM bbox 와 겹치는 OCR 박스가 없어 크롭 안에서 가장 가까운 줄을 선택 "
            f"(VLM 좌표가 밀린 것으로 보인다. VLM 좌표까지 함께 덮었다). "
            f"VLM '{finding.text[:30]}' vs OCR '{ocr_text[:40] or '(판독 실패)'}'"
        )
    else:
        # 기하 선택. 좌표는 OCR 것이지만 값 교차검증은 없었다. VLM 이 읽은 값과
        # 선택된 박스가 읽은 값을 함께 남긴다 — 엉뚱한 셀을 골랐는지 판단하려면
        # 이 두 개가 나란히 있어야 한다.
        conf = finding.conf
        reason = (
            f"VLM bbox 와 위치가 겹치는 OCR 박스를 선택 (텍스트는 불일치). "
            f"VLM '{finding.text[:30]}' vs OCR '{ocr_text[:40] or '(판독 실패)'}'"
        )

    return PiiRegion(
        id="",  # finalize 에서 부여
        type=finding.type,
        bbox=pad_bbox(union_bbox(member), DEFAULT_PAD, page_w, page_h),
        source=Source.OCR_REFINED,
        confidence=conf,
        text=ocr_text or None,
        vlm_text=finding.text or None,
        vlm_bbox=vlm_bbox,
        field=finding.field or None,
        member_boxes=member,
        member_index=[b.index for b in match.boxes],
        char_span=match.char_span,
        ocr_status=worst,
        coarse=False,
        needs_review=(match.agreement is not Agreement.EXACT) or worst is not OcrStatus.OK,
        low_confidence=conf < LOW_CONF_THRESHOLD,
        agreement=match.agreement,
        reason=reason,
    )


def _coarse_region(
    finding: VlmFinding,
    boxes: list[OcrBox],
    page_w: int,
    page_h: int,
) -> PiiRegion:
    """좌표를 확정하지 못한 경우. VLM 좌표를 그대로 쓴다.

    근접 선택(③)이 들어온 뒤로 이 경로에 남는 것은 **둘뿐이다.** 크롭에 OCR 박스가
    하나라도 있으면 ③이 받아낸다.

    ==========================  ============================================
    VLM 의 text 가 비어 있다     VLM 이 위치는 짚었으나 값을 못 읽었다
    크롭에 박스가 하나도 없다     OCR **검출** 문제. det 임계값·업샘플을 본다
    ==========================  ============================================

    그래서 이 건수가 많다는 것은 이제 **OCR 쪽 신호다** — 예전에는 VLM 좌표
    문제와 섞여 있어서 구분이 안 됐다. VLM 좌표가 밀린 건은 ③으로 가고
    ``needs_review`` 와 경고 문구에 따로 집계된다.

    **빈 ``text`` 는 이제 전부 검토 대상이다.** 라벨셋이 10종으로 좁아지기 전에는
    ``SIGNATURE`` (서명·인영)가 있어서 "읽을 글자가 없는 것이 정상인 항목" 이
    존재했고, 그 건들을 검토에서 빼야 검토 큐가 서명으로 가득 차지 않았다. 서명이
    라벨셋에서 빠진 뒤로 10종은 모두 읽을 글자가 있는 값이므로, 빈 ``text`` 는
    정상이 아니라 **VLM 이 값을 못 읽었다는 신호**다.
    """
    no_text = not finding.text.strip()
    read = " ".join(b.text for b in boxes if b.text.strip())

    if no_text:
        reason = (
            "VLM 이 위치는 짚었으나 값을 읽지 못했다 (text 가 비어 있다). "
            "VLM 좌표 사용"
        )
    elif not boxes:
        reason = (
            "크롭에서 OCR 박스가 검출되지 않았다 (손글씨/도장/저대비 추정). "
            "VLM 좌표 사용"
        )
    else:
        reason = (
            f"VLM bbox 와 겹치는 OCR 박스가 없다 (VLM 좌표가 어긋난 것으로 보인다). "
            f"VLM '{finding.text[:30]}' / 크롭에서 읽은 것 '{read[:40] or '(없음)'}'"
        )

    bbox = denorm_bbox(finding.bbox_norm, page_w, page_h)
    return PiiRegion(
        id="",
        type=finding.type,
        bbox=pad_bbox(bbox, COARSE_PAD, page_w, page_h),
        source=Source.VLM_COARSE,
        confidence=finding.conf,
        text=None,
        vlm_text=finding.text or None,
        vlm_bbox=bbox,
        field=finding.field or None,
        member_boxes=[bbox],
        member_index=[],
        ocr_status=OcrStatus.FAILED,
        coarse=True,
        # 이 경로로 온 건은 전부 사람이 봐야 한다. 좌표가 근사치이거나
        # (OCR 이 확정하지 못했다) 값이 없거나 (VLM 이 읽지 못했다) 둘 중 하나다.
        needs_review=True,
        low_confidence=finding.conf < LOW_CONF_THRESHOLD,
        agreement=Agreement.NONE,
        reason=reason,
    )


def _worst_status(boxes: list[OcrBox]) -> OcrStatus:
    order = {OcrStatus.OK: 0, OcrStatus.LOW_CONF: 1, OcrStatus.FAILED: 2}
    worst = OcrStatus.OK
    for box in boxes:
        if order[box.status] > order[worst]:
            worst = box.status
    return worst
