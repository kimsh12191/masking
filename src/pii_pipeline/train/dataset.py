"""OCR 결과를 VLM 학습 타깃으로 바꾼다.

한 타일에 대한 학습 샘플 하나는 이렇게 생겼다.

    입력   타일 이미지 (추론 때 모델이 받는 것과 **같은** 이미지)
    타깃   {"findings":[{"text":"...","bbox_2d":[x1,y1,x2,y2]}, ...]}

``bbox_2d`` 는 추론과 같은 규약(타일 기준 0~1000 정수)이다.

좌표를 만드는 방법 — 디코딩 함수를 뒤집는다
--------------------------------------------

추론에서 모델의 답이 페이지 좌표가 되는 경로는 이것뿐이다::

    bbox_2d --(_sane_bbox: /1000)--> 타일 정규화 --(_to_page: rect)--> 페이지 정규화

그래서 라벨은 이 경로를 **정확히 역으로** 계산한다. 크롭 사각형의 반올림된
픽셀값으로 따로 계산하면 안 된다 — ``crop_norm`` 은 반올림 픽셀로 자르지만
``_to_page`` 는 반올림 전 정규화 사각형을 쓰기 때문에, 둘을 섞으면 서브픽셀
오차가 생기고 그 오차는 학습 내내 한 방향으로 쌓인다.

남는 오차는 **per-mille 양자화뿐**이다. 폭 1760px 타일에서 눈금 하나가 1.76px
이므로 왕복 오차의 상한이 그 절반이다. ``roundtrip`` 이 이 상한을 계산해
검산에 쓴다.

무엇을 정답으로 삼는가
----------------------

============================  ==================  ==========================
OCR 박스                      text                왜
============================  ==================  ==========================
``OK`` (rec_conf 충분)        읽은 값             전사와 좌표를 함께 가르친다
``LOW_CONF`` / 임계값 미만    ``""``              **좌표만.** 오독을 정답으로
                                                  주면 지금 잘하는 전사가
                                                  망가진다
``FAILED`` (det 만 성공)      ``""``              손글씨·도장. "글자는 있는데
                                                  못 읽었다" 는 좌표가 정확
                                                  하다는 뜻이다
============================  ==================  ==========================

``FAILED`` 를 버리지 않는 것이 중요하다. OCR 이 못 읽는 부류가 정확히 손글씨와
도장이고, 그건 VLM 좌표가 가장 나쁜 부류이기도 하다. det 는 성공했으므로
**좌표 라벨로서는 멀쩡하다.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..detect import _sane_bbox, _to_page
from ..ocr.layout import denorm_bbox
from ..schema import BBox, OcrBox, OcrStatus

#: 타일 정규화 사각형 ``(x1, y1, x2, y2)``. ``detect.tile_rects`` 의 원소.
Rect = tuple[float, float, float, float]


def scale_regions(
    rect: Rect,
    page_w: int,
    page_h: int,
    scales: list[float],
    rng: Any,
) -> list[Rect]:
    """타일과 **같은 종횡비**로 더 작은 영역들을 뽑는다 (스케일 증강).

    입력 크기는 고정인데 글자 크기만 달라지게 하는 것이 목적이다. 작은 영역을
    잘라 타일 크기로 확대하면 글자가 그 배율만큼 커 보인다.

        s = 1.0   타일 그대로            글자 1배
        s = 2.0   타일의 절반 영역       글자 2배

    **왜 필요한가.** 캔버스를 고정해도 문서마다 폰트 크기가 다르다. 추론 타일
    하나의 스케일만 학습하면 그보다 작거나 큰 글씨에서 좌표가 흔들린다. 종횡비를
    타일과 맞추는 것이 중요한데, 안 맞추고 확대하면 글자가 찌그러지고 모델이
    왜곡된 글자를 학습하게 된다.

    ``s < 1`` 은 만들지 않는다. 타일이 이미 페이지 폭 전체를 쓰므로 더 넓은
    영역이 없다 — 그 방향은 캔버스 크기를 바꿔야 얻어진다.

    Args:
        rect: 기준 타일의 정규화 사각형.
        page_w: 페이지 폭 (px).
        page_h: 페이지 높이 (px).
        scales: 배율 목록. 1.0 이하는 건너뛴다.
        rng: ``random.Random``. 위치를 뽑는 데 쓴다.

    Returns:
        정규화 사각형 목록. 페이지 안에 완전히 들어간다.
    """
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    out: list[Rect] = []
    for s in scales:
        if s <= 1.0:
            continue
        w, h = rw / s, rh / s
        # 영역이 페이지 밖으로 나가지 않는 범위에서 위치를 뽑는다.
        x = rng.uniform(0.0, max(0.0, 1.0 - w))
        y = rng.uniform(0.0, max(0.0, 1.0 - h))
        out.append((x, y, min(1.0, x + w), min(1.0, y + h)))
    return out


@dataclass
class GroundingConfig:
    """학습 샘플 생성 설정.

    Attributes:
        text_min_conf: 이 값 이상인 박스만 ``text`` 를 정답으로 쓴다. 미만이면
            좌표만 쓰고 ``text`` 는 빈 문자열이다. **오독된 글자를 정답으로 주면
            지금 잘하는 전사 능력이 망가진다.**
        keep_textless: ``FAILED`` 박스(손글씨·도장)를 좌표 라벨로 쓸지.
            끄면 학습이 인쇄 텍스트에만 치우친다.
        min_containment: 박스가 타일 안에 이 비율 이상 들어와야 그 타일의 샘플에
            넣는다. 타일 경계에 걸친 줄을 반쪽만 가르치면 모델이 잘린 박스를
            배운다. 타일이 겹쳐 있으므로 걸린 줄은 옆 타일에서 온전히 잡힌다.
        max_items: 타일당 최대 항목 수. ``schema.VLM_SCHEMA`` 의 ``maxItems`` 와
            맞춰야 한다 — 추론에서 낼 수 없는 길이를 학습시키면 안 된다.
        min_items: 이 개수 미만이면 샘플을 버린다. 빈 타일만 잔뜩 배우면
            "아무것도 없다" 로 답하는 쪽이 쉬워진다.
        min_side_px: 이보다 작은 박스는 버린다. 노이즈 검출이다.
        aug_scales: 스케일 증강 배율 (``scale_regions``). 같은 배율을 두 번
            적으면 위치가 다른 영역이 두 개 나온다.
        query_ratio: 한 샘플에서 물을 값의 비율 구간 ``(최소, 최대)``.
            **전부 묻지 않는 것이 중요하다** (``sample_query`` 참조).
        read_ratio: 빈 항목(도장·손글씨)이 있는 영역에서 ``read`` 과제도 낼 확률.
            기본 0 — 좌표 능력은 내용과 무관해서 글자로 배운 것이 도장에도 쓰인다.
        seed: 증강 위치·질의 추출의 난수 시드. 같은 시드면 같은 데이터셋이다.
    """

    text_min_conf: float = 0.9
    keep_textless: bool = True
    min_containment: float = 0.95
    max_items: int = 32
    min_items: int = 1
    min_side_px: int = 6
    aug_scales: list[float] = field(default_factory=lambda: [1.5, 2.0])
    query_ratio: tuple[float, float] = (0.25, 1.0)
    read_ratio: float = 0.0
    seed: int = 0


@dataclass
class TileSample:
    """타일 하나에 대한 학습 샘플.

    Attributes:
        tile: 타일 인덱스. 증강 영역은 ``-1``.
        rect: 타일의 정규화 사각형. 좌표 검산에 필요하다.
        items: ``{"text": str, "bbox_2d": [x1,y1,x2,y2]}`` 목록 (읽기 순서).
        max_error_px: 이 샘플에서 관측된 최대 왕복 오차 (페이지 픽셀).
            per-mille 양자화 상한을 넘으면 변환이 어딘가 어긋난 것이다.
        n_textless: 좌표만 가르치는 항목 수 (손글씨·도장·저신뢰).
    """

    tile: int
    rect: Rect
    items: list[dict[str, Any]] = field(default_factory=list)
    max_error_px: float = 0.0
    n_textless: int = 0

    def target_json(self) -> str:
        """추론 스키마와 같은 모양의 학습 타깃 문자열."""
        return json.dumps({"findings": self.items}, ensure_ascii=False)

    def query(self) -> list[str]:
        """물을 수 있는 값 전체. **중복을 제거한다.**

        같은 값이 여러 곳에 있다는 사실은 **답**에서 표현된다 (같은 텍스트로
        항목이 여러 개). 질문에 두 번 적으면 "두 번 물었으니 두 개 답한다" 를
        배우게 되고, 그건 추론에서 쓸 수 없는 규칙이다 — 추론 때는 값이 몇 번
        나오는지 아무도 모른다.

        빈 문자열(도장·손글씨)은 지목할 방법이 없으므로 뺀다.

        실제로 물을 것은 이 중 일부다 (``sample_query`` 참조).

        Returns:
            첫 등장 순서를 지킨 중복 없는 값 목록.
        """
        seen: set[str] = set()
        out: list[str] = []
        for item in self.items:
            text = item["text"]
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def locate_items(self, query: list[str] | None = None) -> list[dict[str, Any]]:
        """위치 질의의 정답. **물어본 값들의** 모든 출현.

        묻지 않은 값은 답에서 빠진다. 이 필터가 없으면 "이미지에 있는 것은
        묻지 않아도 답한다" 를 가르치게 되는데, 추론에서는 정확히 그 반대가
        필요하다 — 페이지에 텍스트가 수십 줄이어도 개인정보만 답해야 한다.

        Args:
            query: 물어본 값들. ``None`` 이면 텍스트가 있는 항목 전체.

        Returns:
            읽기 순서의 항목 목록. 질문 순서가 아니다.
        """
        if query is None:
            return [item for item in self.items if item["text"]]
        wanted = set(query)
        return [item for item in self.items if item["text"] in wanted]


def sample_query(
    texts: list[str], rng: Any, ratio_range: tuple[float, float]
) -> list[str]:
    """물어볼 값의 부분집합을 뽑는다.

    **매번 전부 묻지 않는 것이 핵심이다.** 추론에서는 페이지에 텍스트가 수십
    줄이어도 그중 개인정보 몇 개만 답해야 한다. 학습에서 항상 "전부" 를 물으면
    모델은 *"보이는 것을 다 답한다"* 를 배우고, **질문에 없는 것을 무시하는
    연습을 한 번도 하지 못한다.**

    부수 효과로 같은 타일에서 매번 다른 (질문, 답) 쌍이 나온다.

    **순서를 섞는다.** 답은 읽기 순서여야 하는데, 질문이 항상 읽기 순서로
    들어오면 모델이 "질문 순서대로 답한다" 로 배울 수 있다. 그러면 추론에서
    질문 순서라는 것이 없을 때 무너진다.

    Args:
        texts: 물을 수 있는 값 전체 (``TileSample.query()``).
        rng: ``random.Random``.
        ratio_range: 뽑을 비율 구간 ``(최소, 최대)``. ``(1.0, 1.0)`` 이면 전부.

    Returns:
        뽑힌 값들 (섞인 순서). 입력이 비어 있으면 빈 목록.
    """
    if not texts:
        return []
    lo, hi = ratio_range
    ratio = rng.uniform(min(lo, hi), max(lo, hi))
    k = max(1, min(len(texts), round(len(texts) * ratio)))
    picked = rng.sample(texts, k)
    rng.shuffle(picked)
    return picked


# --------------------------------------------------------------------------
# 좌표 변환 — 추론 디코딩의 역함수
# --------------------------------------------------------------------------


def to_permille(bbox: BBox, rect: Rect, page_w: int, page_h: int) -> list[int]:
    """페이지 픽셀 박스를 **타일 기준 0~1000 정수**로 바꾼다.

    ``detect._to_page`` 의 역이다. 그쪽이 쓰는 것과 같은 정규화 사각형을 써야
    추론이 이 값을 원래 자리로 되돌린다.

    Args:
        bbox: 페이지 픽셀 좌표 박스.
        rect: 타일의 정규화 사각형 (``detect.tile_rects`` 원소).
        page_w: 페이지 폭 (px).
        page_h: 페이지 높이 (px).

    Returns:
        ``[x1, y1, x2, y2]`` 0~1000 정수.
    """
    rw = max(1e-9, rect[2] - rect[0])
    rh = max(1e-9, rect[3] - rect[1])

    def local(value: float, page_dim: int, origin: float, span: float) -> int:
        norm = value / max(1, page_dim)
        return int(round(min(max((norm - origin) / span, 0.0), 1.0) * 1000))

    return [
        local(bbox[0], page_w, rect[0], rw),
        local(bbox[1], page_h, rect[1], rh),
        local(bbox[2], page_w, rect[0], rw),
        local(bbox[3], page_h, rect[1], rh),
    ]


def roundtrip(permille: list[int], rect: Rect, page_w: int, page_h: int) -> BBox:
    """``bbox_2d`` 를 **추론과 똑같은 경로로** 페이지 픽셀로 되돌린다.

    검산 전용이다. 여기서 원래 박스가 안 나오면 라벨이 틀린 것이고, 그대로
    학습시키면 모델에게 틀린 자리를 가르치게 된다.
    """
    local = _sane_bbox(list(permille), 1000.0, 1000.0)
    assert local is not None  # 4개 정수는 항상 통과한다
    return denorm_bbox(_to_page(local, rect), page_w, page_h)


def quantization_limit(rect: Rect, page_w: int, page_h: int) -> float:
    """per-mille 눈금 하나가 페이지 픽셀로 얼마인가 (왕복 오차의 상한).

    타일이 넓을수록 눈금이 굵다. 폭 1760px 타일이면 1.76px 이고, 반올림이
    양쪽으로 갈리므로 실제 상한은 그 값이다. 검산 허용치를 상수로 박으면
    타일 크기가 바뀔 때 조용히 틀린다.
    """
    return max(
        (rect[2] - rect[0]) * page_w / 1000.0,
        (rect[3] - rect[1]) * page_h / 1000.0,
    )


# --------------------------------------------------------------------------
# 샘플 생성
# --------------------------------------------------------------------------


def _containment(bbox: BBox, rect_px: BBox) -> float:
    """``bbox`` 면적 중 ``rect_px`` 안에 들어온 비율."""
    w = min(bbox[2], rect_px[2]) - max(bbox[0], rect_px[0])
    h = min(bbox[3], rect_px[3]) - max(bbox[1], rect_px[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return (w * h) / area if area > 0 else 0.0


def rect_to_px(rect: Rect, page_w: int, page_h: int) -> BBox:
    """정규화 사각형을 픽셀로. 포함 판정에만 쓴다 (좌표 계산에는 쓰지 않는다)."""
    return (
        int(round(rect[0] * page_w)),
        int(round(rect[1] * page_h)),
        int(round(rect[2] * page_w)),
        int(round(rect[3] * page_h)),
    )


def _label_text(box: OcrBox, cfg: GroundingConfig) -> str | None:
    """이 박스의 ``text`` 정답. ``None`` 이면 샘플에서 제외."""
    if box.status is OcrStatus.FAILED or box.rec_conf < cfg.text_min_conf:
        return "" if cfg.keep_textless else None
    text = box.text.strip()
    return text if text else ("" if cfg.keep_textless else None)


def build_tile_sample(
    boxes: list[OcrBox],
    tile: int,
    rect: Rect,
    page_w: int,
    page_h: int,
    cfg: GroundingConfig | None = None,
) -> TileSample | None:
    """페이지 OCR 박스에서 타일 하나의 학습 샘플을 만든다.

    OCR 은 **페이지 전체에 한 번만** 돌린다. 타일마다 돌리면 경계에서 같은 줄이
    두 번 다르게 검출되고, 타일 간에 정답이 어긋난다.

    Args:
        boxes: 페이지 좌표계의 OCR 박스 전체.
        tile: 타일 인덱스 (기록용).
        rect: 타일의 정규화 사각형.
        page_w: 페이지 폭 (px).
        page_h: 페이지 높이 (px).
        cfg: 설정.

    Returns:
        샘플. 유효 항목이 ``min_items`` 미만이면 ``None``.
    """
    cfg = cfg or GroundingConfig()
    rect_px = rect_to_px(rect, page_w, page_h)

    picked: list[tuple[OcrBox, str]] = []
    for box in boxes:
        if box.width < cfg.min_side_px or box.height < cfg.min_side_px:
            continue
        if _containment(box.bbox, rect_px) < cfg.min_containment:
            continue
        text = _label_text(box, cfg)
        if text is None:
            continue
        picked.append((box, text))

    if len(picked) < cfg.min_items:
        return None

    # 읽기 순서(위 -> 아래, 왼 -> 오른). 추론에서 기대하는 순서와 같아야 한다.
    picked.sort(key=lambda p: (p[0].bbox[1], p[0].bbox[0]))
    if len(picked) > cfg.max_items:
        # 앞에서 자른다. 뒤를 버리면 "페이지 아래쪽은 답하지 않는다" 를
        # 가르치게 되므로, 넘치는 타일은 애초에 타일 수를 늘려 해결할 일이다.
        picked = picked[: cfg.max_items]

    sample = TileSample(tile=tile, rect=rect)
    for box, text in picked:
        permille = to_permille(box.bbox, rect, page_w, page_h)
        back = roundtrip(permille, rect, page_w, page_h)
        error = max(abs(a - b) for a, b in zip(back, box.bbox, strict=True))
        sample.max_error_px = max(sample.max_error_px, float(error))
        sample.items.append({"text": text, "bbox_2d": permille})
        if not text:
            sample.n_textless += 1
    return sample
