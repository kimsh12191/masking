"""① VLM 탐지 단계 — 이미지를 보고 개인정보를 전사한다.

이 단계의 출력은 ``VlmFinding`` 목록이다. **좌표는 대략이고, 값이 정확하다.**
정밀 좌표는 ``locate.py`` 가 크롭 OCR 로 확정한다.

타일링을 하는 이유:

한 번의 호출로 페이지 전체를 전사하게 하면 뒤쪽 항목이 잘린다. 모델이 분류를
못해서가 아니라 **긴 구조화 출력의 후반부가 무너지기 때문**이다. 가족관계증명서
한 장에 구성원 9명 × 4필드 = 36건인데, 9B 모델이 36건을 한 번에 온전히 뱉기를
기대하는 것은 무리다.

그래서 페이지를 긴 축 방향으로 몇 조각으로 나눠 조각당 한 번씩 호출한다.
호출당 10건 내외면 안정적이다. 부수 효과로 조각을 리사이즈했을 때의 실효
해상도가 올라간다 — 같은 ``image_max_side`` 예산 안에서 글자가 더 커진다.

경계에 걸친 값을 잃지 않도록 조각을 **겹쳐서** 자르고, 겹침 때문에 생기는
중복은 ``_dedup`` 이 정리한다. 겹침 없이 자르면 경계에 걸친 한 줄이 양쪽에서
반씩 잘려 양쪽 모두 못 읽는다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .llm.client import LlmClient, fit_max_side
from .llm.prompts import SYSTEM_VLM, build_user
from .normalize import canonical_text
from .schema import VLM_SCHEMA, VlmFinding

log = logging.getLogger(__name__)

#: 정규화 비교 시 제거할 구분자. OCR/VLM 마다 하이픈·공백 처리가 다르다.
_STRIP = " \t\n-–—_./\\,·:;()[]{}'\"|"

#: bbox 가 퇴화(면적 0)했을 때 부여할 최소 정규화 크기.
_MIN_SIDE = 0.004

#: 동시 호출 상한. vLLM 서버 한 대를 상대로 이보다 늘려도 처리량이 늘지 않고
#: 큐만 길어진다 (배치는 서버가 알아서 묶는다).
_MAX_WORKERS = 8


@dataclass
class DetectConfig:
    """VLM 탐지 설정.

    Attributes:
        tiles: 페이지를 몇 조각으로 나눠 호출할지. ``1`` 이면 전체를 1회 호출한다.
            밀집한 표 문서(등본·가족관계증명서)는 3~4 를 권한다. 조각 수만큼
            호출이 늘지만 시스템 프롬프트는 고정이라 prefix caching 이 걸린다.
        overlap: 조각 간 겹침 비율 (조각 두께 기준). 경계에 걸친 한 줄을
            양쪽에서 반씩 잘라 둘 다 못 읽는 것을 막는다.
        dedup: 겹침 구간의 중복을 제거할지. 끄면 같은 값이 두 번 보고된다.
        hint: user 메시지에 덧붙일 문서 종류 힌트. 비워 두는 것이 기본이다.
        samples: 타일당 호출 횟수. **학습 없이 미탐을 줄이는 유일한 손잡이다.**
            2 이상이면 같은 타일을 여러 번 뽑아 ``_dedup`` 이 합집합을 만든다.
            한 번은 놓치고 다른 번엔 잡히는 항목이 회수된다 — 재현율이 오르고
            정밀도가 떨어진다. 마스킹에서는 미탐이 과탐보다 훨씬 비싸므로
            대체로 맞는 거래지만, 호출 수가 ``tiles × samples`` 로 늘어난다.
        sample_temperature: ``samples >= 2`` 일 때 쓸 온도. **0 이면 안 된다** —
            같은 입력에 같은 답이 와서 샘플을 늘린 의미가 없어진다.
            ``samples == 1`` 이면 무시되고 설정값(0)이 쓰인다 (결정론 유지).
        workers: 동시 호출 수. 0 이면 호출 수만큼 (``_MAX_WORKERS`` 상한).
            타일 호출은 서로 독립이므로 직렬로 돌릴 이유가 없다.
        image_factor: 비전 인코더의 패치 크기 × merge 크기. **모델을 바꾸면 반드시
            같이 바꿔야 한다** — Qwen3-VL 은 16×2=32, Qwen2/2.5-VL 은 14×2=28.
            이미지는 이 값의 배수로 리사이즈되어 인코더에 들어간다.
        max_pixels: 타일 하나의 픽셀 상한. 0 이면 모델 기본값
            (``16384 × image_factor²``). **서버(vLLM) 설정과 맞춰야 의미가 있다** —
            서버가 더 작은 값을 쓰면 타일이 더 축소되고, 작은 한글이 뭉개진다.
            이 값을 넘는 타일은 축소되므로 경고를 낸다. 미탐의 흔한 원인이다.
        coord_convention: 좌표 규약을 고정한다. ``"auto"`` 면 응답마다 추론한다
            (``infer_scale``). **자동 추론에는 원리적 사각지대가 있다** — 타일이
            1000px 보다 크고 모델이 절대 픽셀로 답했는데 그 값들이 우연히 모두
            1000 미만이면 per-mille 로 오판한다. 그 오차는 위치에 비례해 커지는
            밀림으로 나타난다. ``scripts/diagnose.py`` 로 규약을 확인했으면
            ``"per-mille"`` / ``"pixel"`` / ``"unit"`` 로 고정하는 것이 안전하다.
    """

    tiles: int = 3
    overlap: float = 0.08
    dedup: bool = True
    hint: str = ""
    samples: int = 1
    sample_temperature: float = 0.3
    workers: int = 0
    image_factor: int = 32
    max_pixels: int = 0
    coord_convention: str = "auto"


# --------------------------------------------------------------------------
# 타일 기하
# --------------------------------------------------------------------------


def tile_rects(
    width: int, height: int, tiles: int, overlap: float, factor: int = 1
) -> list[tuple[float, float, float, float]]:
    """페이지를 **긴 축 방향으로** 나눈 정규화 사각형 목록을 만든다.

    긴 축을 자르는 것은 조각을 정사각형에 가깝게 만들기 위한 것이다. VLM 의
    비전 인코더는 극단적으로 긴 이미지에서 가로세로 한쪽을 크게 줄인다.

    **경계는 ``factor`` 의 배수로 스냅한다.** 이것이 없으면 조각 크기가 패치
    격자와 안 맞아 ``smart_resize`` 가 조각을 다시 리샘플한다 (1748x826 ->
    1760x832). 그러면 두 가지를 잃는다.

    1. **보간이 들어간다.** 10px 급 한글이 뭉개지고, 그건 곧 미탐이다.
    2. **토큰 격자가 페이지 픽셀과 어긋난다.** 텍스트 한 줄이 토큰 두 행에
       걸치고, 조각 경계에 걸친 토큰이 생긴다.

    페이지 자체도 ``factor`` 의 배수여야 이 스냅이 끝까지 성립한다
    (``preprocess`` 가 오른쪽·아래에 여백을 붙여 맞춘다).

    Args:
        width: 페이지 폭 (px).
        height: 페이지 높이 (px).
        tiles: 조각 수. 1 이하면 전체 1장.
        overlap: 겹침 비율 (0.0~0.5). 조각 두께의 이 비율만큼 양쪽으로 넓힌다.
        factor: 패치 격자 (``DetectConfig.image_factor``). 1 이면 스냅하지 않는다.

    **모든 조각의 두께가 같다.** 예전에는 가운데 조각만 양쪽으로 겹쳐서 더
    두꺼웠다 (896 / 992 / 896). 그러면 모델이 조각마다 다른 기하를 보게 되고,
    좌표를 학습시킬 때 그 변동을 함께 배워야 한다. 두께를 고정하면 모델이
    상대할 기하가 하나뿐이다.

    Returns:
        ``(x1, y1, x2, y2)`` 정규화 사각형 목록. 항상 최소 1개.
        조각들은 **빈틈 없이** 페이지를 덮는다 (첫 조각은 0 에서, 마지막은 끝에서).
    """
    if tiles <= 1:
        return [(0.0, 0.0, 1.0, 1.0)]

    overlap = min(max(overlap, 0.0), 0.5)
    vertical = height >= width  # 세로가 길면 y 를 자른다
    span = height if vertical else width
    f = max(1, factor)

    def up(value: float) -> int:
        """격자 배수로 올림."""
        return min(span, -(-int(-(-value // 1)) // f) * f)

    band = span / tiles
    thickness = max(up(band), up(band * (1 + 2 * overlap)))

    # 시작 위치는 균등 분할하고 **내림**으로 스냅한다. 올림하면 조각이 뒤로
    # 밀려 앞 조각과의 사이에 빈틈이 생길 수 있는데, 빈틈은 곧 미탐이다.
    def starts_for(thick: int) -> list[int]:
        last = span - thick
        out = [int(last * i / (tiles - 1) // f) * f for i in range(tiles)]
        out[0] = 0
        out[-1] = last  # span 도 thick 도 f 의 배수이므로 last 도 배수다
        return out

    # 스냅 때문에 이웃 사이가 벌어지면 두께를 한 칸 늘려 다시 잡는다.
    # thickness 가 span 에 닿으면 조각 하나가 전체를 덮으므로 반드시 끝난다.
    starts = starts_for(thickness)
    while thickness < span and any(
        b > a + thickness for a, b in zip(starts, starts[1:], strict=False)
    ):
        thickness = min(span, thickness + f)
        starts = starts_for(thickness)

    return [
        (0.0, lo / height, 1.0, (lo + thickness) / height)
        if vertical
        else (lo / width, 0.0, (lo + thickness) / width, 1.0)
        for lo in starts
    ]


def crop_norm(image: Any, rect: tuple[float, float, float, float]) -> Any:
    """정규화 사각형으로 이미지를 잘라낸다 (numpy BGR 배열 가정).

    네 변 모두 ``round`` 로 되돌린다. 예전에는 좌상단만 ``int``(내림)였는데,
    그러면 ``tile_rects`` 가 스냅해 준 픽셀 경계가 1px 어긋나 **격자 정렬이
    깨지고** ``_to_page`` 가 쓰는 사각형과 실제 크롭이 달라진다.
    """
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(round(rect[0] * w))))
    y1 = max(0, min(h - 1, int(round(rect[1] * h))))
    x2 = max(x1 + 1, min(w, int(round(rect[2] * w))))
    y2 = max(y1 + 1, min(h, int(round(rect[3] * h))))
    return image[y1:y2, x1:x2]


def _to_page(
    local: tuple[float, float, float, float], rect: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """타일 내 정규화 좌표를 **페이지** 정규화 좌표로 환산한다.

    모델은 자기가 받은 이미지 기준으로만 답한다. 타일 번호를 프롬프트에 넣어
    모델이 환산하게 하면 틀리고, 시스템 프롬프트도 가변이 되어 캐싱이 깨진다.
    환산은 코드가 한다.
    """
    rx, ry = rect[0], rect[1]
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    x1, y1, x2, y2 = local
    return (rx + x1 * rw, ry + y1 * rh, rx + x2 * rw, ry + y2 * rh)


def _numbers(raw: Any) -> list[float] | None:
    """bbox 후보를 숫자 4개로 만든다. 못 만들면 ``None``."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def round_by_factor(value: float, factor: int) -> int:
    return int(round(value / factor) * factor)


def smart_resize(
    height: int,
    width: int,
    factor: int,
    max_pixels: int = 0,
    min_pixels: int = 0,
) -> tuple[int, int]:
    """비전 인코더가 실제로 보는 크기를 계산한다 (Qwen 의 ``smart_resize`` 이식).

    출처: ``QwenLM/Qwen3-VL`` 의 ``qwen-vl-utils/src/qwen_vl_utils/vision_process.py``.
    ``IMAGE_MIN_TOKEN_NUM = 4``, ``IMAGE_MAX_TOKEN_NUM = 16384`` 이고 기본
    한계는 ``토큰수 × factor²`` 다.

    **이 값을 알아야 하는 이유는 하나다.** Qwen2.5-VL 의 grounding 좌표는
    "리사이즈된 이미지의 절대 픽셀" 이다. 우리가 보낸 타일 크기로 나누면
    리사이즈 비율만큼 어긋나고, 그 오차는 위치에 비례해서 커진다 — 화면에는
    "박스가 전체적으로 밀렸다" 로 보인다. 원본 크기가 아니라 **모델이 본 크기**로
    나눠야 한다.

    Args:
        height: 보낼 이미지 높이 (px).
        width: 보낼 이미지 폭 (px).
        factor: 패치 크기 × merge 크기. Qwen3-VL 은 16×2=32, Qwen2/2.5-VL 은 14×2=28.
        max_pixels: 상한. 0 이면 모델 기본값 (``16384 × factor²``).
        min_pixels: 하한. 0 이면 모델 기본값 (``4 × factor²``).

    Returns:
        ``(높이, 폭)``. 둘 다 ``factor`` 의 배수다.
    """
    factor = max(1, factor)
    hi = max_pixels or 16384 * factor * factor
    lo = min_pixels or 4 * factor * factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > hi:
        beta = ((height * width) / hi) ** 0.5
        h_bar = max(factor, int(height / beta / factor) * factor)
        w_bar = max(factor, int(width / beta / factor) * factor)
    elif h_bar * w_bar < lo:
        beta = (lo / max(1, height * width)) ** 0.5
        h_bar = max(factor, -(-int(height * beta) // factor) * factor)
        w_bar = max(factor, -(-int(width * beta) // factor) * factor)
    return h_bar, w_bar


def infer_scale(
    raws: list[Any], tile_w: int, tile_h: int
) -> tuple[float, float, str]:
    """한 응답의 bbox 들을 **함께 보고** 좌표 규약을 정한다.

    기대값은 **0~1000 (per-mille)** 이다. 그게 Qwen3-VL 의 native 좌표계이고
    프롬프트도 그것을 요구한다. 이 함수가 하는 일은 규약을 추측하는 것이 아니라
    **모델이 기대와 다른 규약으로 답했을 때 조용히 망가지지 않게** 하는 것이다.

    Qwen 계열의 규약은 버전마다 다르다 — Qwen2-VL 은 0~1000, Qwen2.5-VL 은
    리사이즈된 이미지의 절대 픽셀, Qwen3-VL 은 다시 0~1000 이다. 그래서 모델을
    바꾸면 이 판정이 달라질 수 있고, ``meta["coord_convention"]`` 에 남는다.

    JSON Schema 에 ``maximum: 1000`` 을 걸어 두었지만 **믿을 수 없다** — 문법 기반
    guided decoding 은 숫자 범위를 강제하지 못하는 것이 보통이다.

    판정은 항목별이 아니라 **응답 단위**로 한다. 한 항목만 보면 작은 값이
    0~1 인지 0~1000 인지 알 수 없지만, 응답 전체의 최댓값을 보면 갈린다.

    Args:
        raws: 이 응답의 ``bbox_2d`` 원본들.
        tile_w: 이 호출에서 **모델이 본** 타일 폭 (px). ``smart_resize`` 적용 후.
        tile_h: 모델이 본 타일 높이 (px).

    Returns:
        ``(x 나눌 값, y 나눌 값, 규약 이름)``. 규약 이름은 경고와 메타에 남는다.
    """
    vals = [abs(v) for raw in raws if (nums := _numbers(raw)) for v in nums]
    hi = max(vals, default=0.0)
    if hi <= 1.0:
        # 소수로 답했다. 값이 하나뿐이고 작으면 per-mille 과 구분되지 않지만,
        # 어느 쪽으로 봐도 0~1 구간이므로 결과는 같다.
        return 1.0, 1.0, "unit"
    if hi <= 1000.0:
        return 1000.0, 1000.0, "per-mille"
    # 1000 을 넘었다 = Qwen2.5-VL 식 절대 픽셀. 나눌 값은 우리가 보낸 크기가
    # 아니라 **모델이 본 크기**다 (smart_resize 참조).
    return float(max(1, tile_w)), float(max(1, tile_h)), "pixel"


def pick_scale(
    setting: str, raws: list[Any], tile_w: int, tile_h: int
) -> tuple[float, float, str]:
    """설정이 규약을 고정했으면 그것을 쓰고, ``"auto"`` 면 추론한다.

    고정할 수 있어야 하는 이유가 있다. ``infer_scale`` 은 응답의 최댓값으로
    판정하는데, **타일이 1000px 보다 크고 모델이 절대 픽셀로 답했는데 그 값들이
    우연히 모두 1000 미만이면** per-mille 로 오판한다 (페이지 왼쪽 위에만 항목이
    있는 경우). 오판의 결과는 위치에 비례해 커지는 밀림이라 눈으로는 그냥
    "박스가 밀렸다" 로만 보인다. 한 번 측정해 확정했으면 추론에 맡기지 않는 것이
    맞다.

    Args:
        setting: ``"auto"`` / ``"unit"`` / ``"per-mille"`` / ``"pixel"``.
        raws: 이 응답의 ``bbox_2d`` 원본들.
        tile_w: 모델이 본 타일 폭 (px).
        tile_h: 모델이 본 타일 높이 (px).

    Returns:
        ``(x 나눌 값, y 나눌 값, 규약 이름)``.

    Raises:
        ValueError: 알 수 없는 규약 이름. 오타를 조용히 ``auto`` 로 넘기면
            고정한 줄 알고 쓰게 된다.
    """
    if setting == "auto":
        return infer_scale(raws, tile_w, tile_h)
    if setting == "unit":
        return 1.0, 1.0, "unit"
    if setting == "per-mille":
        return 1000.0, 1000.0, "per-mille"
    if setting == "pixel":
        return float(max(1, tile_w)), float(max(1, tile_h)), "pixel"
    raise ValueError(
        f"detect.coord_convention 값이 잘못되었습니다: {setting!r} "
        "(auto / unit / per-mille / pixel 중 하나)"
    )


def _sane_bbox(
    raw: Any, scale_x: float = 1.0, scale_y: float = 1.0
) -> tuple[float, float, float, float] | None:
    """모델이 준 bbox 를 정규화 좌표로 정리한다. 못 쓰면 ``None``.

    좌표 순서가 뒤집힌 것(x2<x1)은 정렬해 살린다 — 값 자체는 맞는데 순서만
    틀린 경우가 흔하고, 버리면 탐지 하나를 잃는다. 면적이 0 이면 최소 크기를
    준다 (한 점을 찍은 경우. 패딩을 붙이면 쓸 만한 크롭이 된다).

    **범위를 벗어난 값은 규약 환산 뒤에도 남으면 잘라낸다.** 다만 그 전에
    ``infer_scale`` 이 규약을 맞춰 놓았으므로, 여기까지 와서 잘리는 것은
    소수의 이상치뿐이다 (예전에는 전량이 여기서 뭉개졌다).
    """
    nums = _numbers(raw)
    if nums is None:
        return None
    vals = [
        min(max(v / (scale_x if i % 2 == 0 else scale_y), 0.0), 1.0)
        for i, v in enumerate(nums)
    ]

    x1, x2 = sorted((vals[0], vals[2]))
    y1, y2 = sorted((vals[1], vals[3]))
    if x2 - x1 < _MIN_SIDE:
        x1, x2 = max(0.0, x1 - _MIN_SIDE), min(1.0, x1 + _MIN_SIDE)
    if y2 - y1 < _MIN_SIDE:
        y1, y2 = max(0.0, y1 - _MIN_SIDE), min(1.0, y1 + _MIN_SIDE)
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------
# 중복 제거
# --------------------------------------------------------------------------


def match_key(text: str) -> str:
    """비교용 키. 회피 표기를 접고 구분자·공백을 지운다.

    ``"공1공-1234-5678"`` 과 ``"010 1234 5678"`` 이 같은 키가 되어야
    겹침 구간에서 두 번 잡힌 같은 값을 하나로 볼 수 있다.
    """
    folded = canonical_text(text)
    return "".join(ch for ch in folded if ch not in _STRIP).upper()


def _overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _dedup(findings: list[VlmFinding]) -> list[VlmFinding]:
    """겹침 구간에서 두 번 보고된 같은 값을 하나로 합친다.

    **값이 같아도 위치가 겹치지 않으면 남긴다.** 같은 이름이 문서 여러 곳에
    나오는 것은 정상이고, 그 위치를 각각 마스킹해야 한다. 텍스트만 보고
    합치면 두 번째 위치가 마스킹되지 않는다.
    """
    kept: list[VlmFinding] = []
    for item in sorted(findings, key=lambda f: (-f.conf, f.tile)):
        key = match_key(item.text)
        duplicate = any(
            k.type == item.type
            and match_key(k.text) == key
            and _overlaps(k.bbox_norm, item.bbox_norm)
            for k in kept
        )
        if not duplicate:
            kept.append(item)
    # 읽기 순서(위 -> 아래, 왼 -> 오른)로 돌려준다
    kept.sort(key=lambda f: (round(f.bbox_norm[1], 3), f.bbox_norm[0]))
    return kept


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def detect(
    image: Any,
    client: LlmClient,
    config: DetectConfig | None = None,
    warnings: list[str] | None = None,
) -> tuple[list[VlmFinding], list[dict[str, Any]]]:
    """이미지에서 개인정보를 전사한다.

    Args:
        image: 전처리된 BGR numpy 배열.
        client: vLLM 클라이언트.
        config: 탐지 설정.
        warnings: 경고를 append 할 목록.

    Returns:
        ``(탐지 목록, 호출별 메타정보)``. 한 타일이 실패해도 나머지는 유지한다 —
        페이지 하나를 통째로 버리는 것이 최악이다. 실패한 타일은 메타에
        ``error`` 로 남고 ``warnings`` 에도 기록된다.
    """
    cfg = config or DetectConfig()
    warn = warnings if warnings is not None else []
    h, w = image.shape[:2]

    rects = tile_rects(w, h, cfg.tiles, cfg.overlap, cfg.image_factor)
    user = build_user(cfg.hint)
    samples = max(1, cfg.samples)
    # samples==1 이면 온도를 건드리지 않는다 — 기존의 결정론적 동작을 지킨다.
    temp = None if samples == 1 else cfg.sample_temperature

    # (타일, 샘플) 조합. 순서를 고정해 두면 병렬로 돌려도 결과가 결정론적이다.
    calls = [(t, s) for t in range(len(rects)) for s in range(samples)]
    tile_imgs = [
        crop_norm(image, rect) if len(rects) > 1 else image for rect in rects
    ]
    # 비전 인코더가 실제로 보게 될 크기. 좌표 환산의 기준이고, 축소가 일어나면
    # 작은 한글이 뭉개져 **미탐**으로 이어진다. 그래서 조용히 넘기지 않는다.
    #
    # 순서가 중요하다: 클라이언트가 image_max_side 로 먼저 줄이고, 그 다음
    # 서버의 비전 프로세서가 smart_resize 를 적용한다. 원본 타일 크기로
    # 계산하면 절대 픽셀 좌표가 그 비율만큼 어긋난다.
    seen = [
        smart_resize(
            *fit_max_side(t.shape[0], t.shape[1], client.config.image_max_side),
            cfg.image_factor,
            cfg.max_pixels,
        )
        for t in tile_imgs
    ]
    for i, (tile, (sh, sw)) in enumerate(zip(tile_imgs, seen, strict=True)):
        th, tw = tile.shape[:2]
        if sh * sw < th * tw * 0.81:  # 한 변 10% 이상 줄어든다
            warn.append(
                f"VLM 타일 {i}: {tw}x{th} -> {sw}x{sh} 로 축소되어 인코더에 들어갑니다 "
                f"(max_pixels 한계). 작은 글씨가 뭉개져 미탐이 늘 수 있습니다 — "
                f"detect.tiles 를 늘리거나 서버의 max_pixels 를 올릴 것"
            )

    def one(job: tuple[int, int]) -> dict[str, Any]:
        tile_no, sample_no = job
        payload, meta = client.complete_json(
            system=SYSTEM_VLM,
            user=user,
            schema=VLM_SCHEMA,
            image=tile_imgs[tile_no],
            temperature=temp,
        )
        meta["tile"] = tile_no
        meta["sample"] = sample_no
        meta["rect"] = [round(v, 4) for v in rects[tile_no]]
        meta["payload"] = payload
        return meta

    if len(calls) == 1:
        metas = [one(calls[0])]
    else:
        # 클라이언트를 미리 만들어 둔다 (지연 초기화가 스레드에서 겹치지 않게).
        _ = client.client
        n_workers = cfg.workers or min(len(calls), _MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            metas = list(pool.map(one, calls))

    findings: list[VlmFinding] = []
    for meta in metas:
        tile_no = meta["tile"]
        rect = rects[tile_no]
        where = f"타일 {tile_no}" + (
            f" 샘플 {meta['sample']}" if samples > 1 else ""
        )
        payload = meta.pop("payload", {}) or {}

        if meta.get("error"):
            warn.append(f"VLM {where} 실패: {meta['error']}")
            continue
        if meta.get("salvaged"):
            warn.append(
                f"VLM {where}: JSON 형식 이탈을 복구했습니다 "
                f"({meta['salvaged']}) — 프롬프트/guided decoding 점검 필요"
            )

        raw_items = [i for i in (payload.get("findings") or []) if isinstance(i, dict)]

        # 좌표 규약을 먼저 정한다. 항목별로 판단하면 알 수 없다 (infer_scale).
        # 나눌 기준은 우리가 보낸 크기가 아니라 **모델이 본 크기**다.
        seen_h, seen_w = seen[tile_no]
        scale_x, scale_y, convention = pick_scale(
            cfg.coord_convention,
            [i.get("bbox_2d") for i in raw_items],
            seen_w,
            seen_h,
        )
        meta["coord_convention"] = convention
        if convention == "pixel":
            warn.append(
                f"VLM {where}: bbox 가 0~1000 이 아니라 절대 픽셀로 왔습니다 "
                f"(Qwen2.5-VL 식. 모델이 본 크기 {seen_w}x{seen_h} 로 환산). "
                f"프롬프트가 요구한 형식이 아닙니다 — 모델을 바꾸면 다시 확인할 것"
            )

        n_before = len(findings)
        for item in raw_items:
            label = item.get("type")
            if not label:
                continue
            bbox = _sane_bbox(item.get("bbox_2d"), scale_x, scale_y)
            if bbox is None:
                warn.append(
                    f"VLM {where}: bbox_2d 가 없거나 잘못된 항목 폐기 "
                    f"(type={label}, text={str(item.get('text'))[:20]!r})"
                )
                continue
            findings.append(
                VlmFinding(
                    text=str(item.get("text") or ""),
                    type=str(label),
                    field=str(item.get("field") or ""),
                    bbox_norm=_to_page(bbox, rect),
                    conf=float(item.get("conf") or 0.0),
                    tile=tile_no,
                )
            )
        log.debug("%s: %d건", where, len(findings) - n_before)

    if not cfg.dedup:
        return findings, metas

    deduped = _dedup(findings)
    if len(deduped) < len(findings):
        log.debug(
            "중복 %d건 병합 (타일 겹침%s)",
            len(findings) - len(deduped),
            " + 다중 샘플" if samples > 1 else "",
        )
    return deduped, metas
