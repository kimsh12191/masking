"""전처리: 기울기 보정 및 해상도 정규화.

주의: 여기서 **이미지를 회전/스케일하면 좌표계가 바뀐다.** 최종 결과 좌표는
전처리된 이미지 기준이므로, 원본 좌표계로 되돌려야 하는 경우
``PreprocessResult.transform`` 을 사용해 역변환하라.

기본 정책은 **회전 각도가 임계값을 넘을 때만** 보정하는 것이다. 대부분의 스캔
문서는 그대로 두는 편이 안전하다 (보간으로 작은 글씨가 열화된다).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: 이 각도(도) 미만이면 보정하지 않는다.
DESKEW_MIN_ANGLE = 0.4

#: 이 각도(도)를 넘으면 오검출로 보고 보정하지 않는다.
DESKEW_MAX_ANGLE = 15.0


@dataclass
class PreprocessResult:
    image: Any
    width: int
    height: int
    #: 원본 -> 전처리 변환 정보 (역변환용)
    transform: dict[str, Any] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)


def _estimate_skew(gray: Any) -> float:
    """텍스트 라인 방향으로부터 기울기 각도(도)를 추정한다."""
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 100:
        return 0.0

    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # minAreaRect 의 각도 규약을 [-45, 45] 로 정규화
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle)


def preprocess(
    image_path: str,
    target_long_side: int | None = 2464,
    deskew: bool = True,
    align: int = 32,
    canvas: tuple[int, int] | None = None,
) -> PreprocessResult:
    """이미지 파일을 읽어 OCR 에 적합한 형태로 정규화한다.

    Args:
        image_path: 입력 이미지 경로.
        target_long_side: 긴 변 목표 길이. ``canvas`` 를 주면 무시된다.
        deskew: 기울기 보정 여부.
        align: 이 값의 배수가 되도록 오른쪽·아래에 흰 여백을 붙인다
            (VLM 패치 격자. ``DetectConfig.image_factor``). 1 이면 끈다.
            ``canvas`` 를 주면 캔버스가 이미 배수이므로 하는 일이 없다.
        canvas: ``(폭, 높이)`` 고정 캔버스. 주면 **모든 페이지가 정확히 이 크기**가
            된다 (``preprocess_array`` 참조).

    Returns:
        전처리 결과. ``image`` 는 OpenCV BGR numpy 배열.

    Raises:
        RuntimeError: opencv 미설치 또는 이미지를 읽을 수 없는 경우.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError("opencv-python 이 필요합니다.") from exc

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")

    return preprocess_array(
        img,
        target_long_side=target_long_side,
        deskew=deskew,
        align=align,
        canvas=canvas,
    )


def _fit_canvas(img: Any, canvas_w: int, canvas_h: int, cv2: Any) -> tuple[Any, float, tuple[int, int]]:
    """종횡비를 지키며 캔버스에 맞추고 오른쪽·아래를 흰색으로 채운다.

    **축소만이 아니라 확대도 한다.** 여기가 ``target_long_side`` 방식과 갈리는
    지점이다. 확대를 막으면 저해상도 스캔이 큰 캔버스 구석에 작게 박히고, 그러면
    같은 서식이라도 스캔 DPI 에 따라 글자 크기가 제각각이 된다 — 좌표를 학습시킬
    때 그 변동이 그대로 잡음이 된다. 캔버스를 채우면 글자 크기가 **문서 자체의
    레이아웃에만** 의존한다.

    확대로 없던 정보가 생기지는 않는다 (흐려질 뿐이다). 다만 좌표는 정확히
    보존되고, 이 단계의 목적이 그것이다.

    Returns:
        ``(이미지, 적용 배율, (오른쪽 여백, 아래 여백))``.
    """
    h, w = img.shape[:2]
    scale = min(canvas_w / max(1, w), canvas_h / max(1, h))
    new_w = max(1, min(canvas_w, round(w * scale)))
    new_h = max(1, min(canvas_h, round(h * scale)))
    if (new_w, new_h) != (w, h):
        img = cv2.resize(
            img,
            (new_w, new_h),
            # 축소는 INTER_AREA, 확대는 INTER_CUBIC 이 표준이다.
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
    pad_w, pad_h = canvas_w - new_w, canvas_h - new_h
    if pad_w or pad_h:
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    return img, scale, (pad_w, pad_h)


def preprocess_array(
    img: Any,
    target_long_side: int | None = 2464,
    deskew: bool = True,
    align: int = 32,
    canvas: tuple[int, int] | None = None,
) -> PreprocessResult:
    """이미 메모리에 있는 이미지를 정규화한다.

    PDF 페이지처럼 파일을 거치지 않는 입력에 쓴다.

    두 가지 방식이 있고 ``canvas`` 가 있으면 그쪽이 이긴다.

    ===================  =========================================================
    ``canvas`` 지정      **모든 페이지가 정확히 같은 크기**가 된다. 종횡비를
                         지키며 확대·축소한 뒤 오른쪽·아래를 흰색으로 채운다
    ``target_long_side`` 긴 변만 맞춘다 (축소만). 짧은 변은 종횡비대로라
                         페이지마다 다르다
    ===================  =========================================================

    **좌표 학습을 할 것이면 ``canvas`` 를 쓴다.** 페이지 크기가 고정이라야
    타일도 고정이고, 모델이 한 가지 기하만 학습하면 된다. 페이지마다 크기가
    다르면 같은 per-mille 값이 페이지마다 다른 픽셀을 가리키게 되고, 그
    변동을 모델이 함께 배워야 한다.

    Args:
        img: BGR numpy 배열.
        target_long_side: 긴 변 목표 길이. ``canvas`` 를 주면 무시된다.
        deskew: 기울기 보정 여부.
        align: 이 값의 배수가 되도록 오른쪽·아래에 흰 여백을 붙인다. 1 이면 끈다.
        canvas: ``(폭, 높이)``. **``align`` 의 배수로 둘 것.**

    Returns:
        전처리 결과. 입력 배열은 변경하지 않는다.

    Raises:
        RuntimeError: opencv 미설치.
        ValueError: ``canvas`` 값이 양수가 아닐 때.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError("opencv-python 이 필요합니다.") from exc

    orig_h, orig_w = img.shape[:2]
    applied: list[str] = []
    transform: dict[str, Any] = {
        "original_width": orig_w,
        "original_height": orig_h,
        "scale": 1.0,
        "rotation_deg": 0.0,
    }

    # 1) 해상도 정규화
    if canvas is not None:
        # YAML 에서는 리스트로 들어온다. 길이/부호를 여기서 한 번만 검사한다 —
        # 잘못된 값이 조용히 통과하면 페이지 전체가 엉뚱한 크기가 된다.
        if len(canvas) != 2:
            raise ValueError(f"canvas 는 [폭, 높이] 두 값이어야 합니다: {canvas}")
        canvas_w, canvas_h = int(canvas[0]), int(canvas[1])
        if canvas_w <= 0 or canvas_h <= 0:
            raise ValueError(f"canvas 는 양수여야 합니다: {canvas}")
        img, scale, pad = _fit_canvas(img, canvas_w, canvas_h, cv2)
        transform["scale"] = scale
        transform["canvas"] = [canvas_w, canvas_h]
        transform["canvas_pad"] = list(pad)
        applied.append(f"canvas({orig_w}x{orig_h}->{canvas_w}x{canvas_h}, x{scale:.3f})")
        # 종횡비가 캔버스와 크게 다르면 픽셀 예산의 상당량이 흰 여백으로 간다.
        # 조용히 넘기면 "왜 이 문서만 인식이 나쁘지" 로 헤매게 된다.
        waste = 1.0 - (canvas_w - pad[0]) * (canvas_h - pad[1]) / (canvas_w * canvas_h)
        if waste > 0.15:
            applied.append(
                f"경고: 종횡비가 캔버스와 달라 {waste * 100:.0f}% 가 흰 여백입니다 "
                f"(그만큼 글자가 작아집니다)"
            )
    elif target_long_side and max(orig_h, orig_w) > target_long_side:
        scale = target_long_side / max(orig_h, orig_w)
        img = cv2.resize(
            img,
            (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        transform["scale"] = scale
        applied.append(f"resize(scale={scale:.3f})")

    # 2) 기울기 보정
    if deskew:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = _estimate_skew(gray)
        if DESKEW_MIN_ANGLE <= abs(angle) <= DESKEW_MAX_ANGLE:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, matrix, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            transform["rotation_deg"] = angle
            transform["rotation_center"] = center
            applied.append(f"deskew(angle={angle:.2f})")
        elif abs(angle) > DESKEW_MAX_ANGLE:
            log.info("기울기 추정값 %.2f 도가 임계값을 넘어 보정을 건너뜁니다", angle)

    # 3) 패치 격자 정렬 — **오른쪽·아래에만 여백을 붙인다.**
    #
    # VLM 의 비전 인코더는 이미지를 factor(=패치×merge) 배수로 리사이즈한다.
    # 페이지가 배수가 아니면 조각 크기도 배수가 될 수 없고, 서버가 조각을 다시
    # 리샘플한다 — 10px 급 한글이 보간으로 뭉개지고 그건 곧 미탐이다.
    #
    # 자르지 않고 붙이는 이유: 기존 픽셀이 하나도 움직이지 않아 **좌표가 그대로
    # 유효하다.** 잘라내면 페이지 끝의 값을 잃는데, 그건 미탐이다.
    # canvas 를 썼으면 크기가 이미 고정이다. 여기서 또 붙이면 캔버스가 아니게 된다
    # (캔버스가 align 의 배수가 아니면 붙을 것이 있는데, 그건 설정 오류다).
    if align > 1 and canvas is None:
        h, w = img.shape[:2]
        pad_w = (-w) % align
        pad_h = (-h) % align
        if pad_w or pad_h:
            img = cv2.copyMakeBorder(
                img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            transform["align_pad"] = [pad_w, pad_h]
            applied.append(f"align({w}x{h}->{w + pad_w}x{h + pad_h}, /{align})")

    h, w = img.shape[:2]
    return PreprocessResult(
        image=img, width=w, height=h, transform=transform, applied=applied
    )
