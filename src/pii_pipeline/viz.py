"""시각화 검수 도구.

탐지 결과를 눈으로 확인하기 위한 오버레이를 그린다. 학습데이터 구축 단계에서
**좌표가 실제로 맞는지 확인하는 가장 빠른 방법**이다.

라벨 텍스트는 ASCII (``NAME``, ``RRN`` 등) 만 사용하므로 한글 폰트가 없는
폐쇄망에서도 동작한다. OCR 원문까지 표시하려면 ``font_path`` 로 한글 폰트를 지정하라.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import PageResult, Source

#: 출처별 색상 (RGB). 티어를 한눈에 구분하기 위한 것.
SOURCE_COLORS: dict[Source, tuple[int, int, int]] = {
    Source.RULE: (0, 168, 89),           # 초록 — 체크섬 확정
    Source.LLM_PASS1: (0, 122, 214),     # 파랑 — 텍스트 pass
    Source.VLM_PASS2: (255, 149, 0),     # 주황 — 이미지 pass 회수
    Source.VLM_GROUNDING: (214, 45, 32), # 빨강 — 좌표 근사, 검토 필수
}


def _load_font(font_path: str | None, size: int) -> Any:
    from PIL import ImageFont  # type: ignore[import-not-found]

    if font_path and Path(font_path).is_file():
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_overlay(
    image: Any,
    result: PageResult,
    font_path: str | None = None,
    show_ocr_boxes: bool = False,
    line_width: int = 3,
) -> Any:
    """탐지 영역을 이미지 위에 그린다.

    Args:
        image: 배경 이미지 (PIL Image 또는 numpy BGR 배열).
        result: 파이프라인 결과.
        font_path: 한글 폰트 경로. 없으면 라벨만 ASCII 로 표시한다.
        show_ocr_boxes: OCR 박스 전체를 흐린 회색으로 함께 표시.
        line_width: 테두리 두께.

    Returns:
        오버레이가 그려진 PIL Image (원본은 변경하지 않는다).
    """
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    if not isinstance(image, Image.Image):
        import numpy as np  # type: ignore[import-not-found]

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # BGR -> RGB
        image = Image.fromarray(arr)

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font(font_path, size=max(11, canvas.height // 130))

    if show_ocr_boxes:
        for box in result.ocr_boxes:
            draw.rectangle(box.bbox, outline=(190, 190, 190), width=1)
            draw.text(
                (box.bbox[0] + 1, box.bbox[1] + 1),
                str(box.index),
                fill=(140, 140, 140),
                font=font,
            )

    for region in result.regions:
        color = SOURCE_COLORS.get(region.source, (128, 128, 128))
        x1, y1, x2, y2 = region.bbox

        # coarse 영역은 두껍게 + 내부 보조선으로 구분
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        if region.coarse:
            draw.rectangle((x1 + 3, y1 + 3, x2 - 3, y2 - 3), outline=color, width=1)

        label = f"{region.id} {region.type} {region.confidence:.2f}"
        if region.needs_review:
            label += " !"

        _draw_label(draw, label, (x1, y1, x2, y2), color, font, canvas.size)

    return canvas


def _draw_label(
    draw: Any,
    text: str,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    font: Any,
    canvas_size: tuple[int, int],
    pad: int = 3,
) -> None:
    """라벨 배지를 박스 **바깥**에 그린다.

    박스 내용을 가리면 검수를 할 수 없으므로 위쪽 여백에 붙이고, 위에 자리가
    없으면 아래쪽으로 넘긴다.

    잉크 bbox 의 오프셋(``ink[0]``, ``ink[1]``)을 빼서 글자를 배지 안에 정확히
    맞춘다. ``draw.text()`` 의 원점은 잉크 좌상단이 아니므로 이 보정을 빼면
    글자가 배지 밖으로 흘러 박스 내용을 덮는다.
    """
    x1, y1, x2, y2 = bbox
    page_w, page_h = canvas_size

    ink = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = ink[2] - ink[0], ink[3] - ink[1]
    badge_w, badge_h = text_w + pad * 2, text_h + pad * 2

    # 위쪽 우선, 자리가 없으면 아래쪽
    badge_y = y1 - badge_h if y1 - badge_h >= 0 else min(y2, page_h - badge_h)
    badge_y = max(0, badge_y)

    badge_x = min(x1, max(0, page_w - badge_w))

    draw.rectangle(
        (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h), fill=color
    )
    draw.text(
        (badge_x + pad - ink[0], badge_y + pad - ink[1]),
        text,
        fill=(255, 255, 255),
        font=font,
    )


def legend_text() -> str:
    """색상 범례를 문자열로 반환한다 (CLI 출력용)."""
    return "\n".join(
        [
            "색상 범례:",
            "  초록  rule           체크섬 통과. 검토 불필요.",
            "  파랑  llm_pass1      OCR 텍스트 기반 분류.",
            "  주황  vlm_pass2      이미지 pass 에서 회수. OCR 좌표 사용.",
            "  빨강  vlm_grounding  VLM 좌표 근사. 검토 필수.",
            "  '!'   needs_review 플래그",
        ]
    )
