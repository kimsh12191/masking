"""PDF 입력 처리.

PDF 를 페이지별 이미지로 렌더링해 파이프라인에 넣는다. 결과는 페이지마다
``{이름}_p001.boxes.png`` / ``{이름}_p001.json`` 으로 저장된다.

렌더러는 **pypdfium2** 를 쓴다 (BSD-3-Clause / Apache-2.0, PDFium 번들).
PyMuPDF 는 AGPL 이라 사내 상용 이용에 라이선스 검토가 필요하고, pdf2image 는
poppler 시스템 바이너리를 요구해 폐쇄망 반입이 번거롭다. pypdfium2 는 순수
pip 휠이라 반입이 간단하다.

.. note::
   전자적으로 생성된 PDF(스캔이 아닌)는 **텍스트 레이어에 정확한 좌표가 이미
   들어 있다.** 그런 문서는 OCR 없이 텍스트를 추출하는 편이 좌표가 완벽하고
   훨씬 빠르다. 현재는 모든 페이지를 이미지로 렌더링해 OCR 을 태운다 —
   스캔/전자 문서를 구분하지 않으므로 개선 여지가 있다.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: 렌더링 배율 1.0 은 72dpi 에 해당한다 (PDF 포인트 단위).
POINTS_PER_INCH = 72.0

#: 페이지가 비정상적으로 커도 이 배율을 넘기지 않는다 (메모리 방어).
MAX_SCALE = 8.0


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


@dataclass
class PdfPage:
    """렌더링된 PDF 페이지 하나.

    Attributes:
        page_no: 1부터 시작하는 페이지 번호.
        image: BGR numpy 배열 (OpenCV 관례. 파이프라인 전처리와 형식을 맞춘다).
        width: 렌더링된 픽셀 폭.
        height: 렌더링된 픽셀 높이.
        scale: 적용된 렌더링 배율.
        dpi: 실효 해상도.
    """

    page_no: int
    image: Any
    width: int
    height: int
    scale: float
    dpi: float


def parse_page_range(spec: str, n_pages: int) -> list[int]:
    """``"1-3,7,10-"`` 형식을 1-기반 페이지 번호 목록으로 바꾼다.

    Args:
        spec: 쉼표로 구분된 범위 표기. 빈 문자열이면 전체 페이지.
        n_pages: 문서의 총 페이지 수.

    Returns:
        정렬되고 중복 제거된 페이지 번호 목록.

    Raises:
        ValueError: 형식이 잘못되었거나 범위가 문서를 벗어날 때.
    """
    if not spec or not spec.strip():
        return list(range(1, n_pages + 1))

    pages: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if not re.fullmatch(r"\d*-?\d*", part) or part == "-":
            raise ValueError(f"페이지 범위 형식이 잘못되었습니다: {part!r}")

        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s) if lo_s else 1
            hi = int(hi_s) if hi_s else n_pages
        else:
            lo = hi = int(part)

        if lo < 1 or hi < lo:
            raise ValueError(f"페이지 범위가 잘못되었습니다: {part!r}")
        if lo > n_pages:
            raise ValueError(f"페이지 {lo} 는 문서 범위(1~{n_pages})를 벗어납니다")
        pages.update(range(lo, min(hi, n_pages) + 1))

    return sorted(pages)


def _scale_for(width_pt: float, height_pt: float, target_long_side: int | None) -> float:
    """긴 변이 목표 픽셀에 맞도록 렌더링 배율을 계산한다.

    전처리 단계에서 다시 축소하지 않도록 여기서 목표 크기로 바로 렌더링한다
    (두 번 리샘플링하면 작은 글씨가 열화된다).
    """
    if not target_long_side:
        return 2.0  # 144dpi 기본값
    long_pt = max(width_pt, height_pt)
    if long_pt <= 0:
        return 2.0
    return min(MAX_SCALE, max(0.5, target_long_side / long_pt))


def render_pages(
    pdf_path: str | Path,
    target_long_side: int | None = 2480,
    pages: str | None = None,
    password: str | None = None,
) -> Iterator[PdfPage]:
    """PDF 를 페이지별로 렌더링한다 (지연 평가 — 한 장씩 내보낸다).

    Args:
        pdf_path: 입력 PDF 경로.
        target_long_side: 긴 변 목표 픽셀. ``None`` 이면 144dpi 로 렌더링한다.
        pages: ``"1-3,7"`` 형식의 페이지 범위. ``None`` 이면 전체.
        password: 암호화된 PDF 의 열기 암호.

    Yields:
        ``PdfPage`` — BGR 배열과 페이지 번호.

    Raises:
        RuntimeError: pypdfium2 미설치, 또는 PDF 를 열 수 없을 때.
        ValueError: 페이지 범위가 잘못되었을 때.

    Note:
        메모리를 위해 제너레이터로 만들었다. 페이지 이미지는 A4 300dpi 기준
        장당 약 13MB 이므로 전체를 리스트로 모으면 큰 문서에서 터진다.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "PDF 입력에는 pypdfium2 가 필요합니다 (pip install pypdfium2). "
            "requirements.txt 를 참고하십시오."
        ) from exc

    path = Path(pdf_path)
    if not path.is_file():
        raise RuntimeError(f"PDF 를 찾을 수 없습니다: {path}")

    try:
        doc = pdfium.PdfDocument(str(path), password=password)
    except Exception as exc:  # noqa: BLE001 - pdfium 예외 종류가 버전마다 다르다
        raise RuntimeError(
            f"PDF 를 열 수 없습니다: {path}\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"  암호가 걸린 문서라면 --pdf-password 로 지정하십시오."
        ) from exc

    try:
        n_pages = len(doc)
        if n_pages == 0:
            raise RuntimeError(f"페이지가 없는 PDF 입니다: {path}")

        selected = parse_page_range(pages or "", n_pages)
        log.info("PDF %s: 전체 %d페이지 중 %d페이지 렌더링", path.name, n_pages, len(selected))

        for page_no in selected:
            page = doc[page_no - 1]
            width_pt, height_pt = page.get_size()
            scale = _scale_for(width_pt, height_pt, target_long_side)

            pil = page.render(scale=scale).to_pil().convert("RGB")
            # RGB -> BGR (OpenCV 관례에 맞춘다)
            arr = np.asarray(pil)[:, :, ::-1].copy()

            yield PdfPage(
                page_no=page_no,
                image=arr,
                width=pil.width,
                height=pil.height,
                scale=round(scale, 4),
                dpi=round(scale * POINTS_PER_INCH, 1),
            )
    finally:
        doc.close()


def page_count(pdf_path: str | Path, password: str | None = None) -> int:
    """페이지 수만 확인한다 (렌더링하지 않는다)."""
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError("pypdfium2 가 필요합니다.") from exc

    doc = pdfium.PdfDocument(str(pdf_path), password=password)
    try:
        return len(doc)
    finally:
        doc.close()
