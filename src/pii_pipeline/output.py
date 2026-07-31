"""결과 저장.

페이지 1장당 두 파일을 남긴다.

===================  =========================================================
``{stem}.boxes.png``  박스 영역이 표시된 이미지 (눈으로 검수하는 산출물)
``{stem}.json``       박스 위치 + 개인정보 유형 (다운스트림이 소비하는 산출물)
===================  =========================================================

두 파일의 좌표계는 동일하다 — 둘 다 **전처리된 이미지 기준**이다. 따라서
``boxes.png`` 위의 박스와 JSON 의 ``bbox`` 는 1:1로 대응한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .schema import PageResult
from .viz import draw_overlay

log = logging.getLogger(__name__)

#: 파일명 접미사
IMAGE_SUFFIX = ".boxes.png"
JSON_SUFFIX = ".json"

#: 다중 페이지 문서의 페이지 번호 자릿수 (``_p001``)
PAGE_DIGITS = 3


def default_stem(result: PageResult) -> str:
    """결과에서 파일명 기준 이름을 만든다.

    단일 이미지는 파일명 그대로, PDF 페이지는 ``{PDF이름}_p001`` 형태가 된다.
    페이지 번호를 0 채움으로 넣어 파일 정렬 순서가 페이지 순서와 일치하게 한다.
    """
    stem = Path(result.image_path).stem
    if result.page_no is None:
        return stem
    return f"{stem}_p{result.page_no:0{PAGE_DIGITS}d}"


def save_result(
    result: PageResult,
    out_dir: str | Path,
    stem: str | None = None,
    write_image: bool = True,
    include_ocr: bool = False,
    font_path: str | None = None,
    show_ocr_boxes: bool = False,
    release: bool = True,
) -> dict[str, Path]:
    """결과를 이미지 + JSON 으로 저장한다.

    Args:
        result: 파이프라인 결과.
        out_dir: 출력 디렉터리 (없으면 생성).
        stem: 파일명 기준 이름. 생략하면 입력 파일명을 쓰고,
            ``result.page_no`` 가 있으면 ``_p001`` 을 덧붙인다
            (PDF 1개 → ``doc_p001.json``, ``doc_p002.json`` …).
        write_image: 박스 표시 이미지를 남길지 여부.
        include_ocr: JSON 에 OCR 박스와 LLM 원시응답까지 포함할지 (디버깅용).
        font_path: 한글 폰트 경로. 없으면 라벨(ASCII)만 표시된다.
        show_ocr_boxes: OCR 박스 전체를 회색으로 함께 표시.
        release: 저장 후 ``result.image`` 참조를 해제할지 (배치 메모리 관리).

    Returns:
        ``{"json": Path, "image": Path}``. 이미지를 쓰지 않았으면 ``image`` 키가 없다.

    Note:
        ``result.image`` 가 비어 있으면 (전처리 이미지가 해제된 경우) 이미지 저장을
        건너뛰고 warning 을 남긴다. JSON 은 항상 저장된다.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    name = stem or default_stem(result)

    written: dict[str, Path] = {}

    json_path = out_path / f"{name}{JSON_SUFFIX}"
    json_path.write_text(result.to_json(include_ocr=include_ocr), encoding="utf-8")
    written["json"] = json_path

    if write_image:
        if result.image is None:
            result.warnings.append(
                "전처리 이미지가 없어 박스 표시 이미지를 저장하지 못했습니다"
            )
            log.warning("%s: result.image 가 비어 있어 이미지 저장을 건너뜁니다", name)
        else:
            image_path = out_path / f"{name}{IMAGE_SUFFIX}"
            canvas = draw_overlay(
                result.image,
                result,
                font_path=font_path,
                show_ocr_boxes=show_ocr_boxes,
            )
            canvas.save(image_path)
            written["image"] = image_path

    if release:
        result.release_image()

    return written


def format_summary(result: PageResult, written: dict[str, Path] | None = None) -> str:
    """콘솔 출력용 한 페이지 요약."""
    import json as _json

    stats = result.stats()
    header = result.image_path
    if result.page_no is not None:
        header += f"  (p{result.page_no})"
    lines = [f"=== {header} ==="]
    if written:
        for key in ("image", "json"):
            if key in written:
                lines.append(f"  {key:6s}    {written[key]}")
    # 지표 세 줄이 이 파이프라인의 성적표다. 단계가 두 개뿐이라 각 지표가
    # 어느 단계의 성능인지 1:1 로 대응한다.
    lines += [
        f"  ② VLM 탐지    {stats['n_findings']}건",
        f"  ③ 좌표 확정   {stats['localized_rate']:.0%}"
        f"  (영역 {stats['n_regions']}건, 크롭 OCR 박스 {stats['n_ocr_boxes']}개)",
        f"  ④ 불일치      {stats['n_disagreement']}건"
        f"  체크섬실패 {stats['n_checksum_failed']}건"
        f"  검토필요 {stats['n_needs_review']}건",
        f"  좌표출처      {_json.dumps(stats['by_source'], ensure_ascii=False)}",
        f"  유형별        {_json.dumps(stats['by_type'], ensure_ascii=False)}",
        f"  소요시간      {_json.dumps({k: round(v, 2) for k, v in result.timings.items()})}",
    ]
    lines += [f"  ! {w}" for w in result.warnings]
    return "\n".join(lines)
