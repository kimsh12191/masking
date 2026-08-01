"""파이프라인 오케스트레이션.

    이미지
      ├─① 전처리      기울기 보정 / 해상도 정규화
      ├─② VLM 탐지    이미지를 보고 개인정보를 **전사**한다 (좌표는 대략)
      ├─③ 크롭 OCR    지목된 영역을 잘라 배치 인식 → **정밀 좌표**
      ├─④ 검증        체크섬 · 자리수 · 항목명 필터
      └─⑤ 결과 JSON

단계가 두 개뿐인 것이 핵심이다 (②와 ③). 그래야 오차를 어느 단계에 귀속시킬 수
있고, 개선이 측정된다. ``PageResult.stats()`` 의 세 지표가 정확히 그 용도다.

    n_findings      ② 가 몇 건을 찾았나         -> VLM 재현율
    localized_rate  ③ 이 몇 %의 좌표를 잡았나   -> 기하 성능
    n_disagreement  두 엔진이 다르게 읽은 건수  -> 신뢰도

이전 구조는 규칙 레이어 + 텍스트 LLM pass + 이미지 VLM pass + 값 전파 + 병합
다섯 단계였고, 단계 간 계약(``<CONFIRMED>`` 태그, 라벨셋 분리, (박스,라벨) 원장)이
버그의 원인이었다. 오차를 어디에 귀속시킬 수도 없었다.

**순서가 뒤집혀 있었다는 것이 근본 문제였다.** 개인정보 판단은 의미 문제고
좌표는 기하 문제다. 예전에는 기하(OCR)를 먼저 돌리고 그 결과를 의미 판단의
입력 형식으로 강제했기 때문에, VLM 이 "박스 번호 고르기" 를 해야 했다 —
자기가 가장 못하는 일을. 지금은 판단을 VLM 에, 좌표를 OCR 에 맡긴다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .detect import DetectConfig, detect
from .llm.client import LlmClient, LlmConfig
from .locate import LocateConfig, locate
from .ocr.paddle_runner import OcrConfig, PaddleOcrRunner
from .pdf import is_pdf, render_pages
from .preprocess import preprocess, preprocess_array
from .schema import PageResult
from .verify import VerifyConfig, finalize, verify_regions

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """파이프라인 설정.

    Attributes:
        target_long_side: 전처리 시 긴 변 목표 길이. 결과 좌표계의 기준이 된다.
            높이면 크롭 해상도가 올라가지만 VLM 입력도 커진다 (타일링이 이를
            흡수한다 — ``DetectConfig.tiles`` 참조).
        deskew: 기울기 보정 여부.
        ocr: OCR 설정. 이제 OCR 은 페이지 전체가 아니라 **크롭에만** 돈다.
        llm: vLLM 접속/샘플링 설정.
        detect: ② VLM 탐지 설정 (타일 수 등).
        locate: ③ 좌표 확정 설정 (크롭 패딩, 업샘플 등).
        verify: ④ 검증 설정.
    """

    target_long_side: int | None = 2480
    deskew: bool = True
    ocr: OcrConfig = field(default_factory=OcrConfig.from_env)
    llm: LlmConfig = field(default_factory=LlmConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    locate: LocateConfig = field(default_factory=LocateConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)


@contextmanager
def _timed(timings: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - start


class PiiPipeline:
    """개인정보 영역 탐지 파이프라인.

    OCR 엔진과 LLM 클라이언트는 인스턴스 수명 동안 재사용된다
    (모델 로딩 비용이 크므로 배치 처리 시 인스턴스를 하나만 만들 것).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.ocr = PaddleOcrRunner(self.config.ocr)
        self.llm = LlmClient(self.config.llm)

    def run(
        self,
        image_path: str,
        *,
        image: Any | None = None,
        page_no: int | None = None,
    ) -> PageResult:
        """페이지 1장을 처리한다.

        Args:
            image_path: 입력 경로. ``image`` 를 함께 주면 파일을 읽지 않고
                결과의 출처 표기로만 쓴다 (PDF 페이지 등).
            image: 이미 메모리에 있는 BGR 배열. 주면 파일 읽기를 건너뛴다.
            page_no: 다중 페이지 문서의 1-기반 페이지 번호.

        Returns:
            탐지된 영역과 진단 정보를 담은 ``PageResult``.
            좌표는 **전처리된 이미지 기준**이다.
        """
        cfg = self.config
        timings: dict[str, float] = {}
        warnings: list[str] = []
        raw_llm: dict[str, Any] = {}

        # ── ① 전처리 ──────────────────────────────────────────────
        with _timed(timings, "preprocess"):
            if image is not None:
                pre = preprocess_array(
                    image, target_long_side=cfg.target_long_side, deskew=cfg.deskew
                )
            else:
                pre = preprocess(
                    image_path, target_long_side=cfg.target_long_side, deskew=cfg.deskew
                )

        result = PageResult(
            image_path=image_path,
            page_no=page_no,
            width=pre.width,
            height=pre.height,
            timings=timings,
            warnings=warnings,
            raw_llm=raw_llm,
            image=pre.image,  # 박스 오버레이 렌더링용. 직렬화 대상이 아니다.
        )
        if pre.applied:
            warnings.append("전처리 적용: " + ", ".join(pre.applied))

        # ── ② VLM 탐지 ────────────────────────────────────────────
        with _timed(timings, "detect"):
            findings, metas = detect(pre.image, self.llm, cfg.detect, warnings)
            raw_llm["vlm"] = metas
            result.findings = findings

        if not findings:
            # 빈 결과와 "호출이 실패해서 빈 결과" 를 구분해야 한다. guided
            # decoding 은 형식만 보장하므로, 과부하 상태의 모델이 내는 가장 싼
            # 스키마 적합 응답이 빈 배열이다. 그걸 "개인정보 없음" 으로 조용히
            # 넘기면 미탐이 무음으로 쌓인다.
            if any(m.get("error") for m in metas):
                warnings.append("VLM 호출이 실패해 탐지 결과가 없습니다 (개인정보 없음이 아님)")
            else:
                warnings.append("VLM 이 개인정보를 찾지 못했습니다")
            return result

        # ── ③ 크롭 OCR 로 좌표 확정 ───────────────────────────────
        with _timed(timings, "locate"):
            regions, boxes = locate(
                findings, pre.image, self.ocr, cfg.locate, warnings
            )
            result.ocr_boxes = boxes

        # ── ④ 검증 · 최종 정리 ────────────────────────────────────
        with _timed(timings, "verify"):
            verify_regions(regions, cfg.verify, warnings)
            result.regions = finalize(regions, cfg.verify, warnings)

        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return result

    def run_pdf(
        self,
        pdf_path: str,
        out_dir: str | None = None,
        pages: str | None = None,
        password: str | None = None,
        **save_kwargs: Any,
    ) -> list[PageResult]:
        """PDF 를 페이지별로 처리한다.

        페이지를 한 장씩 렌더링해 처리하고, ``out_dir`` 을 주면 즉시 저장하고
        이미지를 해제한다. 전체를 메모리에 모으지 않으므로 페이지가 많은 문서도
        처리할 수 있다.

        Args:
            pdf_path: 입력 PDF 경로.
            out_dir: 지정하면 페이지마다 즉시 저장한다
                (``{PDF이름}_p001.boxes.png`` / ``{PDF이름}_p001.json``).
                생략하면 모든 결과가 이미지를 물고 있어 메모리를 많이 쓴다.
            pages: ``"1-3,7"`` 형식의 페이지 범위. ``None`` 이면 전체.
            password: 암호화된 PDF 의 열기 암호.
            **save_kwargs: ``save_result()`` 로 전달.

        Returns:
            페이지 순서대로의 결과 목록. 한 페이지가 실패해도 나머지는 계속 처리하고,
            실패한 페이지는 빈 결과 + ``warnings`` 로 남는다.

        Raises:
            RuntimeError: PDF 자체를 열 수 없을 때 (페이지 단위 실패와 구분한다).
            ValueError: 페이지 범위 표기가 잘못되었을 때.
        """
        results: list[PageResult] = []

        for page in render_pages(
            pdf_path,
            target_long_side=self.config.target_long_side,
            pages=pages,
            password=password,
        ):
            try:
                result = self.run(pdf_path, image=page.image, page_no=page.page_no)
                result.warnings.append(
                    f"PDF 렌더링: {page.width}x{page.height}px @ {page.dpi}dpi"
                )
            except Exception as exc:  # noqa: BLE001 - 페이지 하나로 문서 전체를 버리지 않는다
                log.exception("PDF 페이지 처리 실패: %s p%d", pdf_path, page.page_no)
                results.append(
                    PageResult(
                        image_path=pdf_path,
                        page_no=page.page_no,
                        width=page.width,
                        height=page.height,
                        warnings=[f"처리 실패: {type(exc).__name__}: {exc}"],
                    )
                )
                continue

            if out_dir is not None:
                from .output import save_result

                save_result(result, out_dir, release=True, **save_kwargs)
            results.append(result)

        return results

    def run_any(
        self,
        path: str,
        out_dir: str | None = None,
        **kwargs: Any,
    ) -> list[PageResult]:
        """확장자를 보고 이미지/PDF 를 알아서 처리한다.

        Args:
            path: 이미지 또는 PDF 경로.
            out_dir: 지정하면 즉시 저장한다.
            **kwargs: PDF 면 ``pages`` / ``password`` 도 받는다. 나머지는
                ``save_result()`` 로 전달된다.

        Returns:
            결과 목록. 단일 이미지면 길이 1.
        """
        if is_pdf(path):
            return self.run_pdf(path, out_dir=out_dir, **kwargs)

        kwargs.pop("pages", None)
        kwargs.pop("password", None)
        result = self.run(path)
        if out_dir is not None:
            from .output import save_result

            save_result(result, out_dir, release=True, **kwargs)
        return [result]

    def run_batch(
        self,
        image_paths: list[str],
        out_dir: str | None = None,
        **save_kwargs: Any,
    ) -> list[PageResult]:
        """이미지와 PDF 를 섞어서 순차 처리한다. 하나가 실패해도 계속 처리한다.

        Args:
            image_paths: 입력 경로 목록. 이미지와 PDF 를 섞어도 된다.
            out_dir: 지정하면 한 장씩 즉시 저장하고 전처리 이미지를 해제한다.
                생략하면 모든 결과가 이미지를 물고 있으므로 (장당 ~13MB)
                많은 페이지를 처리할 때 메모리를 주의해야 한다.
            **save_kwargs: ``save_result()`` 로 전달 (write_image, font_path 등).
                PDF 용 ``pages`` / ``password`` 도 여기로 넘긴다.

        Returns:
            결과 목록. **입력 개수와 길이가 다를 수 있다** — PDF 1개가 여러
            페이지 결과를 낸다. 실패한 항목은 빈 결과 + ``warnings`` 로 남는다.
        """
        results: list[PageResult] = []
        for path in image_paths:
            try:
                results.extend(self.run_any(path, out_dir=out_dir, **save_kwargs))
            except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
                log.exception("처리 실패: %s", path)
                results.append(
                    PageResult(
                        image_path=path,
                        width=0,
                        height=0,
                        warnings=[f"처리 실패: {type(exc).__name__}: {exc}"],
                    )
                )
        return results
