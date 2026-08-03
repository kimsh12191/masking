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

import asyncio
import logging
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .detect import (
    DetectConfig,
    collect_findings,
    detect,
    label_metas,
    plan_tiles,
    tile_requests,
)
from .llm.client import AsyncLlmClient, LlmClient, LlmConfig
from .locate import LocateConfig, locate
from .ocr.paddle_runner import OcrConfig, PaddleOcrRunner
from .pdf import is_pdf, render_pages
from .preprocess import preprocess, preprocess_array
from .schema import PageResult
from .verify import VerifyConfig, finalize, verify_regions

log = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """비동기 배치 실행 설정 (``PiiPipeline.run_batch_async``).

    **왜 이 설정이 따로 있는가** — 동기 경로는 페이지를 한 장씩 끝까지 처리한다.
    한 페이지의 전처리(CPU)와 크롭 OCR(다른 GPU 또는 같은 GPU)이 도는 동안
    vLLM 은 아무 요청도 받지 않는다. 페이지가 3타일이면 vLLM 이 보는 동시
    요청은 많아야 3개다. 배치 경로는 여러 페이지의 타일을 한꺼번에 띄워
    그 빈 시간을 없앤다.

    Attributes:
        pages: **한 묶음으로 VLM 요청을 넣을 페이지 수.**

            이 값이 정하는 것은 두 가지다. (1) 동시에 띄울 수 있는 요청의 총량
            (``pages × tiles × samples``), (2) 동시에 메모리에 있는 전처리
            이미지 수. 전처리된 A4 한 장이 1760x2464x3 바이트라 **약 13MB** 이고,
            타일 크롭과 JPEG 인코딩 버퍼가 그 위에 얹힌다. 크게 잡으면
            처리량이 오르지만 메모리를 그만큼 쓴다.

            실제 동시 요청 수는 ``llm.concurrency`` 가 다시 한 번 제한한다.
            즉 ``pages`` 는 "얼마나 미리 준비해 둘까", ``concurrency`` 는
            "동시에 몇 개를 서버에 띄울까" 다. ``pages × tiles`` 가
            ``concurrency`` 보다 충분히 커야 서버가 굶지 않는다.
        prefetch_workers: 전처리를 돌릴 스레드 수. 0 이면 ``pages`` 만큼
            (상한 8). 전처리는 OpenCV 연산이라 GIL 밖에서 돌아 스레드로
            병렬화된다. VLM 이 앞 묶음을 처리하는 동안 다음 묶음을 미리
            준비해 두는 것이 목적이다.
    """

    pages: int = 8
    prefetch_workers: int = 0


@dataclass
class PipelineConfig:
    """파이프라인 설정.

    Attributes:
        canvas: ``(폭, 높이)`` **고정 캔버스.** 주면 모든 페이지가 정확히 이
            크기가 되고 ``target_long_side`` 는 무시된다. 종횡비를 지키며
            확대·축소한 뒤 오른쪽·아래를 흰색으로 채운다.

            **좌표 학습을 하려면 이쪽이다.** 페이지가 고정이라야 타일도 고정이고,
            모델이 상대할 기하가 하나뿐이다. ``detect.image_factor`` 의 배수로 둘 것.
        target_long_side: 전처리 시 긴 변 목표 길이. ``canvas`` 가 있으면 무시된다.
            높이면 크롭 해상도가 올라가지만 VLM 입력도 커진다 (타일링이 이를
            흡수한다 — ``DetectConfig.tiles`` 참조).
            **``detect.image_factor`` 의 배수로 둘 것.** A4 300dpi 는 2480 이지만
            32 의 배수가 아니라 2464(=32×77)를 쓴다. 페이지가 격자에 맞아야
            조각도 맞고, 그래야 서버가 조각을 다시 리샘플하지 않는다.
        deskew: 기울기 보정 여부.
        ocr: OCR 설정. 이제 OCR 은 페이지 전체가 아니라 **크롭에만** 돈다.
        llm: vLLM 접속/샘플링 설정.
        detect: ② VLM 탐지 설정 (타일 수 등).
        locate: ③ 좌표 확정 설정 (크롭 패딩, 업샘플 등).
        verify: ④ 검증 설정.
        batch: 비동기 배치 실행 설정. 동기 경로는 보지 않는다.
    """

    canvas: tuple[int, int] | None = (1760, 2464)
    target_long_side: int | None = 2464
    deskew: bool = True
    ocr: OcrConfig = field(default_factory=OcrConfig.from_env)
    llm: LlmConfig = field(default_factory=LlmConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    locate: LocateConfig = field(default_factory=LocateConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)


#: 전처리 프리페치 스레드 상한. 전처리는 CPU 작업이라 코어 수를 넘겨도 이득이 없다.
_MAX_PREFETCH = 8


@contextmanager
def _timed(timings: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - start


def _chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """제너레이터를 ``size`` 개씩 끊어 낸다. **미리 다 꺼내지 않는다.**

    입력이 PDF 페이지 제너레이터라 이 성질이 중요하다 — 전체를 리스트로
    만들면 페이지가 많은 문서에서 메모리가 터진다.
    """
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class PiiPipeline:
    """개인정보 영역 탐지 파이프라인.

    OCR 엔진과 LLM 클라이언트는 인스턴스 수명 동안 재사용된다
    (모델 로딩 비용이 크므로 배치 처리 시 인스턴스를 하나만 만들 것).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.ocr = PaddleOcrRunner(self.config.ocr)
        self.llm = LlmClient(self.config.llm)
        #: 비동기 배치 경로용. 실제 HTTP 클라이언트는 첫 호출에 만들어지므로
        #: 동기 경로만 쓰는 사용자에게 비용이 없다.
        self.async_llm = AsyncLlmClient(self.config.llm)

    async def aclose(self) -> None:
        """비동기 클라이언트의 연결을 닫는다. 배치가 끝나면 부를 것."""
        await self.async_llm.aclose()

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
        result = self.preprocess_page(image_path, image=image, page_no=page_no)

        # ── ② VLM 탐지 ────────────────────────────────────────────
        with _timed(result.timings, "detect"):
            findings, metas = detect(
                result.image, self.llm, self.config.detect, result.warnings
            )

        return self.finish_page(result, findings, metas)

    # ------------------------------------------------------------------
    # 단계 — ``run`` 과 비동기 배치 경로가 공유한다
    # ------------------------------------------------------------------

    def preprocess_page(
        self,
        image_path: str,
        *,
        image: Any | None = None,
        page_no: int | None = None,
    ) -> PageResult:
        """① 전처리만 한다. ``result.image`` 에 전처리된 배열이 담긴다.

        **VLM 호출 전까지의 모든 일**이 여기 있다. 비동기 배치 경로는 여러
        페이지를 먼저 여기까지 진행시켜 놓고 요청을 한꺼번에 넣는다.
        """
        cfg = self.config
        timings: dict[str, float] = {}
        warnings: list[str] = []

        with _timed(timings, "preprocess"):
            if image is not None:
                pre = preprocess_array(
                    image,
                    target_long_side=cfg.target_long_side,
                    deskew=cfg.deskew,
                    canvas=cfg.canvas,
                    align=cfg.detect.image_factor,
                )
            else:
                pre = preprocess(
                    image_path,
                    target_long_side=cfg.target_long_side,
                    deskew=cfg.deskew,
                    canvas=cfg.canvas,
                    align=cfg.detect.image_factor,
                )

        result = PageResult(
            image_path=image_path,
            page_no=page_no,
            width=pre.width,
            height=pre.height,
            timings=timings,
            warnings=warnings,
            raw_llm={},
            image=pre.image,  # 박스 오버레이 렌더링용. 직렬화 대상이 아니다.
        )
        if pre.applied:
            warnings.append("전처리 적용: " + ", ".join(pre.applied))
        return result

    def finish_page(
        self,
        result: PageResult,
        findings: list[Any],
        metas: list[dict[str, Any]],
    ) -> PageResult:
        """③④ — 좌표를 확정하고 검증한다. VLM 결과를 받은 뒤의 모든 일.

        동기 경로와 비동기 배치 경로가 **이 함수를 공유한다.** 여기가 갈리면
        "async 로 돌렸더니 결과가 조금 다르다" 가 되는데, 그 차이는 원인을
        찾기 어렵다.
        """
        cfg = self.config
        warnings = result.warnings
        result.raw_llm["vlm"] = metas
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
        with _timed(result.timings, "locate"):
            regions, boxes = locate(
                findings, result.image, self.ocr, cfg.locate, warnings
            )
            result.ocr_boxes = boxes

        # ── ④ 검증 · 최종 정리 ────────────────────────────────────
        with _timed(result.timings, "verify"):
            verify_regions(regions, cfg.verify, warnings)
            result.regions = finalize(regions, cfg.verify, warnings)

        result.timings["total"] = sum(
            v for k, v in result.timings.items() if k != "total"
        )
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

    # ------------------------------------------------------------------
    # 비동기 배치
    # ------------------------------------------------------------------

    def page_units(
        self,
        image_paths: list[str],
        pages: str | None = None,
        password: str | None = None,
    ) -> Iterator[tuple[str, Any | None, int | None, list[str]]]:
        """입력 경로들을 **페이지 단위로 펼친다.** PDF 는 페이지마다 하나씩 낸다.

        제너레이터인 것이 중요하다 — PDF 를 통째로 렌더링해 두면 페이지가 많은
        문서에서 메모리가 터진다. 배치 경로는 한 묶음이 필요할 때만 여기서
        꺼내 쓴다.

        Yields:
            ``(경로, 이미지 또는 None, 페이지 번호 또는 None, 초기 경고)``.
            이미지가 ``None`` 이면 경로에서 읽으라는 뜻이다 (단일 이미지).
        """
        for path in image_paths:
            if not is_pdf(path):
                yield path, None, None, []
                continue
            try:
                rendered = render_pages(
                    path,
                    target_long_side=self.config.target_long_side,
                    pages=pages,
                    password=password,
                )
                for page in rendered:
                    yield path, page.image, page.page_no, [
                        f"PDF 렌더링: {page.width}x{page.height}px @ {page.dpi}dpi"
                    ]
            except Exception as exc:  # noqa: BLE001 - PDF 하나로 배치를 버리지 않는다
                log.exception("PDF 열기 실패: %s", path)
                yield path, None, None, [
                    f"__failed__PDF 열기 실패: {type(exc).__name__}: {exc}"
                ]

    async def run_batch_async(
        self,
        image_paths: list[str],
        out_dir: str | None = None,
        pages: str | None = None,
        password: str | None = None,
        on_progress: Any | None = None,
        **save_kwargs: Any,
    ) -> list[PageResult]:
        """여러 페이지의 VLM 요청을 **한꺼번에 띄워서** 처리한다.

        동기 ``run_batch`` 와 결과는 같고 순서도 같다. 다른 것은 요청을 넣는
        방식뿐이다.

            동기    페이지1 [전처리 → VLM(타일 병렬) → OCR → 검증] → 페이지2 ...
            배치    페이지 N장 전처리 → **N×타일 요청을 동시에** → 페이지별 OCR·검증

        페이지 사이가 순차일 때 vLLM 이 보는 동시 요청은 많아야 ``tiles`` 개다.
        그 사이 전처리와 크롭 OCR 이 도는 시간에는 0개다. 이 함수는 한 묶음
        (``batch.pages``)의 타일을 모아 한 번의 ``complete_json_many`` 로 넣으므로,
        동시 요청 상한이 페이지 단위가 아니라 **묶음 전체**에 걸린다.

        Args:
            image_paths: 입력 경로 목록. 이미지와 PDF 를 섞어도 된다.
            out_dir: 지정하면 페이지마다 즉시 저장하고 전처리 이미지를 해제한다.
                **긴 배치에서는 지정할 것** — 생략하면 모든 결과가 이미지를
                물고 있어 장당 ~13MB 가 쌓인다.
            pages: PDF 페이지 범위 (``"1-3,7"``).
            password: 암호화된 PDF 의 열기 암호.
            on_progress: ``(완료 페이지수, 지금까지 펼친 페이지수)`` 로 불린다.
                전체 페이지 수는 PDF 를 열기 전까지 알 수 없으므로 두 번째
                인자도 **진행 중에 늘어난다.**
            **save_kwargs: ``save_result()`` 로 전달.

        Returns:
            입력 순서대로의 결과 목록. 한 페이지가 실패해도 나머지는 처리한다.

        Note:
            배치 모드의 ``timings["detect"]`` 는 **묶음 전체의 벽시계 시간을
            페이지 수로 나눈 값**이다. 요청이 겹쳐 돌기 때문에 페이지별 VLM
            시간이라는 것이 존재하지 않는다 — 처리량 관점의 페이지당 원가로
            읽어라. 묶음의 원래 시간은 ``timings["detect_window"]`` 에 있다.
        """
        cfg = self.config
        results: list[PageResult] = []
        n_seen = 0

        for window in _chunked(
            self.page_units(image_paths, pages=pages, password=password),
            max(1, cfg.batch.pages),
        ):
            n_seen += len(window)
            # **묶음 안의 순서를 지킨다.** 성공한 것만 모아 처리하고 나중에
            # 이어 붙이면 실패한 페이지가 앞으로 몰려 입력 순서가 깨진다.
            outs = await self._prepare_window(window)
            prepared = [o for o in outs if isinstance(o, tuple)]

            if prepared:
                await self._detect_window(prepared)

            for out in outs:
                result = out[0] if isinstance(out, tuple) else out
                results.append(result)
                if out_dir is not None:
                    from .output import save_result

                    save_result(result, out_dir, release=True, **save_kwargs)
            del outs, prepared  # 타일 이미지 참조를 빨리 놓아준다
            if on_progress is not None:
                on_progress(len(results), n_seen)

        return results

    async def _prepare_window(
        self, window: list[tuple[str, Any | None, int | None, list[str]]]
    ) -> list[tuple[PageResult, Any, int] | PageResult]:
        """묶음을 전처리하고 타일 계획을 세운다. **스레드로 병렬화한다.**

        전처리는 OpenCV 연산이라 대부분 GIL 밖에서 돌고, 이벤트 루프에서 직접
        돌리면 그 시간 동안 앞 묶음의 응답 수신이 멈춘다.

        Returns:
            **입력과 같은 순서의** 목록. 항목은 성공하면
            ``(결과, 계획, 요청수)``, 실패하면 ``PageResult`` 하나다.
            성공한 것만 걸러 돌려주면 호출부가 순서를 복원할 수 없다.
        """
        cfg = self.config
        workers = cfg.batch.prefetch_workers or min(max(1, len(window)), _MAX_PREFETCH)

        def prepare(
            unit: tuple[str, Any | None, int | None, list[str]],
        ) -> tuple[PageResult, Any, int] | PageResult:
            path, image, page_no, notes = unit
            failure = next((n for n in notes if n.startswith("__failed__")), None)
            if failure is not None:
                return PageResult(
                    image_path=path,
                    page_no=page_no,
                    width=0,
                    height=0,
                    warnings=[failure.removeprefix("__failed__")],
                )
            try:
                result = self.preprocess_page(path, image=image, page_no=page_no)
                result.warnings.extend(notes)
                with _timed(result.timings, "plan"):
                    plan = plan_tiles(
                        result.image,
                        cfg.detect,
                        self.async_llm.config.image_max_side,
                        result.warnings,
                    )
                return result, plan, len(plan.calls)
            except Exception as exc:  # noqa: BLE001 - 페이지 하나로 배치를 버리지 않는다
                log.exception("전처리 실패: %s", path)
                return PageResult(
                    image_path=path,
                    page_no=page_no,
                    width=0,
                    height=0,
                    warnings=[f"처리 실패: {type(exc).__name__}: {exc}"],
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # gather 는 제출 순서대로 돌려준다 — 완료 순서가 아니다.
            return list(
                await asyncio.gather(
                    *(
                        asyncio.get_running_loop().run_in_executor(pool, prepare, unit)
                        for unit in window
                    )
                )
            )

    async def _detect_window(
        self, prepared: list[tuple[PageResult, Any, int]]
    ) -> None:
        """묶음의 타일 요청을 **한 번에** 넣고, 페이지별로 조립·좌표확정까지 한다."""
        cfg = self.config

        requests: list[Any] = []
        offsets: list[int] = []
        for index, (_, plan, count) in enumerate(prepared):
            offsets.append(len(requests))
            requests.extend(tile_requests(plan, tag=index))
            assert count == len(plan.calls)

        start = time.perf_counter()
        raw = await self.async_llm.complete_json_many(requests)
        elapsed = time.perf_counter() - start

        log.info(
            "VLM 묶음: 페이지 %d장, 요청 %d개, %.1f초 (동시 상한 %d)",
            len(prepared),
            len(requests),
            elapsed,
            cfg.llm.concurrency,
        )

        for index, (result, plan, count) in enumerate(prepared):
            chunk = raw[offsets[index] : offsets[index] + count]
            metas = label_metas(chunk, plan)
            findings = collect_findings(metas, plan, cfg.detect, result.warnings)
            # 요청이 겹쳐 돌므로 페이지별 VLM 시간이라는 것이 없다.
            # 처리량 관점의 페이지당 원가를 적고, 원래 값도 함께 남긴다.
            result.timings["detect"] = elapsed / len(prepared)
            result.timings["detect_window"] = elapsed
            self.finish_page(result, findings, metas)
