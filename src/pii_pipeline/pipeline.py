"""파이프라인 오케스트레이션.

    이미지
      ├─① 전처리          기울기 보정 / 해상도 정규화
      ├─② OCR             det 임계값↓, rec 실패 박스도 번호 부여해 유지
      ├─③ 규칙 레이어      정규식 + 체크섬으로 정형 식별자 확정
      ├─④ LLM pass-1      OCR 텍스트만. 문맥 항목 분류.
      ├─④' VLM pass-2     이미지 직접 확인. OCR 이 구조적으로 놓친 것 회수.
      ├─⑤ 값 전파          확정된 값과 같은 텍스트를 가진 나머지 박스 회수.
      ├─⑥ 병합·검증        범위/중복/공간 검사, 제외 훅
      └─⑦ 결과 JSON

중복 차단은 ``ClaimLedger`` 로 **(박스, 라벨) 단위**로 한다. 박스 단위로 막으면
``"홍길동 901231-1234567"`` 처럼 한 박스에 두 종류가 섞였을 때 뒤쪽이 사라진다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .llm.client import LlmClient, LlmConfig
from .llm.prompts import (
    SYSTEM_PASS1,
    SYSTEM_PASS2,
    build_pass1_user,
    build_pass2_user,
)
from .merge import (
    claims_from_hits,
    confirmed_map,
    finalize,
    regions_from_pass1,
    regions_from_pass2,
    regions_from_rules,
)
from .ocr.paddle_runner import OcrConfig, PaddleOcrRunner
from .pdf import is_pdf, render_pages
from .preprocess import preprocess, preprocess_array
from .propagate import PropagateConfig, propagate_regions
from .rules.detectors import detect as rule_detect
from .schema import PASS1_SCHEMA, PASS2_SCHEMA, OcrBox, PageResult

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """파이프라인 설정.

    Attributes:
        enable_pass2: 이미지 검수 pass 사용 여부. 손글씨/도장이 있는 문서에서는
            반드시 켜야 한다 (OCR 이 못 읽은 것은 텍스트 pass 로 회수 불가).
        pass2_blind: ``True`` 면 1차 결과를 감춘다. 앵커링 편향은 없지만
            중복 판단이 늘어난다. 검증셋에서 두 방식을 비교할 것.
        target_long_side: 전처리 시 긴 변 목표 길이.
        deskew: 기울기 보정 여부.
        ocr: OCR 설정.
        llm: LLM 설정.
        propagate: 값 전파 설정. 한 곳에서 확정된 값과 같은 텍스트를 가진
            나머지 박스를 회수한다. 같은 이름/번호가 문서에 여러 번 나오는데
            일부만 탐지되는 상황을 메운다.
    """

    enable_pass2: bool = True
    pass2_blind: bool = False
    target_long_side: int | None = 2480
    deskew: bool = True
    ocr: OcrConfig = field(default_factory=OcrConfig.from_env)
    llm: LlmConfig = field(default_factory=LlmConfig)
    propagate: PropagateConfig = field(default_factory=PropagateConfig)


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
                    image,
                    target_long_side=cfg.target_long_side,
                    deskew=cfg.deskew,
                )
            else:
                pre = preprocess(
                    image_path,
                    target_long_side=cfg.target_long_side,
                    deskew=cfg.deskew,
                )
        page_w, page_h = pre.width, pre.height

        # ── ② OCR ────────────────────────────────────────────────
        with _timed(timings, "ocr"):
            boxes: list[OcrBox] = self.ocr.run(pre.image)

        result = PageResult(
            image_path=image_path,
            page_no=page_no,
            width=page_w,
            height=page_h,
            ocr_boxes=boxes,
            timings=timings,
            warnings=warnings,
            raw_llm=raw_llm,
            image=pre.image,  # 박스 오버레이 렌더링용. 직렬화 대상이 아니다.
        )
        if pre.applied:
            warnings.append("전처리 적용: " + ", ".join(pre.applied))
        if not boxes:
            warnings.append("OCR 박스가 검출되지 않았습니다")
            return result

        # ── ③ 규칙 레이어 ─────────────────────────────────────────
        with _timed(timings, "rules"):
            hits = rule_detect(boxes)
            rule_regions = regions_from_rules(hits, boxes, page_w, page_h)
            confirmed = confirmed_map(hits)

        # (박스, 라벨) 단위 원장. 규칙이 RRN 을 확정한 박스라도 같은 박스의
        # 이름은 아직 미확정이므로, LLM 이 그 박스를 NAME 으로 보고할 수 있어야
        # 한다. 박스 단위로 막으면 부분 마스킹 구현 시 그대로 유출된다.
        claimed = claims_from_hits(hits)
        regions = list(rule_regions)

        # ── ④ LLM pass-1 (텍스트) ─────────────────────────────────
        with _timed(timings, "llm_pass1"):
            payload, meta = self.llm.complete_json(
                system=SYSTEM_PASS1,
                user=build_pass1_user(boxes, page_w, page_h, confirmed),
                schema=PASS1_SCHEMA,
            )
            raw_llm["pass1"] = meta
            if meta.get("error"):
                warnings.append(f"pass1 실패: {meta['error']}")
            regions += regions_from_pass1(
                payload, boxes, claimed, page_w, page_h, warnings
            )

        # ── ④' VLM pass-2 (이미지) ────────────────────────────────
        if cfg.enable_pass2:
            with _timed(timings, "llm_pass2"):
                payload2, meta2 = self.llm.complete_json(
                    system=SYSTEM_PASS2,
                    user=build_pass2_user(
                        boxes,
                        page_w,
                        page_h,
                        confirmed=confirmed,
                        detected_idx=sorted(claimed.indices()),
                        blind=cfg.pass2_blind,
                    ),
                    schema=PASS2_SCHEMA,
                    image=pre.image,
                )
                raw_llm["pass2"] = meta2
                if meta2.get("error"):
                    warnings.append(f"pass2 실패: {meta2['error']}")
                regions += regions_from_pass2(
                    payload2, boxes, claimed, page_w, page_h, warnings
                )

        # ── ⑤ 값 전파 ─────────────────────────────────────────────
        # 같은 값이 문서 여러 곳에 나오는데 일부만 탐지되는 것은 정상적으로
        # 발생한다 (규칙은 박스 단위, pass-1 은 라벨 근처, pass-2 는 상위 몇 개).
        # 한 곳에서 확정됐으면 나머지 위치도 같은 개인정보다.
        with _timed(timings, "propagate"):
            propagated = propagate_regions(
                regions,
                boxes,
                page_w,
                page_h,
                claimed=claimed,
                config=cfg.propagate,
                warnings=warnings,
            )
            regions += propagated

        # ── ⑥ 병합·검증 ───────────────────────────────────────────
        with _timed(timings, "merge"):
            result.regions = finalize(regions, boxes, warnings)

        timings["total"] = sum(
            v for k, v in timings.items() if k != "total"
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
