"""파이프라인 오케스트레이션.

    이미지
      ├─① 전처리          기울기 보정 / 해상도 정규화
      ├─② OCR             det 임계값↓, rec 실패 박스도 번호 부여해 유지
      ├─③ 규칙 레이어      정규식 + 체크섬으로 정형 식별자 확정
      ├─④ LLM pass-1      OCR 텍스트만. 문맥 항목 분류.
      ├─④' VLM pass-2     이미지 직접 확인. OCR 이 구조적으로 놓친 것 회수.
      ├─⑤ 병합·검증        범위/중복/공간 검사, 제외 훅
      └─⑥ 결과 JSON
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
    confirmed_map,
    finalize,
    regions_from_pass1,
    regions_from_pass2,
    regions_from_rules,
)
from .ocr.paddle_runner import OcrConfig, PaddleOcrRunner
from .preprocess import preprocess
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
    """

    enable_pass2: bool = True
    pass2_blind: bool = False
    target_long_side: int | None = 2480
    deskew: bool = True
    ocr: OcrConfig = field(default_factory=OcrConfig.from_env)
    llm: LlmConfig = field(default_factory=LlmConfig)


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

    def run(self, image_path: str) -> PageResult:
        """이미지 1장을 처리한다.

        Args:
            image_path: 입력 이미지 경로.

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
            width=page_w,
            height=page_h,
            ocr_boxes=boxes,
            timings=timings,
            warnings=warnings,
            raw_llm=raw_llm,
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

        claimed: set[int] = set(confirmed)
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
                        detected_idx=sorted(claimed),
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

        # ── ⑤ 병합·검증 ───────────────────────────────────────────
        with _timed(timings, "merge"):
            result.regions = finalize(regions, boxes, warnings)

        timings["total"] = sum(
            v for k, v in timings.items() if k != "total"
        )
        return result

    def run_batch(self, image_paths: list[str]) -> list[PageResult]:
        """여러 장을 순차 처리한다. 1장이 실패해도 나머지는 계속 처리한다."""
        results: list[PageResult] = []
        for path in image_paths:
            try:
                results.append(self.run(path))
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
