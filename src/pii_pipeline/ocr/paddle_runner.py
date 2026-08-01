"""PaddleOCR 래퍼 — 크롭 배치 인식 및 폐쇄망 대응.

이 파이프라인에서 OCR 은 **탐지기가 아니라 측량기**다. 무엇이 개인정보인지는
VLM 이 이미 판단했고, OCR 은 "그 값이 정확히 어느 픽셀에 있는가" 만 답한다.
그래서 페이지 전체가 아니라 **VLM 이 지목한 영역의 크롭들**을 받는다.

크롭 단위로 도는 것이 페이지 단위보다 유리한 이유:

1. **det 가 쉬워진다.** 밀집한 노이즈 표에서 페이지 det 는 인접 셀을 붙이거나
   한 셀을 쪼갠다. 한 칸짜리 크롭에서는 그런 모호함이 없다.
2. **업샘플이 가능하다.** 작은 크롭은 2배로 키워도 비용이 무시할 만하고,
   10px 짜리 글자에서 rec 정확도가 실제로 올라간다 (``locate.py`` 가 키운다).
3. **실패가 국소적이다.** 한 크롭의 rec 실패는 그 항목 하나만
   ``VLM_COARSE`` 로 떨어뜨린다. 페이지 전체 판단에 영향을 주지 않는다.

핵심 설계:

1. **det 임계값을 낮춰 넉넉하게 검출**한다. 검출기는 인식기보다 강해서
   손글씨/도장에 겹친 글자도 "여기 뭔가 있다"는 건 잡아낸다.
2. **rec 실패/저신뢰 박스를 버리지 않는다.** "글자는 있는데 못 읽었다" 는
   정보가 손글씨·도장 영역을 판별하는 근거다.
3. **모델 경로를 명시적으로 지정**한다. PaddleOCR 은 기본적으로 첫 실행 시
   모델을 자동 다운로드하는데, 폐쇄망에서는 여기서 실패한다.
   ``scripts/download_models.py`` 로 미리 받아 반입할 것.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import OcrBox, OcrStatus
from .layout import quad_to_bbox, sort_reading_order

log = logging.getLogger(__name__)


def parse_gpu_id(raw: str) -> int:
    """GPU 번호 문자열을 정수로 바꾼다.

    환경변수는 항상 문자열이므로 여기서 한 번만 검사한다. 오타를 조용히
    0 번으로 떨어뜨리면 "왜 여전히 vLLM 과 같은 GPU 를 쓰지" 로 헤매게 된다.

    Raises:
        ValueError: 정수가 아니거나 음수일 때.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GPU 번호는 정수여야 합니다: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"GPU 번호는 0 이상이어야 합니다: {value}")
    return value


@dataclass
class OcrConfig:
    """PaddleOCR 설정.

    Attributes:
        lang: 인식 언어. 한국어 금융문서는 ``"korean"``.
        det_model_dir: 검출 모델 디렉터리. 폐쇄망에서는 필수.
        rec_model_dir: 인식 모델 디렉터리. 폐쇄망에서는 필수.
        cls_model_dir: 방향분류 모델 디렉터리.
        det_db_thresh: 픽셀 이진화 임계값. 낮추면 흐린 글자도 검출.
        det_db_box_thresh: 박스 채택 임계값. **낮게 유지해야 손글씨가 살아남는다.**
        det_db_unclip_ratio: 검출 박스 확장 비율. 크면 글자 잘림이 줄어든다.
        rec_conf_ok: 이 값 이상이면 ``OcrStatus.OK``.
        rec_conf_floor: 이 값 미만이면 ``OcrStatus.FAILED`` 로 간주 (텍스트 신뢰 불가).
        use_gpu: GPU 사용 여부.
        gpu_id: 사용할 GPU 번호 (``nvidia-smi`` 의 인덱스). vLLM 이 0 번을 쓰고
            있으면 1 번 등으로 옮겨 메모리 경합을 피할 수 있다.
            ``CUDA_VISIBLE_DEVICES`` 가 설정되어 있으면 **그 목록 안에서의
            상대 번호**로 해석된다 (예: ``CUDA_VISIBLE_DEVICES=2,3`` 에서
            ``gpu_id=1`` 은 물리 2번이 아니라 3번).
        gpu_mem: PaddleOCR 에 허용할 GPU 메모리 (MB). vLLM 과 경합하므로 제한한다.
        max_side_len: 검출 입력 최대 변 길이. 작으면 작은 글씨를 놓친다.
    """

    lang: str = "korean"
    det_model_dir: str | None = None
    rec_model_dir: str | None = None
    cls_model_dir: str | None = None
    det_db_thresh: float = 0.20
    det_db_box_thresh: float = 0.30
    det_db_unclip_ratio: float = 1.8
    rec_conf_ok: float = 0.80
    rec_conf_floor: float = 0.35
    use_gpu: bool = True
    gpu_id: int = 0
    gpu_mem: int = 4000
    max_side_len: int = 2560

    @classmethod
    def from_env(cls) -> OcrConfig:
        """환경변수에서 모델 경로와 GPU 번호를 읽는다 (폐쇄망 배포 편의)."""
        config = cls(
            det_model_dir=os.getenv("PII_OCR_DET_DIR"),
            rec_model_dir=os.getenv("PII_OCR_REC_DIR"),
            cls_model_dir=os.getenv("PII_OCR_CLS_DIR"),
        )
        raw = os.getenv("PII_OCR_GPU_ID")
        if raw:
            config.gpu_id = parse_gpu_id(raw)
        return config

    def validate_offline(self) -> list[str]:
        """폐쇄망 실행 가능 여부를 점검하고 문제 목록을 반환한다."""
        problems: list[str] = []
        for name, path in (
            ("det_model_dir", self.det_model_dir),
            ("rec_model_dir", self.rec_model_dir),
        ):
            if not path:
                problems.append(
                    f"{name} 가 설정되지 않았습니다. 폐쇄망에서는 자동 다운로드가 "
                    f"실패하므로 scripts/download_models.py 로 미리 반입하고 경로를 지정하세요."
                )
            elif not Path(path).is_dir():
                problems.append(f"{name} 경로가 존재하지 않습니다: {path}")
        return problems


class PaddleOcrRunner:
    """PaddleOCR 실행기.

    ``paddleocr`` 패키지는 지연 임포트한다 (미설치 환경에서도 나머지 모듈이
    임포트 가능해야 테스트를 돌릴 수 있다).
    """

    def __init__(self, config: OcrConfig | None = None) -> None:
        self.config = config or OcrConfig.from_env()
        self._engine: Any | None = None

    # ------------------------------------------------------------------
    # 엔진 초기화
    # ------------------------------------------------------------------

    def _build_engine(self) -> Any:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise RuntimeError(
                "paddleocr 가 설치되어 있지 않습니다. requirements.txt 를 참고하세요. "
                "버전은 반드시 핀 고정하십시오 (2.x 와 3.x 의 API 가 다릅니다)."
            ) from exc

        problems = self.config.validate_offline()
        for p in problems:
            log.warning("OCR 설정 경고: %s", p)

        kwargs: dict[str, Any] = {
            "lang": self.config.lang,
            "use_angle_cls": True,
            "det_db_thresh": self.config.det_db_thresh,
            "det_db_box_thresh": self.config.det_db_box_thresh,
            "det_db_unclip_ratio": self.config.det_db_unclip_ratio,
            "det_limit_side_len": self.config.max_side_len,
            "det_limit_type": "max",
            "show_log": False,
        }
        if self.config.use_gpu:
            kwargs["use_gpu"] = True
            kwargs["gpu_id"] = self.config.gpu_id
            kwargs["gpu_mem"] = self.config.gpu_mem
            log.info(
                "OCR GPU 사용: gpu_id=%d gpu_mem=%dMB%s",
                self.config.gpu_id,
                self.config.gpu_mem,
                (
                    f" (CUDA_VISIBLE_DEVICES={visible} 기준 상대 번호)"
                    if (visible := os.getenv("CUDA_VISIBLE_DEVICES"))
                    else ""
                ),
            )
        else:
            kwargs["use_gpu"] = False

        for key, value in (
            ("det_model_dir", self.config.det_model_dir),
            ("rec_model_dir", self.config.rec_model_dir),
            ("cls_model_dir", self.config.cls_model_dir),
        ):
            if value:
                kwargs[key] = value

        return PaddleOCR(**kwargs)

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------

    def run(self, image: Any) -> list[OcrBox]:
        """이미지 1장에서 OCR 박스를 추출한다.

        Args:
            image: numpy 배열 (BGR) 또는 이미지 파일 경로.

        Returns:
            읽기 순서로 정렬되고 ``index`` 가 채워진 박스 목록.
            rec 실패 박스도 ``OcrStatus.FAILED`` 로 포함된다.
        """
        raw = self.engine.ocr(image, cls=True)
        return self.parse(raw)

    def run_many(self, images: list[Any]) -> list[list[OcrBox]]:
        """여러 크롭을 연속으로 인식한다.

        엔진을 한 번만 예열하고 크롭들을 이어서 넘긴다. PaddleOCR 2.x 의
        ``ocr()`` 은 배열 하나만 받으므로 여기서 순회하며, rec 배치는 크롭
        내부에서 Paddle 이 알아서 묶는다. **크롭 간 병렬화는 아니다** —
        이득은 엔진 재사용과 크롭당 입력이 작다는 점에서 나온다.

        Args:
            images: BGR numpy 배열 목록. 빈 배열이 섞여 있어도 된다.

        Returns:
            입력과 **같은 길이**의 결과 목록. 한 크롭이 실패하면 그 자리는
            빈 목록이 된다 (전체를 버리지 않는다). 좌표는 각 크롭 기준이므로
            호출부가 페이지 좌표로 환산해야 한다.
        """
        out: list[list[OcrBox]] = []
        for n, img in enumerate(images):
            if img is None or getattr(img, "size", 1) == 0:
                out.append([])
                continue
            try:
                out.append(self.run(img))
            except Exception as exc:  # noqa: BLE001 - 크롭 하나로 페이지를 버리지 않는다
                log.warning("크롭 %d OCR 실패: %s: %s", n, type(exc).__name__, exc)
                out.append([])
        return out

    def parse(self, raw: Any) -> list[OcrBox]:
        """PaddleOCR 원시 출력을 ``OcrBox`` 목록으로 변환한다.

        PaddleOCR 의 반환 형태는 버전마다 다르다. 2.x 는
        ``[[[quad, (text, conf)], ...]]`` (페이지 리스트로 한 겹 감싸짐).
        여기서는 페이지 한 겹을 벗기고 항목을 순회한다.
        """
        boxes: list[OcrBox] = []
        if not raw:
            return boxes

        page = raw[0] if isinstance(raw[0], list) else raw
        if page is None:
            return boxes

        for item in page:
            if not item:
                continue
            quad = item[0]
            payload = item[1] if len(item) > 1 else None

            text = ""
            rec_conf = 0.0
            if isinstance(payload, (list, tuple)) and payload:
                text = str(payload[0] or "")
                if len(payload) > 1 and payload[1] is not None:
                    rec_conf = float(payload[1])

            boxes.append(
                OcrBox(
                    index=-1,  # sort_reading_order 에서 부여
                    bbox=quad_to_bbox(quad),
                    text=text.strip(),
                    status=self._status(text, rec_conf),
                    rec_conf=round(rec_conf, 4),
                )
            )

        return sort_reading_order(boxes)

    def _status(self, text: str, conf: float) -> OcrStatus:
        if not text.strip() or conf < self.config.rec_conf_floor:
            # det 는 성공했으나 내용을 신뢰할 수 없다 -> 이미지 pass 로 넘긴다
            return OcrStatus.FAILED
        if conf < self.config.rec_conf_ok:
            return OcrStatus.LOW_CONF
        return OcrStatus.OK
