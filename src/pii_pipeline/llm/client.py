"""vLLM (OpenAI 호환 서버) 클라이언트.

중요한 설정 두 가지:

* ``temperature=0`` — 결정론적 출력. 학습데이터 구축과 감사 대응에 필수다.
* ``enable_thinking=False`` — Qwen3 계열은 추론 모드가 켜지면 사고 토큰을 수천 개
  뱉는다. 분류 태스크에 이득이 없고 지연시간 예산만 소모한다.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LlmConfig:
    """vLLM 서버 접속 및 샘플링 설정.

    Attributes:
        base_url: vLLM OpenAI 호환 엔드포인트.
        model: 모델 이름 (vLLM 기동 시 지정한 것과 일치해야 함).
        api_key: 더미 값이어도 무관 (vLLM 은 검증하지 않음).
        temperature: 0 고정 권장.
        max_tokens: 출력 상한.
            **512 는 부족하다.** 한 항목이 pass-1 은 약 20토큰, pass-2 는
            ``reason`` 까지 포함해 40~60토큰이다. 밀집한 신청서(주민등록등본,
            대출신청서)는 탐지가 25건을 쉽게 넘기므로 512 에서는 JSON 이
            중간에 잘린다. 잘리면 그 페이지의 LLM 탐지가 **전량** 날아간다.
        max_tokens_on_truncation: 출력이 잘렸을 때 재시도에 쓸 상한.
            ``temperature=0`` + 같은 입력이면 재시도해도 **똑같은 지점에서
            똑같이 잘린다.** 잘림은 상한을 올려야만 벗어날 수 있다.
        timeout: 초 단위 요청 타임아웃.
        guided_backend: guided decoding 백엔드.
        enable_thinking: Qwen3 계열 추론 모드. **반드시 False.**
        image_max_side: pass-2 로 보낼 이미지의 긴 변 길이.
            너무 줄이면 작은 글씨를 못 읽어 pass-2 의 존재 의미가 사라진다.
        max_retries: 스키마 위반/네트워크 오류 재시도 횟수.
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3.5-9B"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_tokens_on_truncation: int = 4096
    timeout: float = 60.0
    guided_backend: str = "xgrammar"
    enable_thinking: bool = False
    image_max_side: int = 1800
    max_retries: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)


class LlmClient:
    """구조화 JSON 응답을 강제하는 얇은 래퍼."""

    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config = config or LlmConfig()
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - 환경 의존
                raise RuntimeError(
                    "openai 패키지가 필요합니다 (vLLM OpenAI 호환 API 호출용). "
                    "requirements.txt 를 참고하세요."
                ) from exc
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # 이미지 인코딩
    # ------------------------------------------------------------------

    def encode_image(self, image: Any) -> str:
        """PIL/numpy 이미지를 data URL 로 변환한다. 긴 변을 설정값으로 축소한다."""
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise RuntimeError("Pillow 가 필요합니다.") from exc

        if not isinstance(image, Image.Image):
            import numpy as np  # type: ignore[import-not-found]

            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]  # BGR -> RGB
            image = Image.fromarray(arr)

        max_side = self.config.image_max_side
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    # ------------------------------------------------------------------
    # 호출
    # ------------------------------------------------------------------

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        image: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """스키마를 강제해 JSON 응답을 받는다.

        Args:
            system: 시스템 프롬프트 (고정 → prefix caching).
            user: user 메시지 텍스트.
            schema: guided decoding 용 JSON Schema.
            image: 있으면 멀티모달 요청으로 보낸다 (pass-2).

        Returns:
            ``(파싱된 dict, 메타정보)``. 실패 시 dict 는 빈 값이고 메타에
            ``error`` 가 담긴다 — 예외를 던지지 않는다 (배치 처리 중단 방지).
        """
        content: Any = user
        if image is not None:
            # 인코딩 실패도 예외로 던지지 않는다. 이 함수가 예외를 던지면
            # 이미 계산된 규칙 레이어 결과까지 함께 날아간다.
            try:
                content = [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": self.encode_image(image)}},
                ]
            except Exception as exc:  # noqa: BLE001 - 페이지 전체를 잃지 않는다
                error = f"이미지 인코딩 실패: {type(exc).__name__}: {exc}"
                log.warning("%s", error)
                return {}, {"error": error, "attempt": 0}

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        extra_body: dict[str, Any] = {
            "guided_json": schema,
            "guided_decoding_backend": self.config.guided_backend,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            **self.config.extra_body,
        }

        last_error: str | None = None
        max_tokens = self.config.max_tokens
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                raw = (resp.choices[0].message.content or "").strip()
                meta = {
                    "raw": raw,
                    "attempt": attempt,
                    "finish_reason": resp.choices[0].finish_reason,
                    "max_tokens": max_tokens,
                }
                if getattr(resp, "usage", None):
                    meta["usage"] = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                if meta["finish_reason"] == "length":
                    # max_tokens 에서 잘렸다면 JSON 이 불완전하다.
                    # temperature=0 + 같은 입력이면 같은 상한으로는 매번 같은
                    # 지점에서 잘린다. 상한을 올려야 재시도에 의미가 생긴다.
                    last_error = (
                        f"출력이 max_tokens({max_tokens}) 에서 잘렸습니다"
                    )
                    log.warning("%s (attempt %d)", last_error, attempt)
                    if max_tokens < self.config.max_tokens_on_truncation:
                        max_tokens = self.config.max_tokens_on_truncation
                        log.warning("max_tokens 를 %d 로 올려 재시도합니다", max_tokens)
                    continue
                return json.loads(raw), meta
            except json.JSONDecodeError as exc:
                last_error = f"JSON 파싱 실패: {exc}"
                log.warning("%s (attempt %d)", last_error, attempt)
            except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("LLM 호출 실패: %s (attempt %d)", last_error, attempt)

        return {}, {"error": last_error or "unknown", "attempt": self.config.max_retries}
