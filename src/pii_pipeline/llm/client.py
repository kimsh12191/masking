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
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: ```json ... ``` 코드블록 껍데기.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def root_key_of(schema: dict[str, Any]) -> str | None:
    """스키마의 최상위 배열 키를 찾는다 (``findings``).

    복구할 때 "항목만 나열된 응답"을 어느 키로 감쌀지 알아야 한다.
    호출부가 따로 알려주지 않아도 되도록 스키마에서 끌어낸다.
    """
    required = schema.get("required") or []
    if len(required) == 1:
        return str(required[0])
    props = list((schema.get("properties") or {}).keys())
    return str(props[0]) if len(props) == 1 else None


def salvage_json(raw: str, root_key: str | None) -> dict[str, Any] | None:
    """규격을 벗어난 응답에서 쓸 수 있는 것을 건져낸다.

    guided decoding 이 실제로 걸리지 않은 서버에서는 모델이 형식을 흘린다.
    실제로 관측된 형태는 세 가지다.

    1. 항목을 한 줄에 하나씩 (JSONL) — ``{"idx":...}\\n{"idx":...}``
       → ``json.loads`` 가 ``Extra data: line 2 column 1`` 로 죽는다.
    2. 배열만 — ``[{"idx":...}, ...]``
    3. 코드블록·설명 문구가 앞뒤로 붙음.

    한 페이지의 pass 결과를 통째로 버리는 것보다 건져내는 쪽이 낫다.
    다만 **추측으로 값을 만들지는 않는다.** 파싱 가능한 객체만 모은다.

    Args:
        raw: 모델 원문.
        root_key: 감싸는 키. ``None`` 이면 첫 객체만 반환한다.

    Returns:
        복구한 dict, 건질 게 없으면 ``None``.
    """
    text = _FENCE_RE.sub("", raw.strip())

    # ② 배열만 온 경우
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and root_key:
            # 항목은 dict 만 남긴다. 다운스트림은 dict 를 가정한다.
            return {root_key: [o for o in parsed if isinstance(o, dict)]}
        return None

    # ①③ 객체가 여러 개 연달아 오거나 뒤에 잡음이 붙은 경우
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break  # JSON 이 아닌 텍스트가 나오면 거기서 멈춘다
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            objects.append(obj)

    if not objects:
        return None
    if root_key is None:
        return objects[0]

    # 감싼 객체가 (여러 개라도) 온 경우 — 배열을 이어붙인다
    wrapped = [o for o in objects if isinstance(o.get(root_key), list)]
    if wrapped:
        merged: list[Any] = []
        for obj in wrapped:
            merged.extend(o for o in obj[root_key] if isinstance(o, dict))
        return {root_key: merged}

    # 항목만 나열된 경우 — 우리가 감싸준다
    return {root_key: objects}


@dataclass
class LlmConfig:
    """vLLM 서버 접속 및 샘플링 설정.

    Attributes:
        base_url: vLLM OpenAI 호환 엔드포인트.
        model: 모델 이름 (vLLM 기동 시 지정한 것과 일치해야 함).
        api_key: 더미 값이어도 무관 (vLLM 은 검증하지 않음).
        temperature: 0 고정 권장.
        max_tokens: 출력 상한.
            **512 는 부족하다.** 한 항목이 ``text``/``field``/``bbox_norm`` 까지
            포함해 50~70토큰이다. 타일 하나에서 20건이 나오면 1400토큰이고,
            잘리면 그 타일의 탐지가 **전량** 날아간다 (JSON 이 불완전해짐).
        max_tokens_on_truncation: 출력이 잘렸을 때 재시도에 쓸 상한.
            ``temperature=0`` + 같은 입력이면 재시도해도 **똑같은 지점에서
            똑같이 잘린다.** 잘림은 상한을 올려야만 벗어날 수 있다.
        timeout: 초 단위 요청 타임아웃.
        guided_backend: guided decoding 백엔드.
        guided: guided decoding 을 걸지 여부. **끄지 마라 — 단 하나의 예외가
            ``enable_thinking=True`` 다.** 스키마를 강제하면 모델이 첫 토큰부터
            JSON 을 뱉어야 해서 **사고 토큰이 나올 자리가 없다.** 즉 추론 모드를
            켜 놓고 guided 를 함께 걸면 추론 모드가 조용히 무력화된다 — 켠 줄
            알고 측정하면 잘못된 결론을 얻는다. 끄면 ``salvage_json`` 이 형식을
            건져낸다.
        enable_thinking: Qwen3 계열 추론 모드. 운영에서는 **False** 다
            (지연시간을 먹고 guided 와 충돌한다). 측정·라벨 생성에서만 켠다.
        image_max_side: VLM 으로 보낼 이미지의 긴 변 길이.
            **타일링과 함께 봐야 하는 값이다.** A4 를 ``target_long_side=2480``
            으로 전처리하면 1748x2480 이고, 그대로 보내면 여기서 0.73배로
            줄어 주민등록번호 숫자가 10px 대로 떨어진다 — 읽을 수 없다.
            ``DetectConfig.tiles=3`` 이면 타일 하나가 1748x약985 라서 긴 변이
            1748 이고 **축소가 아예 일어나지 않는다.** 타일 수를 줄이려면
            이 값을 함께 올려야 한다.
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
    guided: bool = True
    enable_thinking: bool = False
    image_max_side: int = 2000
    max_retries: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)


def fit_max_side(height: int, width: int, max_side: int) -> tuple[int, int]:
    """긴 변을 ``max_side`` 로 맞춘 크기. 확대는 하지 않는다.

    **``detect`` 와 여기가 같은 계산을 써야 한다.** 모델이 실제로 본 이미지
    크기를 알아야 절대 픽셀 좌표를 옳게 환산할 수 있는데, 그 크기는 이 축소를
    거친 뒤의 값이다. 두 곳에 따로 적어 두면 한쪽만 바뀌었을 때 좌표가 조용히
    어긋나고, 화면에는 "박스가 전체적으로 밀렸다" 로만 보인다.

    Args:
        height: 원본 높이 (px).
        width: 원본 폭 (px).
        max_side: 긴 변 상한. 0 이하면 축소하지 않는다.

    Returns:
        ``(높이, 폭)``.
    """
    long_side = max(height, width)
    if max_side <= 0 or long_side <= max_side:
        return (height, width)
    scale = max_side / long_side
    return (max(1, int(height * scale)), max(1, int(width * scale)))


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

        h, w = fit_max_side(image.height, image.width, self.config.image_max_side)
        if (w, h) != image.size:
            image = image.resize((w, h), Image.LANCZOS)

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
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """스키마를 강제해 JSON 응답을 받는다.

        Args:
            system: 시스템 프롬프트 (고정 → prefix caching).
            user: user 메시지 텍스트.
            schema: guided decoding 용 JSON Schema.
            image: 있으면 멀티모달 요청으로 보낸다.
            temperature: 이 호출에만 적용할 온도. ``None`` 이면 설정값(0).
                다중 샘플링에서만 쓴다 — 0 이면 몇 번 뽑아도 같은 답이 온다.

        Returns:
            ``(파싱된 dict, 메타정보)``. 실패 시 dict 는 빈 값이고 메타에
            ``error`` 가 담긴다 — 예외를 던지지 않는다 (배치 처리 중단 방지).
            형식만 어긋난 응답을 ``salvage_json`` 으로 건져낸 경우 메타에
            ``salvaged`` 가 남는다 (프롬프트·guided decoding 점검 신호).
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
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            **self.config.extra_body,
        }
        if self.config.guided:
            extra_body["guided_json"] = schema
            extra_body["guided_decoding_backend"] = self.config.guided_backend

        last_error: str | None = None
        max_tokens = self.config.max_tokens
        root_key = root_key_of(schema)
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=(
                        self.config.temperature if temperature is None else temperature
                    ),
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
                try:
                    return json.loads(raw), meta
                except json.JSONDecodeError as exc:
                    # 재시도해도 temperature=0 이면 같은 응답이 온다. 형식만
                    # 어긋난 것이라면 건져내는 편이 페이지를 버리는 것보다 낫다.
                    recovered = salvage_json(raw, root_key)
                    if recovered is None:
                        raise
                    meta["salvaged"] = str(exc)
                    log.warning(
                        "JSON 형식 이탈을 복구했습니다 (%s) — 프롬프트/guided "
                        "decoding 설정을 점검하십시오. 원문 앞부분: %.120s",
                        exc,
                        raw,
                    )
                    return recovered, meta
            except json.JSONDecodeError as exc:
                last_error = f"JSON 파싱 실패: {exc}"
                log.warning("%s (attempt %d)", last_error, attempt)
            except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("LLM 호출 실패: %s (attempt %d)", last_error, attempt)

        return {}, {"error": last_error or "unknown", "attempt": self.config.max_retries}
