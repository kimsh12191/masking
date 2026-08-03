"""vLLM (OpenAI 호환 서버) 클라이언트.

중요한 설정 두 가지:

* ``temperature=0`` — 결정론적 출력. 학습데이터 구축과 감사 대응에 필수다.
* ``enable_thinking=False`` — Qwen3 계열은 추론 모드가 켜지면 사고 토큰을 수천 개
  뱉는다. 분류 태스크에 이득이 없고 지연시간 예산만 소모한다.
"""

from __future__ import annotations

import asyncio
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
            **``DetectConfig.image_factor`` 의 배수로 둘 것** (1984 = 32×62).
            배수가 아니면 여기서 줄인 결과가 패치 격자와 안 맞아 서버가 다시
            리샘플한다 — ``detect`` 가 조각을 격자에 맞춰 놓은 것이 무효가 된다.
            **타일링과 함께 봐야 하는 값이다.** A4 를 ``target_long_side=2480``
            으로 전처리하면 1748x2480 이고, 그대로 보내면 여기서 0.73배로
            줄어 주민등록번호 숫자가 10px 대로 떨어진다 — 읽을 수 없다.
            ``DetectConfig.tiles=3`` 이면 타일 하나가 1748x약985 라서 긴 변이
            1748 이고 **축소가 아예 일어나지 않는다.** 타일 수를 줄이려면
            이 값을 함께 올려야 한다.
        max_retries: 스키마 위반/네트워크 오류 재시도 횟수.
        concurrency: **동시에 서버에 떠 있을 요청 수의 상한**
            (``AsyncLlmClient`` 만 본다).

            vLLM 은 서버에서 continuous batching 을 한다. 즉 클라이언트가 요청을
            모아 한 덩어리로 보낼 필요가 없고, **요청을 여러 개 띄워 놓기만
            하면** 서버가 알아서 같은 forward pass 에 태운다. 그래서 이 값은
            "배치 크기" 가 아니라 "동시에 떠 있는 수" 다.

            올리면 GPU 이용률이 오르고 처리량이 늘지만, 무한정 올릴 수는 없다.
            요청마다 KV 캐시가 필요하고, 서버의 ``--max-num-seqs`` 와
            ``--gpu-memory-utilization`` 이 실제 상한을 정한다. 그 상한을
            넘기면 서버가 요청을 큐에 쌓아 두므로 **지연시간만 늘고 처리량은
            그대로다** — 클라이언트에서 막는 것이 낫다.

            서버의 ``--max-num-seqs`` 이하로 두는 것이 기본이다.
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
    image_max_side: int = 1984
    max_retries: int = 2
    concurrency: int = 8
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    """호출 하나에 필요한 것 전부.

    ``complete_json`` 의 인자를 그대로 묶은 것이다. 여러 요청을 한꺼번에 넣는
    ``AsyncLlmClient.complete_json_many`` 가 목록으로 받기 위해 필요하다.

    Attributes:
        tag: 호출부가 결과를 되찾을 때 쓰는 표식 (타일 번호, 페이지 등).
            **클라이언트는 읽지 않고 메타에 그대로 되돌려준다.** 응답이 입력
            순서대로 오더라도 호출부가 순서에만 의존하지 않게 하려는 것이다.
    """

    system: str
    user: str
    schema: dict[str, Any]
    image: Any | None = None
    temperature: float | None = None
    tag: Any = None


@dataclass
class Outcome:
    """응답 하나를 해석한 결과.

    **이 판단을 sync 와 async 가 공유하는 것이 이 자료구조의 목적이다.**
    절단이냐 파싱 실패냐 복구 가능이냐에 따라 재시도 여부와 방법이 다른데,
    그 분기를 두 곳에 적어 두면 한쪽만 고쳐져 갈라진다. 갈라진 것은 예외가
    아니라 **탐지율 차이로만** 나타나서 눈에 띄지 않는다.

    Attributes:
        payload: 파싱된 dict. ``None`` 이면 실패다.
        salvaged: ``salvage_json`` 으로 건져냈다면 그 이유 (프롬프트 점검 신호).
        error: 실패 이유.
        truncated: ``max_tokens`` 에서 잘렸다. **재시도 방법이 다르다** —
            ``temperature=0`` + 같은 입력이면 같은 상한으로는 매번 같은 지점에서
            잘리므로, 상한을 올려야만 벗어난다.
    """

    payload: dict[str, Any] | None = None
    salvaged: str | None = None
    error: str | None = None
    truncated: bool = False


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


def interpret_response(
    raw: str, finish_reason: str | None, root_key: str | None
) -> Outcome:
    """응답 본문 하나를 해석한다. **순수 함수 — sync/async 가 공유한다.**

    절단을 먼저 본다. 잘린 JSON 은 파싱도 복구도 시도하지 않는다 — 뒤가 없는
    배열에서 건져낸 항목은 "앞부분만 탐지" 라는 뜻이고, 그걸 정상 결과로
    돌려주면 미탐이 무음으로 쌓인다. 상한을 올려 다시 받는 것이 맞다.

    Args:
        raw: 모델 원문.
        finish_reason: OpenAI 응답의 종료 이유.
        root_key: 복구할 때 감쌀 키 (``root_key_of`` 의 결과).

    Returns:
        해석 결과.
    """
    if finish_reason == "length":
        return Outcome(truncated=True)
    try:
        return Outcome(payload=json.loads(raw))
    except json.JSONDecodeError as exc:
        recovered = salvage_json(raw, root_key)
        if recovered is None:
            return Outcome(error=f"JSON 파싱 실패: {exc}")
        return Outcome(payload=recovered, salvaged=str(exc))


def response_parts(resp: Any) -> tuple[str, str | None, dict[str, int] | None]:
    """OpenAI 응답에서 필요한 것만 뽑는다 (``(원문, 종료이유, 토큰수)``).

    sync 와 async SDK 의 응답 객체가 같은 모양이라 그대로 공유한다.
    """
    choice = resp.choices[0]
    raw = (choice.message.content or "").strip()
    usage = None
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    return raw, choice.finish_reason, usage


class _ClientBase:
    """sync/async 클라이언트의 공통부 — 설정, 이미지 인코딩, 요청 조립.

    나누는 기준은 **await 가 필요한가**다. 요청을 만드는 일과 응답을 해석하는
    일에는 I/O 가 없으므로 여기 있고, 실제 호출과 재시도 루프만 각 클래스에
    있다. 그 둘도 ``interpret_response`` 를 공유하므로 판단은 한 곳뿐이다.
    """

    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config = config or LlmConfig()
        self._client: Any | None = None

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
    # 요청 조립
    # ------------------------------------------------------------------

    def _messages(self, system: str, user: str, image_url: str | None) -> list[dict]:
        content: Any = user
        if image_url is not None:
            content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def _extra_body(self, schema: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            **self.config.extra_body,
        }
        if self.config.guided:
            body["guided_json"] = schema
            body["guided_decoding_backend"] = self.config.guided_backend
        return body

    def _sampling(self, temperature: float | None) -> float:
        return self.config.temperature if temperature is None else temperature

    # ------------------------------------------------------------------
    # 재시도 루프의 공통 판단
    # ------------------------------------------------------------------

    def _resolve(
        self,
        outcome: Outcome,
        meta: dict[str, Any],
        max_tokens: int,
        attempt: int,
    ) -> tuple[dict[str, Any] | None, str | None, int]:
        """해석 결과를 보고 **돌려줄지 / 다시 할지**를 정한다.

        Returns:
            ``(반환할 payload 또는 None, 마지막 에러, 다음 시도의 max_tokens)``.
        """
        if outcome.truncated:
            error = f"출력이 max_tokens({max_tokens}) 에서 잘렸습니다"
            log.warning("%s (attempt %d)", error, attempt)
            if max_tokens < self.config.max_tokens_on_truncation:
                max_tokens = self.config.max_tokens_on_truncation
                log.warning("max_tokens 를 %d 로 올려 재시도합니다", max_tokens)
            return None, error, max_tokens

        if outcome.payload is None:
            log.warning("%s (attempt %d)", outcome.error, attempt)
            return None, outcome.error, max_tokens

        if outcome.salvaged:
            meta["salvaged"] = outcome.salvaged
            log.warning(
                "JSON 형식 이탈을 복구했습니다 (%s) — 프롬프트/guided decoding "
                "설정을 점검하십시오. 원문 앞부분: %.120s",
                outcome.salvaged,
                meta.get("raw", ""),
            )
        return outcome.payload, None, max_tokens


class LlmClient(_ClientBase):
    """구조화 JSON 응답을 강제하는 얇은 래퍼 (동기)."""

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
        image_url: str | None = None
        if image is not None:
            # 인코딩 실패도 예외로 던지지 않는다. 이 함수가 예외를 던지면
            # 이미 계산된 규칙 레이어 결과까지 함께 날아간다.
            try:
                image_url = self.encode_image(image)
            except Exception as exc:  # noqa: BLE001 - 페이지 전체를 잃지 않는다
                error = f"이미지 인코딩 실패: {type(exc).__name__}: {exc}"
                log.warning("%s", error)
                return {}, {"error": error, "attempt": 0}

        messages = self._messages(system, user, image_url)
        extra_body = self._extra_body(schema)
        root_key = root_key_of(schema)

        last_error: str | None = None
        max_tokens = self.config.max_tokens
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self._sampling(temperature),
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
            except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("LLM 호출 실패: %s (attempt %d)", last_error, attempt)
                continue

            raw, finish_reason, usage = response_parts(resp)
            meta: dict[str, Any] = {
                "raw": raw,
                "attempt": attempt,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
            }
            if usage:
                meta["usage"] = usage

            payload, last_error, max_tokens = self._resolve(
                interpret_response(raw, finish_reason, root_key), meta, max_tokens, attempt
            )
            if payload is not None:
                return payload, meta

        return {}, {"error": last_error or "unknown", "attempt": self.config.max_retries}


class AsyncLlmClient(_ClientBase):
    """요청 여러 개를 **동시에 띄우는** 클라이언트.

    왜 필요한가 — vLLM 은 서버에서 continuous batching 을 한다. 클라이언트가
    요청을 모아 한 덩어리로 보내는 것이 아니라, **요청이 여러 개 떠 있으면**
    서버가 알아서 같은 forward pass 에 태운다. 그래서 처리량을 올리는 방법은
    "in-flight 요청 수를 늘리는 것" 하나다.

    기존 동기 경로는 ``detect`` 가 스레드 풀로 **한 페이지의 타일**만 동시에
    돌린다. 페이지 사이는 완전히 순차라서, 한 페이지의 크롭 OCR·전처리(CPU)가
    도는 동안 vLLM 은 아무것도 받지 않고 논다. 페이지가 3타일이면 GPU 가 보는
    동시 요청은 최대 3개뿐이다.

    이 클래스는 그 상한을 걷어낸다 — 여러 페이지의 타일을 한꺼번에 띄우고,
    ``LlmConfig.concurrency`` 로만 제한한다.

    판단 로직은 동기 쪽과 **같은 함수**(``interpret_response`` / ``_resolve``)를
    쓴다. 절단·복구 처리가 두 경로에서 갈리면 "async 로 돌렸더니 탐지가 조금
    다르다" 가 되는데, 그건 원인을 찾기 어려운 종류의 차이다.
    """

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - 환경 의존
                raise RuntimeError(
                    "openai 패키지가 필요합니다 (vLLM OpenAI 호환 API 호출용). "
                    "requirements.txt 를 참고하세요."
                ) from exc
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    async def aclose(self) -> None:
        """HTTP 연결을 닫는다. 배치가 끝나면 부를 것."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        image: Any | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """스키마를 강제해 JSON 응답을 받는다 (비동기).

        동기 ``LlmClient.complete_json`` 과 **같은 것을 돌려준다** —
        ``(파싱된 dict, 메타)``. 실패해도 예외를 던지지 않는다.
        """
        image_url: str | None = None
        if image is not None:
            try:
                # **별 스레드에서 인코딩한다.** JPEG 인코딩은 CPU 작업이고,
                # 이벤트 루프에서 그대로 돌리면 그 시간만큼 다른 요청의
                # 송수신이 멈춘다. 요청을 수십 개 띄우는 것이 목적인데
                # 인코딩이 직렬화되면 그 목적이 사라진다.
                image_url = await asyncio.to_thread(self.encode_image, image)
            except Exception as exc:  # noqa: BLE001 - 페이지 전체를 잃지 않는다
                error = f"이미지 인코딩 실패: {type(exc).__name__}: {exc}"
                log.warning("%s", error)
                return {}, {"error": error, "attempt": 0}

        messages = self._messages(system, user, image_url)
        extra_body = self._extra_body(schema)
        root_key = root_key_of(schema)

        last_error: str | None = None
        max_tokens = self.config.max_tokens
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self._sampling(temperature),
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
            except asyncio.CancelledError:
                # 취소는 삼키지 않는다. 배치를 중단시키려는 신호이므로
                # 여기서 "실패한 요청" 으로 바꿔 버리면 중단이 안 된다.
                raise
            except Exception as exc:  # noqa: BLE001 - 배치 중단 방지
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("LLM 호출 실패: %s (attempt %d)", last_error, attempt)
                continue

            raw, finish_reason, usage = response_parts(resp)
            meta: dict[str, Any] = {
                "raw": raw,
                "attempt": attempt,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
            }
            if usage:
                meta["usage"] = usage

            payload, last_error, max_tokens = self._resolve(
                interpret_response(raw, finish_reason, root_key), meta, max_tokens, attempt
            )
            if payload is not None:
                return payload, meta

        return {}, {"error": last_error or "unknown", "attempt": self.config.max_retries}

    async def complete_json_many(
        self,
        requests: list[Request],
        concurrency: int | None = None,
        on_done: Any | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """요청 여러 개를 동시에 넣는다.

        Args:
            requests: 요청 목록.
            concurrency: 동시 요청 상한. ``None`` 이면 ``LlmConfig.concurrency``.
            on_done: 요청 하나가 끝날 때마다 부를 함수 ``(완료수, 전체수)``.
                진행 표시용이고, **완료 순서로 불린다** (제출 순서가 아니다).

        Returns:
            ``(payload, meta)`` 목록. **입력과 같은 순서, 같은 길이다.**
            실패한 요청도 자리를 지킨다 (``meta["error"]``) — 길이가 달라지면
            호출부가 결과를 입력에 되짚을 수 없다. ``Request.tag`` 를 준
            경우 ``meta["tag"]`` 로 되돌려주므로 순서에 의존하지 않아도 된다.
        """
        limit = max(1, concurrency or self.config.concurrency)
        # 세마포어는 **부를 때마다 새로 만든다.** 인스턴스에 캐시해 두면 다른
        # 이벤트 루프에서 재사용될 수 있고, 그때 조용히 오작동한다.
        gate = asyncio.Semaphore(limit)
        total = len(requests)
        done = 0

        async def one(request: Request) -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal done
            async with gate:
                payload, meta = await self.complete_json(
                    system=request.system,
                    user=request.user,
                    schema=request.schema,
                    image=request.image,
                    temperature=request.temperature,
                )
            if request.tag is not None:
                meta["tag"] = request.tag
            done += 1
            if on_done is not None:
                on_done(done, total)
            return payload, meta

        # gather 는 **제출 순서대로** 결과를 돌려준다 (완료 순서가 아니다).
        return list(await asyncio.gather(*(one(r) for r in requests)))
