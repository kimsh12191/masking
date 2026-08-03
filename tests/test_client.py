"""LLM 클라이언트 테스트.

네트워크를 타지 않고, **실패했을 때 페이지 전체를 잃지 않는지**를 본다.
규칙 레이어 결과는 LLM 이 죽어도 살아남아야 한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline.llm.client import (
    AsyncLlmClient,
    LlmClient,
    LlmConfig,
    Request,
    interpret_response,
    root_key_of,
    salvage_json,
)
from pii_pipeline.schema import VLM_SCHEMA


def fake_response(content: str, finish_reason: str = "stop"):
    """``content`` 를 그대로 돌려주는 최소 OpenAI 호환 응답."""

    class Msg:
        pass

    Msg.content = content

    class Choice:
        message = Msg()

    Choice.finish_reason = finish_reason

    class Resp:
        choices = [Choice()]
        usage = None

    class Fake:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs: Any) -> Any:
                    return Resp()

    return Fake


class TestCompleteJsonFailures:
    def test_image_encoding_failure_returns_error_not_raises(self) -> None:
        """인코딩 실패가 예외로 새면 이미 계산된 규칙 레이어 결과까지 날아간다."""
        client = LlmClient(LlmConfig())
        payload, meta = client.complete_json(
            system="s", user="u", schema=VLM_SCHEMA, image=object()
        )
        assert payload == {}
        assert "이미지 인코딩 실패" in meta["error"]

    def test_encoding_failure_does_not_call_the_api(self, monkeypatch) -> None:
        client = LlmClient(LlmConfig())
        called: list[int] = []
        monkeypatch.setattr(
            type(client), "client", property(lambda self: called.append(1))
        )
        client.complete_json(system="s", user="u", schema=VLM_SCHEMA, image=object())
        assert called == []

    def test_api_failure_returns_error_not_raises(self, monkeypatch) -> None:
        class Boom:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        raise ConnectionError("연결 거부")

        client = LlmClient(LlmConfig(max_retries=0))
        monkeypatch.setattr(type(client), "client", property(lambda self: Boom))
        payload, meta = client.complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert payload == {}
        assert "연결 거부" in meta["error"]

    def test_bad_json_is_retried_then_reported(self, monkeypatch) -> None:
        attempts: list[int] = []

        class Msg:
            content = "이건 JSON 이 아니다"

        class Choice:
            message = Msg()
            finish_reason = "stop"

        class Resp:
            choices = [Choice()]
            usage = None

        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        attempts.append(1)
                        return Resp()

        client = LlmClient(LlmConfig(max_retries=2))
        monkeypatch.setattr(type(client), "client", property(lambda self: Fake))
        payload, meta = client.complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert payload == {}
        assert "JSON 파싱 실패" in meta["error"]
        assert len(attempts) == 3  # 최초 1회 + 재시도 2회

    def test_truncated_output_is_treated_as_failure(self, monkeypatch) -> None:
        """max_tokens 에서 잘리면 JSON 이 불완전하므로 성공으로 보지 않는다."""

        class Msg:
            content = '{"findings":'

        class Choice:
            message = Msg()
            finish_reason = "length"

        class Resp:
            choices = [Choice()]
            usage = None

        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        return Resp()

        client = LlmClient(LlmConfig(max_retries=0))
        monkeypatch.setattr(type(client), "client", property(lambda self: Fake))
        payload, meta = client.complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert payload == {}
        assert "max_tokens" in meta["error"]


class TestSalvage:
    """guided decoding 이 실제로 걸리지 않은 서버에서 관측된 형식 이탈.

    현장 로그: ``JSON 파싱 실패: Extra data: line 2 column 1``
    — 모델이 항목을 한 줄에 하나씩 뱉었다. 그 타일의 결과가 전량 날아갔다.
    """

    def test_root_key_from_schema(self) -> None:
        assert root_key_of(VLM_SCHEMA) == "findings"

    def test_jsonl_items_are_wrapped(self) -> None:
        raw = (
            '{"text":"홍길동","type":"NAME","field":"성명","bbox_2d":[0.1,0.2,0.3,0.4],"conf":0.9}\n'
            '{"text":"010-1234-5678","type":"PHONE","field":"연락처","bbox_2d":[0.1,0.5,0.4,0.6],"conf":0.8}'
        )
        out = salvage_json(raw, "findings")
        assert out is not None
        assert [i["text"] for i in out["findings"]] == ["홍길동", "010-1234-5678"]

    def test_bare_array_is_wrapped(self) -> None:
        assert salvage_json('[{"idx":[1],"type":"NAME"}]', "missed") == {
            "missed": [{"idx": [1], "type": "NAME"}]
        }

    def test_code_fence_is_stripped(self) -> None:
        raw = '```json\n{"missed":[{"idx":[3],"type":"RRN"}]}\n```'
        assert salvage_json(raw, "missed") == {"missed": [{"idx": [3], "type": "RRN"}]}

    def test_repeated_wrapper_objects_are_merged(self) -> None:
        raw = '{"missed":[{"idx":[1],"type":"NAME"}]}\n{"missed":[{"idx":[2],"type":"RRN"}]}'
        assert salvage_json(raw, "missed") == {
            "missed": [{"idx": [1], "type": "NAME"}, {"idx": [2], "type": "RRN"}]
        }

    def test_trailing_prose_is_ignored(self) -> None:
        raw = '{"missed":[]}\n이상입니다. 더 필요하면 알려주세요.'
        assert salvage_json(raw, "missed") == {"missed": []}

    def test_non_dict_items_are_dropped(self) -> None:
        """다운스트림(merge)은 항목이 dict 라고 가정한다."""
        assert salvage_json('[1, {"idx":[1],"type":"NAME"}, "x"]', "missed") == {
            "missed": [{"idx": [1], "type": "NAME"}]
        }

    def test_unrecoverable_returns_none(self) -> None:
        assert salvage_json("이건 JSON 이 아니다", "missed") is None
        assert salvage_json("", "missed") is None
        assert salvage_json('{"missed": ', "missed") is None

    def test_client_recovers_jsonl_and_marks_meta(self, monkeypatch) -> None:
        raw = (
            '{"text":"홍길동","type":"NAME","field":"성명","bbox_2d":[0.1,0.2,0.3,0.4],"conf":0.9}\n'
            '{"text":"010-1234-5678","type":"PHONE","field":"연락처","bbox_2d":[0.1,0.5,0.4,0.6],"conf":0.8}'
        )
        client = LlmClient(LlmConfig(max_retries=0))
        monkeypatch.setattr(
            type(client), "client", property(lambda self: fake_response(raw))
        )
        payload, meta = client.complete_json(
            system="s", user="u", schema=VLM_SCHEMA
        )
        assert [i["type"] for i in payload["findings"]] == ["NAME", "PHONE"]
        assert "Extra data" in meta["salvaged"]
        assert "error" not in meta

    def test_unsalvageable_still_reports_error(self, monkeypatch) -> None:
        client = LlmClient(LlmConfig(max_retries=0))
        monkeypatch.setattr(
            type(client), "client", property(lambda self: fake_response("설명만 있다"))
        )
        payload, meta = client.complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert payload == {}
        assert "JSON 파싱 실패" in meta["error"]


class TestRequestShape:
    @pytest.fixture
    def captured(self, monkeypatch):
        seen: dict[str, Any] = {}

        class Msg:
            content = '{"missed":[]}'

        class Choice:
            message = Msg()
            finish_reason = "stop"

        class Resp:
            choices = [Choice()]
            usage = None

        class Fake:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        seen.update(kwargs)
                        return Resp()

        def build(**cfg: Any) -> LlmClient:
            client = LlmClient(LlmConfig(**cfg))
            monkeypatch.setattr(type(client), "client", property(lambda self: Fake))
            return client

        return build, seen

    def test_thinking_is_disabled(self, captured) -> None:
        """추론 모드가 켜지면 사고 토큰을 수천 개 뱉어 지연시간 예산을 날린다."""
        build, seen = captured
        build().complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert seen["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_temperature_is_zero(self, captured) -> None:
        build, seen = captured
        build().complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert seen["temperature"] == 0.0

    def test_schema_is_enforced(self, captured) -> None:
        build, seen = captured
        build().complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert seen["extra_body"]["guided_json"] == VLM_SCHEMA
        assert seen["extra_body"]["guided_decoding_backend"] == "xgrammar"

    def test_text_only_request_has_string_content(self, captured) -> None:
        build, seen = captured
        build().complete_json(system="s", user="u", schema=VLM_SCHEMA)
        assert seen["messages"][1]["content"] == "u"


# --------------------------------------------------------------------------
# 응답 해석 — sync/async 가 공유하는 판단
# --------------------------------------------------------------------------


class TestInterpretResponse:
    """이 함수가 sync/async 의 **유일한 공통 판단**이다.

    절단이냐 파싱 실패냐 복구 가능이냐에 따라 재시도 방법이 다르고, 그 분기가
    두 경로에서 갈리면 "async 로 돌렸더니 탐지가 조금 다르다" 가 된다.
    """

    def test_clean_json_passes_through(self) -> None:
        out = interpret_response('{"findings":[]}', "stop", "findings")
        assert out.payload == {"findings": []}
        assert out.salvaged is None
        assert not out.truncated

    def test_truncation_is_not_salvaged(self) -> None:
        """잘린 배열에서 건져내면 '앞부분만 탐지' 를 정상 결과로 돌려준다.

        그건 미탐이 무음으로 쌓이는 길이다. 상한을 올려 다시 받아야 한다.
        """
        out = interpret_response('{"findings":[{"a":1}', "length", "findings")
        assert out.truncated
        assert out.payload is None

    def test_recoverable_json_is_marked(self) -> None:
        out = interpret_response('{"a":1}\n{"a":2}', "stop", "findings")
        assert out.payload == {"findings": [{"a": 1}, {"a": 2}]}
        assert out.salvaged is not None  # 프롬프트 점검 신호가 남아야 한다

    def test_unrecoverable_json_is_an_error(self) -> None:
        out = interpret_response("설명만 있고 JSON 이 없다", "stop", "findings")
        assert out.payload is None
        assert out.error is not None
        assert not out.truncated


# --------------------------------------------------------------------------
# 비동기 클라이언트
# --------------------------------------------------------------------------


class FakeAsyncApi:
    """동시에 몇 개가 떠 있었는지 기록하는 최소 AsyncOpenAI 대역."""

    def __init__(self, content: str = '{"findings":[]}', delay: float = 0.01) -> None:
        self.content = content
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        outer = self

        class completions:  # noqa: N801
            @staticmethod
            async def create(**kwargs: Any) -> Any:
                import asyncio

                outer.in_flight += 1
                outer.peak = max(outer.peak, outer.in_flight)
                outer.calls += 1
                try:
                    await asyncio.sleep(outer.delay)
                finally:
                    outer.in_flight -= 1
                return fake_response(outer.content)().chat.completions.create()

        class chat:  # noqa: N801
            pass

        chat.completions = completions
        self.chat = chat


def async_client(api: Any, **cfg: Any) -> AsyncLlmClient:
    client = AsyncLlmClient(LlmConfig(**cfg))
    client._client = api  # noqa: SLF001 - 네트워크를 타지 않게 직접 꽂는다
    return client


class TestAsyncLlmClient:
    def test_returns_the_same_shape_as_the_sync_client(self) -> None:
        import asyncio

        api = FakeAsyncApi('{"findings":[{"text":"x"}]}')
        payload, meta = asyncio.run(
            async_client(api).complete_json(system="s", user="u", schema=VLM_SCHEMA)
        )
        assert payload == {"findings": [{"text": "x"}]}
        assert meta["attempt"] == 0
        assert meta["finish_reason"] == "stop"

    def test_api_failure_returns_error_not_raises(self) -> None:
        import asyncio

        class Boom:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs: Any) -> Any:
                        raise RuntimeError("연결 거부")

        payload, meta = asyncio.run(
            async_client(Boom, max_retries=1).complete_json(
                system="s", user="u", schema=VLM_SCHEMA
            )
        )
        assert payload == {}
        assert "연결 거부" in meta["error"]

    def test_many_preserves_input_order(self) -> None:
        """순서가 흔들리면 호출부가 결과를 입력에 되짚을 수 없다."""
        import asyncio

        api = FakeAsyncApi()
        requests = [
            Request(system="s", user=f"u{i}", schema=VLM_SCHEMA, tag=i)
            for i in range(6)
        ]
        out = asyncio.run(async_client(api).complete_json_many(requests))
        assert [meta["tag"] for _, meta in out] == [0, 1, 2, 3, 4, 5]

    def test_many_returns_one_result_per_request(self) -> None:
        import asyncio

        api = FakeAsyncApi()
        requests = [Request(system="s", user="u", schema=VLM_SCHEMA) for _ in range(5)]
        out = asyncio.run(async_client(api).complete_json_many(requests))
        assert len(out) == len(requests)

    def test_concurrency_cap_is_enforced(self) -> None:
        """상한이 없으면 서버가 요청을 큐에 쌓아 지연시간만 늘어난다."""
        import asyncio

        api = FakeAsyncApi()
        requests = [Request(system="s", user="u", schema=VLM_SCHEMA) for _ in range(12)]
        asyncio.run(async_client(api, concurrency=3).complete_json_many(requests))
        assert api.calls == 12
        assert api.peak <= 3

    def test_requests_actually_overlap(self) -> None:
        """상한만 지키고 직렬로 돌면 이 기능의 목적이 사라진다."""
        import asyncio

        api = FakeAsyncApi()
        requests = [Request(system="s", user="u", schema=VLM_SCHEMA) for _ in range(8)]
        asyncio.run(async_client(api, concurrency=4).complete_json_many(requests))
        assert api.peak > 1

    def test_per_call_concurrency_overrides_the_config(self) -> None:
        import asyncio

        api = FakeAsyncApi()
        requests = [Request(system="s", user="u", schema=VLM_SCHEMA) for _ in range(10)]
        asyncio.run(
            async_client(api, concurrency=8).complete_json_many(requests, concurrency=2)
        )
        assert api.peak <= 2

    def test_progress_callback_counts_every_request(self) -> None:
        import asyncio

        api = FakeAsyncApi()
        seen: list[tuple[int, int]] = []
        requests = [Request(system="s", user="u", schema=VLM_SCHEMA) for _ in range(4)]
        asyncio.run(
            async_client(api).complete_json_many(
                requests, on_done=lambda done, total: seen.append((done, total))
            )
        )
        assert [d for d, _ in seen] == [1, 2, 3, 4]
        assert {t for _, t in seen} == {4}

    def test_a_failing_request_keeps_its_slot(self) -> None:
        """실패해도 길이가 줄면 호출부가 결과를 입력에 되짚을 수 없다."""
        import asyncio

        class Flaky:
            n = 0

            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs: Any) -> Any:
                        Flaky.n += 1
                        if Flaky.n == 2:
                            raise RuntimeError("한 개만 실패")
                        return fake_response('{"findings":[]}')().chat.completions.create()

        requests = [
            Request(system="s", user="u", schema=VLM_SCHEMA, tag=i) for i in range(3)
        ]
        out = asyncio.run(
            async_client(Flaky, max_retries=0, concurrency=1).complete_json_many(requests)
        )
        assert len(out) == 3
        assert [meta["tag"] for _, meta in out] == [0, 1, 2]
        assert sum(1 for _, meta in out if meta.get("error")) == 1
