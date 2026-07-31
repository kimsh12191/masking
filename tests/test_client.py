"""LLM 클라이언트 테스트.

네트워크를 타지 않고, **실패했을 때 페이지 전체를 잃지 않는지**를 본다.
규칙 레이어 결과는 LLM 이 죽어도 살아남아야 한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline.llm.client import LlmClient, LlmConfig
from pii_pipeline.schema import PASS2_SCHEMA


class TestCompleteJsonFailures:
    def test_image_encoding_failure_returns_error_not_raises(self) -> None:
        """인코딩 실패가 예외로 새면 이미 계산된 규칙 레이어 결과까지 날아간다."""
        client = LlmClient(LlmConfig())
        payload, meta = client.complete_json(
            system="s", user="u", schema=PASS2_SCHEMA, image=object()
        )
        assert payload == {}
        assert "이미지 인코딩 실패" in meta["error"]

    def test_encoding_failure_does_not_call_the_api(self, monkeypatch) -> None:
        client = LlmClient(LlmConfig())
        called: list[int] = []
        monkeypatch.setattr(
            type(client), "client", property(lambda self: called.append(1))
        )
        client.complete_json(system="s", user="u", schema=PASS2_SCHEMA, image=object())
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
        payload, meta = client.complete_json(system="s", user="u", schema=PASS2_SCHEMA)
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
        payload, meta = client.complete_json(system="s", user="u", schema=PASS2_SCHEMA)
        assert payload == {}
        assert "JSON 파싱 실패" in meta["error"]
        assert len(attempts) == 3  # 최초 1회 + 재시도 2회

    def test_truncated_output_is_treated_as_failure(self, monkeypatch) -> None:
        """max_tokens 에서 잘리면 JSON 이 불완전하므로 성공으로 보지 않는다."""

        class Msg:
            content = '{"missed":'

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
        payload, meta = client.complete_json(system="s", user="u", schema=PASS2_SCHEMA)
        assert payload == {}
        assert "max_tokens" in meta["error"]


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
        build().complete_json(system="s", user="u", schema=PASS2_SCHEMA)
        assert seen["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_temperature_is_zero(self, captured) -> None:
        build, seen = captured
        build().complete_json(system="s", user="u", schema=PASS2_SCHEMA)
        assert seen["temperature"] == 0.0

    def test_schema_is_enforced(self, captured) -> None:
        build, seen = captured
        build().complete_json(system="s", user="u", schema=PASS2_SCHEMA)
        assert seen["extra_body"]["guided_json"] == PASS2_SCHEMA
        assert seen["extra_body"]["guided_decoding_backend"] == "xgrammar"

    def test_text_only_request_has_string_content(self, captured) -> None:
        build, seen = captured
        build().complete_json(system="s", user="u", schema=PASS2_SCHEMA)
        assert seen["messages"][1]["content"] == "u"
