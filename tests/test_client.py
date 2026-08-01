"""LLM 클라이언트 테스트.

네트워크를 타지 않고, **실패했을 때 페이지 전체를 잃지 않는지**를 본다.
규칙 레이어 결과는 LLM 이 죽어도 살아남아야 한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_pipeline.llm.client import (
    LlmClient,
    LlmConfig,
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
