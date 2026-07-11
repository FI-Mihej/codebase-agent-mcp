from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import BadRequestError
from openai.types.chat import ChatCompletion

from codebase_agent.config import AgentConfig, OpenAICompatibleConfig, LibraryConfig
from codebase_agent.app_context import AppContext
from codebase_agent.openai_compatible_client import OpenAICompatibleClient
from codebase_agent.built_in_plugins.local_fs_tools import LocalFilesystemTools
from codebase_agent.jobs import CodebaseAnalysisJobManager
from codebase_agent.server import (
    codebase_start_job_analysis_job_with_config,
    codebase_start_job_analysis_with_config,
    codebase_start_job_related_files_search_job_with_config,
    codebase_start_job_related_files_search_with_config,
    codebase_list_libraries_from_config,
)
from codebase_agent.built_in_plugins.qdrant_client import QdrantClientABC
from codebase_agent.types import (
    ClientRequestType,
    MalformedOpenAICompatibleResponse,
    DirectErrorResponseForClientLLM,
    QueryGranularityError,
    ToolResult,
    UnknownLibraryError,
)


def _config(
    tmp_path: Path,
    *,
    max_tool_rounds: int = 12,
    tool_backend: str = "openai_tools",
    max_tokens: int = 4096,
    context_compression_threshold: float = 0.8,
) -> AgentConfig:
    library_root = tmp_path / "lib"
    library_root.mkdir(exist_ok=True)
    return AgentConfig(
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            tool_backend=tool_backend,
            built_in_plugins=["built_in_fs"],
            max_tokens=max_tokens,
            max_tool_rounds=max_tool_rounds,
            context_compression_threshold=context_compression_threshold,
        ),
        libraries=[
            LibraryConfig(
                name="example-lib",
                allowed=True,
                path=library_root,
                instructions="Use tests as evidence.",
            )
        ],
    )


def _config_with_denied_tools(tmp_path: Path, denied_tools: list[str]) -> AgentConfig:
    library_root = tmp_path / "lib"
    library_root.mkdir(exist_ok=True)
    return AgentConfig(
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            tool_backend="openai_tools",
            built_in_plugins=[
                {
                    "name": "built_in_fs",
                    "allowed": True,
                    "denied_tools": denied_tools,
                }
            ],
        ),
        libraries=[
            LibraryConfig(
                name="example-lib",
                allowed=True,
                path=library_root,
                instructions="Use tests as evidence.",
            )
        ],
    )


def _config_without_built_in_plugins(tmp_path: Path) -> AgentConfig:
    library_root = tmp_path / "lib"
    library_root.mkdir(exist_ok=True)
    return AgentConfig(
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            tool_backend="openai_tools",
            built_in_plugins=[],
        ),
        libraries=[
            LibraryConfig(
                name="example-lib",
                allowed=True,
                path=library_root,
                instructions="Use tests as evidence.",
            )
        ],
    )


class FakeQdrantClient(QdrantClientABC):
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    def name(self) -> str:
        return "qdrant_fake"

    def all_tools_names(self) -> list[str]:
        return []

    def allowed_tool_names(self) -> list[str]:
        return []

    def worker_by_tool_name(self, tool_name: str) -> Any:
        return None

    def execute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return {"ok": False, "error": {"code": "tool_not_enabled", "message": tool_name}}

    def all_tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def upsert(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        response: str,
        suppress_exceptions: bool = False,
    ) -> None:
        self.upserts.append(
            {
                "library_name": library_name,
                "client_request_type": client_request_type,
                "query": query,
                "response": response,
                "suppress_exceptions": suppress_exceptions,
            }
        )


def _app_context(config: AgentConfig) -> AppContext:
    qdrant_client = FakeQdrantClient()
    local_fs_tools = LocalFilesystemTools(config=config)
    return AppContext(
        config=config,
        analysis_jobs=CodebaseAnalysisJobManager(),
        plugins={qdrant_client.name(): qdrant_client, local_fs_tools.name(): local_fs_tools},
        qdrant_client=qdrant_client,
        local_fs_tools=local_fs_tools,
        text_file_tools=None,
    )


class FailingFilesystemTools(LocalFilesystemTools):
    def execute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return {
            "ok": False,
            "error": {
                "code": "mcp_tool_error",
                "message": "TypeError: '<=' not supported",
            },
        }


def _app_context_with_failing_fs(config: AgentConfig) -> AppContext:
    qdrant_client = FakeQdrantClient()
    local_fs_tools = FailingFilesystemTools(config=config)
    return AppContext(
        config=config,
        analysis_jobs=CodebaseAnalysisJobManager(),
        plugins={qdrant_client.name(): qdrant_client, local_fs_tools.name(): local_fs_tools},
        qdrant_client=qdrant_client,
        local_fs_tools=local_fs_tools,
        text_file_tools=None,
    )


def _bad_request_error(message: str, body: dict[str, Any]) -> BadRequestError:
    request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=body)


def _chat_completion(payload: dict[str, Any]) -> ChatCompletion:
    normalized = {
        "id": "chatcmpl-test",
        "created": 0,
        "model": "local-model",
        "object": "chat.completion",
        **payload,
    }
    normalized["choices"] = [
        {
            "index": index,
            "finish_reason": (
                "tool_calls"
                if choice.get("message", {}).get("tool_calls")
                else "stop"
            ),
            **choice,
        }
        for index, choice in enumerate(normalized.get("choices", []))
    ]
    if "usage" in normalized:
        usage = normalized["usage"]
        usage.setdefault("completion_tokens", 0)
        usage.setdefault("total_tokens", usage.get("prompt_tokens", 0) + usage["completion_tokens"])
    return ChatCompletion.model_validate(normalized)


def _mock_openai_client(config: AgentConfig, completion_create: Any) -> OpenAICompatibleClient:
    async def wrapped_create(**kwargs: Any) -> Any:
        result = await completion_create(**copy.deepcopy(kwargs))
        if isinstance(result, dict) and "choices" in result:
            return _chat_completion(result)
        return result

    return OpenAICompatibleClient(config.openai_compatible, completion_create=wrapped_create)


def test_codebase_list_libraries_from_config(tmp_path: Path) -> None:
    assert codebase_list_libraries_from_config(_config(tmp_path)) == {"libraries": ["example-lib"]}


def test_unknown_library_handling(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path)
        with pytest.raises(UnknownLibraryError):
            await codebase_start_job_analysis_with_config(
                app_context=_app_context(config),
                config=config,
                library_name="missing",
                query="How?",
                client=None,
            )

    asyncio.run(run())


def test_codebase_start_job_related_files_search_unknown_library_handling(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path)
        with pytest.raises(UnknownLibraryError):
            await codebase_start_job_related_files_search_with_config(
                app_context=_app_context(config),
                config=config,
                library_name="missing",
                query="Where is auth?",
                client=None,
            )

    asyncio.run(run())


def test_filesystem_sandbox_rejects_outside_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = tmp_path / "lib"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    tools = LocalFilesystemTools(config=config)

    result = tools.execute("example-lib", ClientRequestType.analysis, "fs__read_text_file", {"path": str(outside)})

    assert result["ok"] is False
    assert "outside configured library root" in result["error"]["message"]


@pytest.mark.parametrize("requested_path", [r"C:\outside\secret.txt", r"C:secret.txt", r"\outside\secret.txt"])
def test_filesystem_sandbox_rejects_win32_qualified_paths(tmp_path: Path, requested_path: str) -> None:
    config = _config(tmp_path)
    tools = LocalFilesystemTools(config=config)

    result = tools.execute("example-lib", ClientRequestType.analysis, "fs__read_text_file", {"path": requested_path})

    assert result["ok"] is False
    assert "outside configured library root" in result["error"]["message"]


def test_filesystem_sandbox_allows_absolute_path_inside_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = tmp_path / "lib"
    readme = root / "README.md"
    readme.write_text("inside", encoding="utf-8")
    tools = LocalFilesystemTools(config=config)

    result = tools.execute("example-lib", ClientRequestType.analysis, "fs__read_text_file", {"path": str(readme)})

    assert result["ok"] is True
    assert result["result"]["path"] == "README.md"
    assert result["result"]["content"] == "inside"


def test_filesystem_tool_rejects_denied_tool(tmp_path: Path) -> None:
    config = _config_with_denied_tools(tmp_path, ["fs__read_text_file"])
    root = tmp_path / "lib"
    (root / "README.md").write_text("secret", encoding="utf-8")
    tools = LocalFilesystemTools(config=config)

    result = tools.execute("example-lib", ClientRequestType.analysis, "fs__read_text_file", {"path": "README.md"})

    assert result["ok"] is False
    assert result["error"]["code"] == "tool_disabled"
    assert "disabled by configuration" in result["error"]["message"]


def test_mocked_openai_compatible_response_with_no_tool_calls(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "Final answer"}}]}

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert result == "Final answer"
    assert len(calls) == 2
    assert calls[0]["messages"][-1] == {"role": "user", "content": "How?"}
    assert "tools" not in calls[0]
    assert "tools" not in calls[1]


def test_openai_tools_without_allowed_builtin_fs_does_not_advertise_filesystem_tools(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "Final answer"}}]}

    config = _config_without_built_in_plugins(tmp_path)
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert result == "Final answer"
    assert "tools" not in calls[0]


def test_openai_compatible_context_overflow_without_tools_returns_direct_message(tmp_path: Path) -> None:
    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        raise _bad_request_error(
            "This model's maximum context length is 4096 tokens.",
            {"error": {"code": "context_length_exceeded"}},
        )

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert "Retry with a narrower request" in result


def test_openai_compatible_detects_lm_studio_context_size_overflow_without_tools(tmp_path: Path) -> None:
    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        message = "request (1101409 tokens) exceeds the available context size (65536 tokens), try increasing it"
        raise _bad_request_error(
            message,
            {"error": {"message": message}},
        )

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert "Retry with a narrower request" in result


@pytest.mark.xfail(
    raises=AttributeError,
    reason="Current overflow handling reads serialized assistant tool-call messages as SDK objects.",
)
def test_openai_compatible_context_overflow_after_tool_round_raises_domain_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (config.allowed_libraries()[0].path / "README.md").write_text("hello", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__read_text_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        raise _bad_request_error(
            "Prompt is too long for this context window.",
            {"error": {"message": "Prompt is too long for this context window."}},
        )

    client = _mock_openai_client(config, fake_create)

    with pytest.raises(DirectErrorResponseForClientLLM) as exc_info:
        asyncio.run(
            codebase_start_job_analysis_with_config(
                app_context=_app_context(config),
                config=config,
                library_name="example-lib",
                query="How?",
                client=client,
            )
        )

    assert len(calls) == 3
    assert "not possible to compact the chat history" in str(exc_info.value)


def test_codebase_start_job_related_files_search_parses_mocked_openai_compatible_json(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"files":[{"path":"src/auth.py","score":0.94,'
                            '"reason":"Contains refresh_token references.",'
                            '"evidence":["refresh_token"]}],'
                            '"notes":"Search covered source files."}'
                        ),
                    }
                }
            ]
        }

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_related_files_search_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="Where is token refresh?",
            client=client,
        )
    )

    assert result["files"] == [
        {
            "path": "src/auth.py",
            "score": 0.94,
            "reason": "Contains refresh_token references.",
            "evidence": ["refresh_token"],
        }
    ]
    assert result["notes"] == "Search covered source files."
    assert "Where is token refresh?" in calls[0]["messages"][0]["content"]


def test_codebase_start_job_related_files_search_job_serializes_structured_result(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {"choices": [{"message": {"role": "assistant", "content": '{"valid":true,"reason":"One entity."}'}}]}

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"files":[{"path":"src/auth.py","score":0.94,'
                            '"reason":"Contains refresh_token references.",'
                            '"evidence":["refresh_token"]}],'
                            '"notes":"Search covered source files."}'
                        ),
                    }
                }
            ]
        }

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_related_files_search_job_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="Where is token refresh?",
            client=client,
        )
    )

    assert len(calls) == 3
    assert "tools" not in calls[0]
    assert json.loads(result) == {
        "files": [
            {
                "path": "src/auth.py",
                "score": 0.94,
                "reason": "Contains refresh_token references.",
                "evidence": ["refresh_token"],
            }
        ],
        "notes": "Search covered source files.",
    }


def test_codebase_start_job_analysis_job_validates_query_before_analysis(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {"choices": [{"message": {"role": "assistant", "content": '{"valid":true,"reason":"One action."}'}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "Final answer"}}]}

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_job_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="Analyze token refresh.",
            client=client,
        )
    )

    assert result == "Final answer"
    assert len(calls) == 3
    assert "strict request gatekeeper" in calls[0]["messages"][0]["content"]
    assert "One topic or one context per request" in calls[0]["messages"][0]["content"]
    assert "tools" not in calls[0]


def test_codebase_start_job_analysis_job_rejects_broad_query_before_analysis(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"valid":false,"reason":"The request asks for X and Y."}',
                    }
                }
            ]
        }

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    with pytest.raises(QueryGranularityError) as exc_info:
        asyncio.run(
            codebase_start_job_analysis_job_with_config(
                app_context=_app_context(config),
                config=config,
                library_name="example-lib",
                query="Analyze X and Y.",
                client=client,
            )
        )

    assert len(calls) == 1
    assert exc_info.value.code == "query_granularity_violation"
    assert "The request asks for X and Y." in str(exc_info.value)
    assert "Client LLM instruction: One topic or one context per request" in str(exc_info.value)


def test_final_response_persists_compressed_no_tool_result(tmp_path: Path) -> None:
    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<analysis>hidden reasoning</analysis>\nFinal answer\n```tool_call\n{}\n```",
                    }
                }
            ]
        }

    config = _config(tmp_path, tool_backend="none")
    client = _mock_openai_client(config, fake_create)

    app_context = _app_context(config)

    result = asyncio.run(
        client.consult(
            app_context=app_context,
            library=config.allowed_libraries()[0],
            client_request_type=ClientRequestType.analysis,
            system_prompt="system",
            query="How?",
        )
    )

    raw_content = "<analysis>hidden reasoning</analysis>\nFinal answer\n```tool_call\n{}\n```"
    assert result == raw_content
    assert app_context.qdrant_client.upserts == [
        {
            "library_name": "example-lib",
            "client_request_type": ClientRequestType.analysis,
            "query": "How?",
            "response": raw_content,
            "suppress_exceptions": True,
        }
    ]


def test_context_compression_preserves_pending_tool_call(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        max_tokens=100,
        context_compression_threshold=0.5,
    )
    library_root = config.allowed_libraries()[0].path
    (library_root / "README.md").write_text("# Example\nUse Widget.\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    original_prompt = "How? Include this fence: ````"

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "usage": {"prompt_tokens": 75},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__read_text_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        if "tools" not in kwargs:
            compression_prompt = kwargs["messages"][-1]["content"]
            assert "Only summarize the conversation history" in compression_prompt
            assert "Compress the conversation history while preserving all information necessary" in compression_prompt
            assert "Original prompt:\n`````\nHow? Include this fence: ````\n`````" in compression_prompt
            assert "Conversation history, excluding the system prompt:\n`````" in compression_prompt
            if len(calls) > 2:
                return {"choices": [{"message": {"role": "assistant", "content": "Use Widget."}}]}
            return {"choices": [{"message": {"role": "assistant", "content": "Need README evidence."}}]}

        assert len(calls) == 3
        resumed_messages = kwargs["messages"]
        assert resumed_messages[0] == {"role": "system", "content": "system"}
        assert resumed_messages[1] == {"role": "user", "content": original_prompt}
        assert resumed_messages[2] == {"role": "assistant", "content": "Need README evidence."}
        assert resumed_messages[3] == {
            "role": "developer",
            "content": "Conversation history was compacted. Continue your work.",
        }
        assert resumed_messages[4]["role"] == "assistant"
        assert resumed_messages[4]["tool_calls"][0]["id"] == "call_1"
        assert resumed_messages[5]["role"] == "tool"
        assert "Use Widget." in resumed_messages[5]["content"]
        return {"choices": [{"message": {"role": "assistant", "content": "Use Widget."}}]}

    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        client.consult(
            app_context=_app_context(config),
            library=config.allowed_libraries()[0],
            client_request_type=ClientRequestType.analysis,
            system_prompt="system",
            query=original_prompt,
        )
    )

    assert result == "Use Widget."
    assert len(calls) == 4


def test_mocked_openai_compatible_response_with_one_fs__read_text_file_tool_call(tmp_path: Path) -> None:
    config = _config(tmp_path)
    library_root = config.allowed_libraries()[0].path
    (library_root / "README.md").write_text("# Example\nUse Widget.\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__read_text_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Use Widget."}}]}

    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert result == "Use Widget."
    assert len(calls) == 3
    tool_message = calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "Use Widget." in tool_message["content"]


def test_plugin_error_result_is_passed_to_model_as_tool_message(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__read_text_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if "tools" not in kwargs:
            return {"choices": [{"message": {"role": "assistant", "content": "Recovered with another approach."}}]}

        tool_message = kwargs["messages"][-1]
        tool_result = json.loads(tool_message["content"])
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_1"
        assert tool_result == {
            "ok": False,
            "error": {
                "code": "mcp_tool_error",
                "message": "TypeError: '<=' not supported",
            },
        }
        return {"choices": [{"message": {"role": "assistant", "content": "Recovered with another approach."}}]}

    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context_with_failing_fs(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert result == "Recovered with another approach."
    assert len(calls) == 3


def test_denied_tool_is_not_advertised_to_openai_compatible(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "Final answer"}}]}

    config = _config_with_denied_tools(tmp_path, ["fs__read_text_file"])
    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert result == "Final answer"
    advertised_names = {tool["function"]["name"] for tool in calls[0]["tools"]}
    assert advertised_names == {"fs__list_files", "fs__search_text_in_files"}


def test_denied_tool_call_returns_structured_tool_error(tmp_path: Path) -> None:
    config = _config_with_denied_tools(tmp_path, ["fs__read_text_file"])
    (config.allowed_libraries()[0].path / "README.md").write_text("hello", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__read_text_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I could not read the file because the tool is disabled.",
                    }
                }
            ]
        }

    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    tool_message = calls[1]["messages"][-1]
    tool_result = json.loads(tool_message["content"])
    assert result == "I could not read the file because the tool is disabled."
    assert tool_result["ok"] is False
    assert tool_result["error"]["code"] == "tool_disabled"


def test_denied_tool_call_with_malformed_arguments_returns_tool_error(tmp_path: Path) -> None:
    config = _config_with_denied_tools(tmp_path, ["fs__read_text_file"])
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs__read_text_file",
                                        "arguments": "{bad",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Recovered"}}]}

    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    tool_result = json.loads(calls[1]["messages"][-1]["content"])
    assert result == "Recovered"
    assert tool_result["error"]["code"] == "tool_disabled"


def test_malformed_tool_call_arguments(tmp_path: Path) -> None:
    config = _config(tmp_path)

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fs__read_text_file",
                                    "arguments": "{bad",
                                },
                            }
                        ],
                    }
                }
            ]
        }

    client = _mock_openai_client(config, fake_create)

    with pytest.raises(MalformedOpenAICompatibleResponse):
        asyncio.run(
            codebase_start_job_analysis_with_config(
                app_context=_app_context(config),
                config=config,
                library_name="example-lib",
                query="How?",
                client=client,
            )
        )


def test_bounded_termination_after_max_tool_rounds(tmp_path: Path) -> None:
    config = _config(tmp_path, max_tool_rounds=1)
    (config.allowed_libraries()[0].path / "README.md").write_text("hello", encoding="utf-8")

    async def fake_create(**kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Partial answer",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fs__read_text_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    client = _mock_openai_client(config, fake_create)

    result = asyncio.run(
        codebase_start_job_analysis_with_config(
            app_context=_app_context(config),
            config=config,
            library_name="example-lib",
            query="How?",
            client=client,
        )
    )

    assert "Partial answer" in result
