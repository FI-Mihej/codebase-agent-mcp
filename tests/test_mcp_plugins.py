from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from codebase_agent.config import AgentConfig, LibraryConfig, OpenAICompatibleConfig, PluginConfig
from codebase_agent.mcp_stdio_plugin import StdioMCPPlugin, StdioMCPPluginConfig
from codebase_agent.plugin_factory import (
    TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME,
    build_plugin_bundle,
    build_static_mcp_plugin_config,
    build_text_file_read_and_refactor_mcp_config,
)
from codebase_agent.types import ClientRequestType, PluginABC, ToolResult


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: Any = None,
    ) -> CallToolResult:
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "read_timeout_seconds": read_timeout_seconds,
            }
        )
        return CallToolResult(
            content=[TextContent(type="text", text="tool response")],
            structuredContent={"value": 42},
            isError=False,
        )


class FakeErrorSession(FakeSession):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: Any = None,
    ) -> CallToolResult:
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "read_timeout_seconds": read_timeout_seconds,
            }
        )
        return CallToolResult(
            content=[TextContent(type="text", text="TypeError: '<=' not supported")],
            isError=True,
        )


class DummyPlugin(PluginABC):
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

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

    async def aexecute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return self.execute(library_name, client_request_type, tool_name, arguments)

    def executor(self) -> Any:
        return self.execute

    def all_tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        return []


def _agent_config(tmp_path: Path) -> AgentConfig:
    allowed_root = tmp_path / "allowed"
    denied_root = tmp_path / "denied"
    allowed_root.mkdir()
    denied_root.mkdir()
    return AgentConfig(
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            built_in_plugins=[
                {
                    "name": TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME,
                    "configuration": {
                        "command": "text-file-server",
                        "args": ["stale-config-value"],
                    },
                }
            ],
            mcp_plugins=[
                {
                    "name": "external_mcp",
                    "configuration": {"command": "external-server", "args": ["--flag"]},
                }
            ],
        ),
        libraries=[
            LibraryConfig(name="allowed-lib", allowed=True, path=allowed_root),
            LibraryConfig(name="denied-lib", allowed=False, path=denied_root),
        ],
    )


def test_stdio_mcp_plugin_exposes_allowed_openai_tool_definitions() -> None:
    plugin = StdioMCPPlugin(
        plugin_config=PluginConfig(name="external", denied_tools=["danger_tool"]),
        stdio_config=StdioMCPPluginConfig(command="fake-server"),
        session=FakeSession(),
        tools=[
            Tool(
                name="safe_tool",
                description="Read safe data.",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            Tool(name="danger_tool", inputSchema={"type": "object", "properties": {}}),
        ],
    )

    assert plugin.all_tools_names() == ["danger_tool", "safe_tool"]
    assert plugin.allowed_tool_names() == ["safe_tool"]
    assert plugin.allowed_tool_definitions() == [
        {
            "type": "function",
            "function": {
                "name": "safe_tool",
                "description": "Read safe data.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


def test_stdio_mcp_plugin_executes_async_tool_call() -> None:
    session = FakeSession()
    plugin = StdioMCPPlugin(
        plugin_config=PluginConfig(name="external"),
        stdio_config=StdioMCPPluginConfig(command="fake-server", read_timeout_seconds=2),
        session=session,
        tools=[Tool(name="safe_tool", inputSchema={"type": "object", "properties": {}})],
    )

    result = asyncio.run(
        plugin.aexecute(
            "example-lib",
            ClientRequestType.analysis,
            "safe_tool",
            {"path": "README.md"},
        )
    )

    assert result == {
        "ok": True,
        "result": {
            "structured_content": {"value": 42},
            "content": [{"type": "text", "text": "tool response"}],
        },
    }
    assert session.calls[0]["name"] == "safe_tool"
    assert session.calls[0]["arguments"] == {"path": "README.md"}


def test_stdio_mcp_plugin_returns_structured_error_for_mcp_tool_error() -> None:
    session = FakeErrorSession()
    plugin = StdioMCPPlugin(
        plugin_config=PluginConfig(name="external"),
        stdio_config=StdioMCPPluginConfig(command="fake-server", read_timeout_seconds=2),
        session=session,
        tools=[Tool(name="safe_tool", inputSchema={"type": "object", "properties": {}})],
    )

    result = asyncio.run(
        plugin.aexecute(
            "example-lib",
            ClientRequestType.analysis,
            "safe_tool",
            {"place": {"stop": 720}},
        )
    )

    assert result == {
        "ok": False,
        "error": {
            "code": "mcp_tool_error",
            "message": "TypeError: '<=' not supported",
        },
    }
    assert session.calls[0]["arguments"] == {"place": {"stop": 720}}


def test_stdio_mcp_plugin_resolves_file_path_arguments_under_library_root(tmp_path: Path) -> None:
    library_root = tmp_path / "allowed"
    nested_file = library_root / "cengal" / "module.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("hello", encoding="utf-8")
    session = FakeSession()
    plugin = StdioMCPPlugin(
        plugin_config=PluginConfig(name=TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME),
        stdio_config=StdioMCPPluginConfig(
            command="fake-server",
            library_roots={"allowed-lib": library_root},
            resolve_file_path_arguments=True,
        ),
        session=session,
        tools=[Tool(name="text_file__file_content_length", inputSchema={"type": "object", "properties": {}})],
    )

    result = asyncio.run(
        plugin.aexecute(
            "allowed-lib",
            ClientRequestType.analysis,
            "text_file__file_content_length",
            {"source_file_path": "cengal/module.py"},
        )
    )

    assert result["ok"] is True
    assert session.calls[0]["arguments"] == {"source_file_path": str(nested_file.resolve())}


def test_stdio_mcp_plugin_rejects_file_path_arguments_outside_library_root(tmp_path: Path) -> None:
    library_root = tmp_path / "allowed"
    library_root.mkdir()
    session = FakeSession()
    plugin = StdioMCPPlugin(
        plugin_config=PluginConfig(name=TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME),
        stdio_config=StdioMCPPluginConfig(
            command="fake-server",
            library_roots={"allowed-lib": library_root},
            resolve_file_path_arguments=True,
        ),
        session=session,
        tools=[Tool(name="text_file__file_content_length", inputSchema={"type": "object", "properties": {}})],
    )

    result = asyncio.run(
        plugin.aexecute(
            "allowed-lib",
            ClientRequestType.analysis,
            "text_file__file_content_length",
            {"source_file_path": r"C:\outside\secret.py"},
        )
    )

    assert result["ok"] is False
    assert "outside configured library root" in result["error"]["message"]
    assert session.calls == []


def test_stdio_mcp_plugin_denied_tool_returns_error_without_calling_session() -> None:
    session = FakeSession()
    plugin = StdioMCPPlugin(
        plugin_config=PluginConfig(name="external", denied_tools=["safe_tool"]),
        stdio_config=StdioMCPPluginConfig(command="fake-server"),
        session=session,
        tools=[Tool(name="safe_tool", inputSchema={"type": "object", "properties": {}})],
    )

    result = asyncio.run(
        plugin.aexecute("example-lib", ClientRequestType.analysis, "safe_tool", {})
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "tool_disabled"
    assert session.calls == []


def test_static_mcp_plugin_config_reads_stdio_configuration() -> None:
    config = build_static_mcp_plugin_config(
        PluginConfig(
            name="external_mcp",
            configuration={
                "command": "external-server",
                "args": ["--root", "C:/repo"],
                "cwd": "C:/work",
                "read_timeout_seconds": 5,
            },
        )
    )

    assert config.command == "external-server"
    assert config.args == ["--root", "C:/repo"]
    assert config.cwd == "C:/work"
    assert config.read_timeout_seconds == 5
    assert config.encoding_error_handler == "replace"


def test_text_file_mcp_config_generates_args_from_allowed_library_paths(tmp_path: Path) -> None:
    config = _agent_config(tmp_path)
    plugin_config = config.openai_compatible.allowed_built_in_plugin_config(
        TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME
    )

    stdio_config = build_text_file_read_and_refactor_mcp_config(config, plugin_config)

    assert stdio_config.command == "text-file-server"
    assert stdio_config.args == [str(tmp_path / "allowed")]
    assert stdio_config.library_roots == {"allowed-lib": tmp_path / "allowed"}
    assert stdio_config.resolve_file_path_arguments is True
    assert stdio_config.encoding_error_handler == "replace"


def test_build_plugin_bundle_adds_built_in_and_external_mcp_plugins(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    created: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> DummyPlugin:
        created.append(kwargs)
        return DummyPlugin(kwargs["plugin_config"].name)

    monkeypatch.setattr(StdioMCPPlugin, "create", fake_create)

    async def run() -> None:
        async with AsyncExitStack() as exit_stack:
            bundle = await build_plugin_bundle(_agent_config(tmp_path), exit_stack)
            assert TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME in bundle.plugins
            assert "external_mcp" in bundle.plugins

    asyncio.run(run())

    assert [call["plugin_config"].name for call in created] == [
        TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME,
        "external_mcp",
    ]
    assert created[0]["stdio_config"].command == "text-file-server"
    assert created[0]["stdio_config"].args == [str(tmp_path / "allowed")]
    assert created[0]["stdio_config"].library_roots == {"allowed-lib": tmp_path / "allowed"}
    assert created[0]["stdio_config"].resolve_file_path_arguments is True
    assert created[1]["stdio_config"].command == "external-server"
    assert created[1]["stdio_config"].args == ["--flag"]
