#!/usr/bin/env python
# coding=utf-8

# Copyright © 2026 ButenkoMS. All rights reserved. Contacts: <gtalk@butenkoms.space>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Plugin adapter for external stdio MCP servers."""


from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel, Field

from codebase_agent.config import PluginConfig
from codebase_agent.io_debug import IODebugLogger, logged_io_streams
from codebase_agent.types import ClientRequestType, PluginABC, ToolResult


class StdioMCPPluginConfig(BaseModel):
    """Configuration used to launch a stdio MCP server process."""

    command: str = Field(..., min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "replace"
    read_timeout_seconds: float | None = None
    library_roots: dict[str, Path] = Field(default_factory=dict)
    resolve_file_path_arguments: bool = False


class StdioMCPPlugin(PluginABC):
    """Expose a stdio MCP server as a local OpenAI-compatible tool plugin."""

    def __init__(
        self,
        *,
        plugin_config: PluginConfig,
        stdio_config: StdioMCPPluginConfig,
        session: ClientSession,
        tools: list[Tool],
    ) -> None:
        self._plugin_config = plugin_config
        self._stdio_config = stdio_config
        self._session = session
        self._tools = tools

    @classmethod
    async def create(
        cls,
        *,
        plugin_config: PluginConfig,
        stdio_config: StdioMCPPluginConfig,
        exit_stack: AsyncExitStack,
        io_debug_logger: IODebugLogger | None = None,
    ) -> "StdioMCPPlugin":
        server = StdioServerParameters(
            command=stdio_config.command,
            args=stdio_config.args,
            env=stdio_config.env,
            cwd=stdio_config.cwd,
            encoding=stdio_config.encoding,
            encoding_error_handler=stdio_config.encoding_error_handler,
        )
        read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server))
        if io_debug_logger is not None:
            read_stream, write_stream = await exit_stack.enter_async_context(
                logged_io_streams(
                    read_stream,
                    write_stream,
                    logger=io_debug_logger,
                    path=f"server-plugin:{plugin_config.name}",
                    inbound_direction="plugin_to_server",
                    outbound_direction="server_to_plugin",
                )
            )
        session = await exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=_read_timeout(stdio_config),
            )
        )
        await session.initialize()
        tools_result = await session.list_tools()
        return cls(
            plugin_config=plugin_config,
            stdio_config=stdio_config,
            session=session,
            tools=list(tools_result.tools),
        )

    def name(self) -> str:
        return self._plugin_config.name

    def all_tools_names(self) -> list[str]:
        return sorted(tool.name for tool in self._tools)

    def allowed_tool_names(self) -> list[str]:
        if not self._plugin_config.allowed:
            return []

        denied_tools = set(self._plugin_config.denied_tools)
        return sorted(tool_name for tool_name in self.all_tools_names() if tool_name not in denied_tools)

    def worker_by_tool_name(self, tool_name: str) -> Any:
        if tool_name in self.allowed_tool_names():
            return self.aexecute
        return None

    def execute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        raise NotImplementedError("StdioMCPPlugin only supports async execution.")

    async def aexecute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if not self._plugin_config.allowed:
            return _tool_error("tool_not_enabled", f"Tool is not enabled: {tool_name}")
        if tool_name in self._plugin_config.denied_tools:
            return _tool_error("tool_disabled", f"Tool is disabled by configuration: {tool_name}")
        if tool_name not in self.all_tools_names():
            return _tool_error("tool_not_enabled", f"Tool is not provided by plugin {self.name()}: {tool_name}")

        try:
            result = await self._session.call_tool(
                tool_name,
                arguments=self._normalize_arguments(library_name, arguments),
                read_timeout_seconds=_read_timeout(self._stdio_config),
            )
        except Exception as exc:
            return _tool_error("mcp_tool_error", str(exc))

        if result.isError:
            return _tool_error("mcp_tool_error", _call_tool_error_message(result))

        return {"ok": True, "result": _call_tool_result_payload(result)}

    def _normalize_arguments(self, library_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._stdio_config.resolve_file_path_arguments:
            return arguments

        normalized = dict(arguments)
        for name, value in arguments.items():
            if name.endswith("_file_path") and isinstance(value, str) and value:
                normalized[name] = str(self._resolve_under_library_root(library_name, value))
        return normalized

    def _resolve_under_library_root(self, library_name: str, requested_path: str) -> Path:
        library_root = self._stdio_config.library_roots.get(library_name)
        if library_root is None:
            raise ValueError(f"Unknown library for plugin {self.name()}: {library_name}")

        root = library_root.resolve()
        requested = Path(requested_path)
        windows_requested = PureWindowsPath(requested_path)
        if not requested.is_absolute() and (windows_requested.drive or windows_requested.root):
            raise ValueError(f"Rejected path outside configured library root: {requested_path}")

        candidate = requested if requested.is_absolute() else root / requested
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Rejected path outside configured library root: {requested_path}") from exc
        return resolved

    def executor(self) -> Any:
        return self.aexecute

    def all_tool_definitions(self) -> list[dict[str, Any]]:
        return [_tool_to_openai_tool_definition(tool) for tool in self._tools]

    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        allowed_names = set(self.allowed_tool_names())
        return [tool for tool in self.all_tool_definitions() if tool["function"]["name"] in allowed_names]


def _read_timeout(config: StdioMCPPluginConfig) -> timedelta | None:
    if config.read_timeout_seconds is None:
        return None
    return timedelta(seconds=config.read_timeout_seconds)


def _tool_to_openai_tool_definition(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def _call_tool_result_payload(result: CallToolResult) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if result.structuredContent is not None:
        payload["structured_content"] = result.structuredContent
    if result.content:
        payload["content"] = [_mcp_model_to_json(content) for content in result.content]
    return payload


def _call_tool_error_message(result: CallToolResult) -> str:
    text_parts: list[str] = []
    for content in result.content:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text:
            text_parts.append(text)
    if text_parts:
        return "\n".join(text_parts)
    return str(_call_tool_result_payload(result))


def _mcp_model_to_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return value


def _tool_error(code: str, message: str) -> ToolResult:
    return {"ok": False, "error": {"code": code, "message": message}}
