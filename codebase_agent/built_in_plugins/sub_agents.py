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


from __future__ import annotations

from collections.abc import Awaitable, Callable
from fnmatch import fnmatchcase
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Set, Dict, List, Optional, TYPE_CHECKING
from cengal.introspection.inspect import gsodi

from codebase_agent.types import (
    SubagentsToolsContext,
    ToolExecutionError,
    ToolResult,
    ClientRequestType,
    PluginABC,
    UnknownLibraryError,
)
from codebase_agent.config import AgentConfig, load_config, validate_library_path, LibraryConfig, PluginConfig
from pydantic import BaseModel, Field
import logging
from cengal.introspection.inspect import get_exception, exception_to_printable_text
from codebase_agent.prompts import (
    build_prompt_for__subagent__generic,
)
if TYPE_CHECKING:
    from codebase_agent.app_context import AppContext


logger = logging.getLogger(__name__)


CompletionCreate = Callable[..., Awaitable[Any]]
SUBAGENTS_TOOL_NAMES_LIST = [
    "subagent__generic",
    "subagent__map_reduce__single_file",
]
SUBAGENTS_TOOL_NAMES = frozenset(SUBAGENTS_TOOL_NAMES_LIST)


class BuiltInSubagentsConfig(BaseModel):
    """Configuration for the built-in subagents plugin."""

    concurrency_limit: int = Field(default=0, ge=0)


def _require_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f"Missing or invalid string argument: {name}")
    
    return value


class SubagentsTools(PluginABC):

    def __init__(
        self, 
        config: AgentConfig, 
    ) -> None:
        self._config: AgentConfig = config
        self._app_context: Optional[AppContext] = None

    def set_app_context(self, app_context):
        self._app_context = app_context

    def execute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        raise NotImplementedError("SubagentsTools does not support sync execution.")

    async def aexecute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        denied_tools = self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name())
        try:
            if not self._config.openai_compatible.has_allowed_built_in_plugin(self.name()):
                raise ToolExecutionError(
                    f"Tool is not enabled: {tool_name}",
                    code="tool_not_enabled",
                )
            if tool_name in denied_tools:
                raise ToolExecutionError(
                    f"Tool is disabled by configuration: {tool_name}",
                    code="tool_disabled",
                )
            
            if tool_name == "subagent__generic":
                result = await self._subagent__generic(
                    library_name, 
                    client_request_type, 
                    _require_string(arguments, "query"),
                )
            elif tool_name == "subagent__map_reduce__single_file":
                result = await self._subagent__map_reduce__single_file(
                    library_name, 
                    client_request_type, 
                    _require_string(arguments, "file_path"),
                    _require_string(arguments, "map_query"),
                )
            else:
                raise ToolExecutionError(f"Tool is not enabled: {tool_name}", code="tool_not_enabled")
        except ToolExecutionError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        except Exception as exc:
            return {"ok": False, "error": {"code": "unexpected_tool_error", "message": str(exc)}}

        return {"ok": True, "result": result}
    
    def executor(self) -> Any:
        return self.aexecute

    async def _subagent__generic(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
    ) -> dict[str, Any]:
        built_in_subagents_config: BuiltInSubagentsConfig = BuiltInSubagentsConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=SubagentsToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            concurrency_limit=built_in_subagents_config.concurrency_limit,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        from codebase_agent.jobs import gen_job_id
        from codebase_agent.openai_compatible_client import OpenAICompatibleClient

        if self._app_context is None:
            raise RuntimeError("AppContext was not set.")

        library: LibraryConfig | None = self._config.get_allowed_library(library_name)
        if library is None:
            raise UnknownLibraryError(f"Unknown library name: {library_name}")

        openai_compatible_client = OpenAICompatibleClient(self._config.openai_compatible)
        system_prompt: str = build_prompt_for__subagent__generic(
            app_context=self._app_context,
            library=library,
            openai_compatible=self._config.openai_compatible,
            user_query=query,
        )
        result: str | None = None
        exception_str: str | None = None
        try:
            result: str = await openai_compatible_client.consult(
                app_context=self._app_context,
                job_id=gen_job_id(),
                library=library,
                client_request_type=ClientRequestType.subagent,
                system_prompt=system_prompt,
                query=query,
            )
        except Exception as exc:
            logger.exception("<<subagent__generic>>.exception")
            exception_str = exception_to_printable_text(get_exception())

        if exception_str is None:
            return {
                "result": result
            }
        else:
            return {
                "exception": exception_str
            }

    async def _subagent__map_reduce__single_file(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        file_path: str,
        map_query: str,
    ) -> dict[str, Any]:
        return {
            "result": ""
        }
        built_in_subagents_config: BuiltInSubagentsConfig = BuiltInSubagentsConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=SubagentsToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            concurrency_limit=built_in_subagents_config.concurrency_limit,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        from codebase_agent.jobs import gen_job_id
        from codebase_agent.openai_compatible_client import OpenAICompatibleClient

        if self._app_context is None:
            raise RuntimeError("AppContext was not set.")

        library: LibraryConfig | None = self._config.get_allowed_library(library_name)
        if library is None:
            raise UnknownLibraryError(f"Unknown library name: {library_name}")

        openai_compatible_client = OpenAICompatibleClient(self._config.openai_compatible)
        system_prompt: str = build_prompt_for__subagent__generic(
            app_context=self._app_context,
            library=library,
            openai_compatible=self._config.openai_compatible,
            user_query=query,
        ),
        result: str | None = None
        exception_str: str | None = None
        try:
            result: str = await openai_compatible_client.consult(
                app_context=self._app_context,
                job_id=gen_job_id(),
                library=library,
                client_request_type=ClientRequestType.subagent,
                system_prompt=system_prompt,
                query=query,
            )
        except Exception as exc:
            logger.exception("<<subagent__generic>>.exception")
            exception_str = exception_to_printable_text(get_exception())

        if exception_str is None:
            return {
                "result": result
            }
        else:
            return {
                "exception": exception_str
            }

    def name(self) -> str:
        return "built_in_subagents"

    def all_tools_names(self) -> list[str]:
        return SUBAGENTS_TOOL_NAMES_LIST

    def allowed_tool_names(self) -> list[str]:
        if not self._config.openai_compatible.has_allowed_built_in_plugin(self.name()):
            return []

        denied_tools = self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name())
        denied_tools_set: Set[str] = denied_tools
        all_tools_set: Set[str] = SUBAGENTS_TOOL_NAMES
        return sorted(list(all_tools_set - denied_tools_set))
    
    def worker_by_tool_name(self, tool_name: str) -> Any:
        mapping = {
            "subagent__generic": self._subagent__generic,
            "subagent__map_reduce__single_file": self._subagent__map_reduce__single_file,
        }
        return mapping.get(tool_name)
    
    def all_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI compatible tool definitions for sandboxed filesystem access."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "subagent__generic",
                    "description": "Launches a full-fledged subagent with the same library and the same tools as you; delegate tasks to the subagent so as not to waste your own context window.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query for subagent.",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "subagent__map_reduce__single_file",
                    "description": "It runs a MapReduce process on chunks of a specified file; the file is split into chunks, the size of which is automatically calculated based on the available context window size. The Map phase is performed with overlapping chunks - for example, if the chunk size is 100 characters, the following ranges will be processed: [slice(0, 100), slice(50, 150), slice(100, 200), slice(150, 250), ...]. First, each chunk is analyzed individually (in parallel) using both specific map instructions (including chunk content) and your specified map_query. During this process, each sub-agent automatically receives instructions to output the line ranges to which its findings belong. Afterward, a Reduce phase is performed based on those line ranges; where necessary, additional inter-chunk processing is conducted by a sub-agent who receives both specific reduction instructions and the original map_query to maintain context of the task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to file.",
                            },
                            "map_query": {
                                "type": "string",
                                "description": "Query for an each map-phase subagent.",
                            },
                        },
                        "required": ["file_path", "map_query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
    
    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        allowed_names = set(self.allowed_tool_names())
        return [tool for tool in self.all_tool_definitions() if tool["function"]["name"] in allowed_names]

    def _resolve_under_root(self, library_root: Path, requested_path: str | Path) -> Path:
        requested_text = str(requested_path or ".")
        requested = Path(requested_text)
        windows_requested = PureWindowsPath(requested_text)
        if not requested.is_absolute() and (windows_requested.drive or windows_requested.root):
            raise ToolExecutionError(
                f"Rejected path outside configured library root: {requested_path}"
            )
        candidate = requested if requested.is_absolute() else library_root / requested
        resolved = candidate.resolve()
        try:
            resolved.relative_to(library_root)
        except ValueError as exc:
            raise ToolExecutionError(
                f"Rejected path outside configured library root: {requested_path}"
            ) from exc
        return resolved

    def _normalize_relative_path(self, requested_path: str | Path) -> Path:
        requested_text = str(requested_path or ".")
        normalized_text = requested_text.replace("\\", "/")
        requested = Path(normalized_text)
        windows_requested = PureWindowsPath(requested_text)
        if requested.is_absolute() or windows_requested.drive or windows_requested.root:
            raise ToolExecutionError(
                f"Rejected path outside configured library root: {requested_path}"
            )
        return requested

    def _safe_glob_pattern(self, pattern: str) -> str:
        normalized = pattern.replace("\\", "/")
        requested = Path(normalized)
        windows_requested = PureWindowsPath(pattern)
        if requested.is_absolute() or windows_requested.drive or windows_requested.root:
            raise ToolExecutionError(
                f"Rejected pattern outside configured library root: {pattern}"
            )
        if any(part == ".." for part in normalized.split("/")):
            raise ToolExecutionError(
                f"Rejected pattern outside configured library root: {pattern}"
            )
        return normalized

    def _expand_braces(self, pattern: str) -> list[str]:
        start = pattern.find("{")
        if start == -1:
            return [pattern]
        end = pattern.find("}", start + 1)
        if end == -1:
            return [pattern]

        prefix = pattern[:start]
        suffix = pattern[end + 1:]
        expanded: list[str] = []
        for option in pattern[start + 1:end].split(","):
            for tail in self._expand_braces(suffix):
                expanded.append(f"{prefix}{option}{tail}")
                if len(expanded) >= 128:
                    return expanded
        return expanded

    def _file_entry(self, library_root: Path, path: Path) -> dict[str, Any]:
        return {
            "path": self._relative(library_root, path),
            "type": "directory" if path.is_dir() else "file",
            "bytes": None if path.is_dir() else path.stat().st_size,
        }

    def _relative(self, library_root: Path, path: Path) -> str:
        try:
            return path.resolve().relative_to(library_root).as_posix() or "."
        except ValueError:
            return str(path)
