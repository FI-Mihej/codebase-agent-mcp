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


"""Shared domain types and exceptions."""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from abc import ABC
from enum import Enum
from typing import Any, Literal, TypedDict


ToolBackend = Literal["none", "openai_tools"]


class CodebaseAgentError(Exception):
    """Base error for expected CodebaseAgent failures."""

    code = "codebase_agent_error"


class ConfigurationError(CodebaseAgentError):
    """Raised when configuration cannot be loaded or validated."""

    code = "configuration_error"


class UnknownLibraryError(CodebaseAgentError):
    """Raised when a requested library is not configured."""

    code = "unknown_library"


class InvalidLibraryPathError(CodebaseAgentError):
    """Raised when a configured library path is unavailable."""

    code = "invalid_library_path"


class EmptyQueryError(CodebaseAgentError):
    """Raised when codebase_start_job_analysis receives an empty query."""

    code = "empty_query"


class QueryGranularityError(CodebaseAgentError):
    """Raised when a codebase request asks for more than one entity/action."""

    code = "query_granularity_violation"


class ConcurrentJobLimitError(CodebaseAgentError):
    """Raised when the configured concurrent background job limit is reached."""

    code = "concurrent_job_limit_reached"


class InvalidJobIdError(CodebaseAgentError):
    """Raised when a job lookup receives a malformed job id."""

    code = "invalid_job_id"


class OpenAICompatibleConnectionError(CodebaseAgentError):
    """Raised when OpenAI compatible cannot be reached."""

    code = "openai_compatible_connection_failure"


class OpenAICompatibleTimeoutError(CodebaseAgentError):
    """Raised when an OpenAI compatible request times out."""

    code = "openai_compatible_timeout"


class OpenAICompatibleContextOverflowError(CodebaseAgentError):
    """Raised when an OpenAI compatible request exceeds the model context window."""

    code = "openai_compatible_context_overflow"


class OpenAICompatibleContentFilterError(CodebaseAgentError):
    """Raised when an OpenAI compatible request is blocked by the content filter."""

    code = "openai_compatible_content_filter_error"


class MissingModelError(CodebaseAgentError):
    """Raised when OpenAI compatible reports the configured model is unavailable."""

    code = "missing_or_unavailable_model"


class MalformedOpenAICompatibleResponse(CodebaseAgentError):
    """Raised when OpenAI compatible returns an unusable chat-completions response."""

    code = "malformed_openai_compatible_response"


class ToolExecutionError(CodebaseAgentError):
    """Raised internally by sandboxed filesystem tools."""

    code = "tool_execution_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class DirectErrorResponseForClientLLM(CodebaseAgentError):
    """Raised internally by sandboxed filesystem tools."""

    code = "direct_error_response_for_client_llm"

    def __init__(self, message: str) -> None:
        super().__init__(message)
    
    @property
    def message(self) -> str:
        return self.args[0] if self.args else ""


@dataclass(frozen=True)
class ToolContext:
    """Runtime limits for local tools exposed to OpenAI compatible."""

    root: Path
    max_file_bytes: int
    max_search_results: int
    denied_tools: frozenset[str] = frozenset()


class ToolResult(TypedDict, total=False):
    ok: bool
    result: Any
    error: dict[str, str]


class ClientRequestType(Enum):
    find_files = "find_files"
    analysis = "analysis"


class PluginABC(ABC):
    """Abstract base class for plugins."""

    def name(self) -> str:
        ...

    def all_tools_names(self) -> list[str]:
        ...

    def allowed_tool_names(self) -> list[str]:
        ...
    
    def worker_by_tool_name(self, tool_name: str) -> Any:
        ...

    def execute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        ...

    async def aexecute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        ...
    
    def executor(self) -> Any:
        ...
    
    def all_tool_definitions(self) -> list[dict[str, Any]]:
        ...
    
    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        ...
