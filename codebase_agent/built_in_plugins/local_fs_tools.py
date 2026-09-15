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

from codebase_agent.types import (
    LocalFSToolsContext,
    ToolExecutionError,
    ToolResult,
    ClientRequestType,
    PluginABC,
)
from codebase_agent.config import AgentConfig, load_config, validate_library_path, LibraryConfig, PluginConfig
from pydantic import BaseModel, Field
if TYPE_CHECKING:
    from codebase_agent.app_context import AppContext


CompletionCreate = Callable[..., Awaitable[Any]]
FILESYSTEM_TOOL_NAMES_LIST = [
    "fs__list_files",
    "fs__read_text_file",
    "fs__search_text_in_files",
    "fs__glob",
    "fs__grep",
    "fs__get_normalized_full_path",
]
FILESYSTEM_TOOL_NAMES = frozenset(FILESYSTEM_TOOL_NAMES_LIST)


class BuiltInFSConfig(BaseModel):
    """Configuration for the built-in filesystem plugin."""

    max_file_bytes: int = Field(default=1_048_576, gt=0)
    max_search_results: int = Field(default=50, gt=0)


def _require_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f"Missing or invalid string argument: {name}")
    
    return value


class LocalFilesystemTools(PluginABC):
    """Sandboxed filesystem tool executor for a single configured library."""

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
        """Execute an allowed filesystem tool and return a structured result."""

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
            if tool_name == "fs__list_files":
                result = self._fs__list_files(
                    library_name,
                    client_request_type,
                    arguments.get("path", "."),
                    recursive=bool(arguments.get("recursive", False)),
                )
            elif tool_name == "fs__read_text_file":
                result = self._fs__read_text_file(
                    library_name, 
                    client_request_type, 
                    _require_string(arguments, "path"),
                )
            elif tool_name == "fs__search_text_in_files":
                result = self._fs__search_text_in_files(
                    library_name,
                    client_request_type,
                    query=_require_string(arguments, "query"),
                    path=arguments.get("path", "."),
                    case_sensitive=bool(arguments.get("case_sensitive", False)),
                )
            elif tool_name == "fs__glob":
                result = self._fs__glob(
                    library_name,
                    client_request_type,
                    pattern=_require_string(arguments, "pattern"),
                    path=arguments.get("path", "."),
                )
            elif tool_name == "fs__grep":
                result = self._fs__grep(
                    library_name,
                    client_request_type,
                    pattern=_require_string(arguments, "pattern"),
                    path=arguments.get("path", "."),
                    include=arguments.get("include"),
                )
            elif tool_name == "fs__get_normalized_full_path":
                result = self._fs__get_normalized_full_path(
                    library_name,
                    client_request_type,
                    _require_string(arguments, "path"),
                )
            else:
                raise ToolExecutionError(f"Tool is not enabled: {tool_name}", code="tool_not_enabled")
        except ToolExecutionError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        except Exception as exc:
            return {"ok": False, "error": {"code": "unexpected_tool_error", "message": str(exc)}}

        return {"ok": True, "result": result}

    async def aexecute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        raise NotImplementedError("LocalFilesystemTools does not support async execution.")
    
    def executor(self) -> Any:
        return self.execute

    def _fs__list_files(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        requested_path: str | Path, 
        *, 
        recursive: bool
    ) -> dict[str, Any]:
        built_in_fs_config: BuiltInFSConfig = BuiltInFSConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=LocalFSToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            max_file_bytes=built_in_fs_config.max_file_bytes,
            max_search_results=built_in_fs_config.max_search_results,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        root = self._resolve_under_root(library_root, requested_path)
        if not root.exists():
            raise ToolExecutionError(f"Path does not exist: {requested_path}")

        if root.is_file():
            return {"root": self._relative(library_root, root), "entries": [self._file_entry(library_root, root)]}
        if not root.is_dir():
            raise ToolExecutionError(f"Path is not a directory or file: {requested_path}")

        iterator = root.rglob("*") if recursive else root.iterdir()
        entries = [self._file_entry(library_root, path) for path in sorted(iterator, key=lambda item: str(item))]
        return {"root": self._relative(library_root, root), "entries": entries}

    def _fs__read_text_file(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        requested_path: str | Path
    ) -> dict[str, Any]:
        built_in_fs_config: BuiltInFSConfig = BuiltInFSConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=LocalFSToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            max_file_bytes=built_in_fs_config.max_file_bytes,
            max_search_results=built_in_fs_config.max_search_results,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        path = self._resolve_under_root(library_root, requested_path)
        if not path.exists():
            raise ToolExecutionError(f"File does not exist: {requested_path}")
        if not path.is_file():
            raise ToolExecutionError(f"Path is not a file: {requested_path}")
        size = path.stat().st_size
        if size > context.max_file_bytes:
            raise ToolExecutionError(
                f"File exceeds max_file_bytes ({context.max_file_bytes}): {requested_path}"
            )

        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ToolExecutionError(f"File appears to be binary: {requested_path}")
        return {"path": self._relative(library_root, path), "bytes": size, "content": raw.decode("utf-8", errors="replace")}

    def _fs__search_text_in_files(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        *,
        query: str,
        path: str | Path,
        case_sensitive: bool,
    ) -> dict[str, Any]:
        built_in_fs_config: BuiltInFSConfig = BuiltInFSConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=LocalFSToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            max_file_bytes=built_in_fs_config.max_file_bytes,
            max_search_results=built_in_fs_config.max_search_results,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        if not query:
            raise ToolExecutionError("fs__search_text_in_files query must not be empty")

        root = self._resolve_under_root(library_root, path)
        if not root.exists():
            raise ToolExecutionError(f"Path does not exist: {path}")

        files = [root] if root.is_file() else [item for item in root.rglob("*") if item.is_file()]
        needle = query if case_sensitive else query.lower()
        results: list[dict[str, Any]] = []

        for file_path in files:
            if len(results) >= context.max_search_results:
                break
            if file_path.stat().st_size > context.max_file_bytes:
                continue
            raw = file_path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            haystack_lines = text.splitlines()
            for line_number, line in enumerate(haystack_lines, start=1):
                comparison = line if case_sensitive else line.lower()
                if needle in comparison:
                    results.append(
                        {
                            "path": self._relative(library_root, file_path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
                    if len(results) >= context.max_search_results:
                        break

        return {
            "query": query,
            "root": self._relative(library_root, root),
            "case_sensitive": case_sensitive,
            "max_results": context.max_search_results,
            "results": results,
            "truncated": len(results) >= context.max_search_results,
        }

    def _fs__glob(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        *,
        pattern: str,
        path: str | Path,
    ) -> dict[str, Any]:
        built_in_fs_config: BuiltInFSConfig = BuiltInFSConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=LocalFSToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            max_file_bytes=built_in_fs_config.max_file_bytes,
            max_search_results=built_in_fs_config.max_search_results,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        if not pattern:
            raise ToolExecutionError("fs__glob pattern must not be empty")

        root = self._resolve_under_root(library_root, path)
        if not root.exists():
            raise ToolExecutionError(f"Path does not exist: {path}")
        if not root.is_dir():
            raise ToolExecutionError(f"Path is not a directory: {path}")

        normalized_pattern = self._safe_glob_pattern(pattern)
        matches: list[Path] = []
        seen: set[Path] = set()
        truncated = False

        for expanded_pattern in self._expand_braces(normalized_pattern):
            for candidate in root.glob(expanded_pattern):
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(library_root)
                except ValueError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                matches.append(resolved)
                if len(matches) >= context.max_search_results:
                    truncated = True
                    break
            if truncated:
                break

        return {
            "pattern": pattern,
            "root": self._relative(library_root, root),
            "max_results": context.max_search_results,
            "results": sorted(self._relative(library_root, item) for item in matches),
            "truncated": truncated,
        }

    def _fs__grep(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        *,
        pattern: str,
        path: str | Path,
        include: Any,
    ) -> dict[str, Any]:
        built_in_fs_config: BuiltInFSConfig = BuiltInFSConfig.model_validate(self._config.openai_compatible.all_built_in_plugin_config(self.name()).configuration)
        context=LocalFSToolsContext(
            root=validate_library_path(self._config.get_library(library_name).path, library_name),
            max_file_bytes=built_in_fs_config.max_file_bytes,
            max_search_results=built_in_fs_config.max_search_results,
            denied_tools=self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name()),
        )
        library_root = context.root.resolve()

        if not pattern:
            raise ToolExecutionError("fs__grep pattern must not be empty")
        if include is not None and not isinstance(include, str):
            raise ToolExecutionError("Missing or invalid string argument: include")

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolExecutionError(f"Invalid regex pattern: {exc}. When using `fs__grep`, avoid complex regular expressions with nested quantifiers or multiple repeats (e.g., do not use `.*` inside other groups). If a simple pattern doesn't work, try searching for a specific unique string instead of a complex regex.") from exc

        root = self._resolve_under_root(library_root, path)
        if not root.exists():
            raise ToolExecutionError(f"Path does not exist: {path}")
        if not root.is_file() and not root.is_dir():
            raise ToolExecutionError(f"Path is not a directory or file: {path}")

        include_patterns = None
        if include:
            include_patterns = self._expand_braces(self._safe_glob_pattern(include))

        files = [root] if root.is_file() else [item for item in root.rglob("*") if item.is_file()]
        results: list[dict[str, Any]] = []

        for file_path in sorted(files, key=lambda item: str(item)):
            if len(results) >= context.max_search_results:
                break
            if include_patterns is not None:
                relative_to_root = self._relative(root, file_path)
                relative_to_library = self._relative(library_root, file_path)
                if not any(
                    fnmatchcase(relative_to_root, include_pattern)
                    or fnmatchcase(relative_to_library, include_pattern)
                    or fnmatchcase(file_path.name, include_pattern)
                    for include_pattern in include_patterns
                ):
                    continue
            if file_path.stat().st_size > context.max_file_bytes:
                continue
            raw = file_path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(
                        {
                            "path": self._relative(library_root, file_path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
                    if len(results) >= context.max_search_results:
                        break

        return {
            "pattern": pattern,
            "root": self._relative(library_root, root),
            "include": include,
            "max_results": context.max_search_results,
            "results": results,
            "truncated": len(results) >= context.max_search_results,
        }

    def _fs__get_normalized_full_path(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        requested_path: str | Path,
    ) -> dict[str, Any]:
        library_root = validate_library_path(self._config.get_library(library_name).path, library_name).resolve()
        normalized_requested_path = self._normalize_relative_path(requested_path)
        full_path = self._resolve_under_root(library_root, normalized_requested_path)
        return {"full_path": str(full_path)}

    def name(self) -> str:
        return "built_in_fs"

    def all_tools_names(self) -> list[str]:
        return FILESYSTEM_TOOL_NAMES_LIST

    def allowed_tool_names(self) -> list[str]:
        if not self._config.openai_compatible.has_allowed_built_in_plugin(self.name()):
            return []

        denied_tools = self._config.openai_compatible.denied_tools_for_allowed_built_in_plugin(self.name())
        denied_tools_set: Set[str] = denied_tools
        all_tools_set: Set[str] = FILESYSTEM_TOOL_NAMES
        return sorted(list(all_tools_set - denied_tools_set))
    
    def worker_by_tool_name(self, tool_name: str) -> Any:
        mapping = {
            "fs__list_files": self._fs__list_files,
            "fs__read_text_file": self._fs__read_text_file,
            "fs__search_text_in_files": self._fs__search_text_in_files,
            "fs__glob": self._fs__glob,
            "fs__grep": self._fs__grep,
            "fs__get_normalized_full_path": self._fs__get_normalized_full_path,
        }
        return mapping.get(tool_name)
    
    def all_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI compatible tool definitions for sandboxed filesystem access."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "fs__list_files",
                    "description": "List files and directories under the configured library root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path under the library root. Defaults to '.'.",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to recurse into subdirectories.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs__read_text_file",
                    "description": "Read one UTF-8-compatible text file under the configured library root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to a text file under the library root.",
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs__search_text_in_files",
                    "description": "Search text files under the configured library root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text to search for."},
                            "path": {
                                "type": "string",
                                "description": "Relative file or directory path to search. Defaults to '.'.",
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Use case-sensitive matching.",
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
                    "name": "fs__glob",
                    "description": "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n- Returns matching file paths\n- Use this tool when you need to find files by name patterns\n- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided.",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "The glob pattern to match files against",
                            },
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs__grep",
                    "description": "- Fast content search tool that works with any codebase size\n- Searches file contents using regular expressions\n- Supports full regex syntax (eg. \"log.*Error\", \"function\\s+\\w+\", etc.)\n- Filter files by pattern with the include parameter (eg. \"*.js\", \"*.{ts,tsx}\") and returns file paths and line numbers with matching lines\n- Use this tool when you need to find files containing specific patterns\n- If you need to identify/count the number of matches within files, use the Bash tool with `rg` (ripgrep) directly. Do NOT use `grep`.\n- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include": {
                                "type": "string",
                                "description": "File pattern to include in the search (e.g., \"*.js\", \"*.{ts,tsx}\")",
                            },
                            "path": {
                                "type": "string",
                                "description": "The directory to search in. Defaults to the current working directory.",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "The regex pattern to search for in file contents",
                            },
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs__get_normalized_full_path",
                    "description": "Normalize a relative file or directory path under the configured library root and return the full absolute path using current-system path separators.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative file or directory path under the library root. POSIX and Win32 separators are accepted.",
                            },
                        },
                        "required": ["path"],
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
