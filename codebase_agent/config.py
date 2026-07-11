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


"""Configuration loading and validation."""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from codebase_agent.types import ConfigurationError, InvalidLibraryPathError, ToolBackend
from cengal.file_system.app_fs_structure.app_dir_path import AppDirectoryType, AppDirPath, app_dir_path
from cengal.file_system.file_manager import path_relative_to_current_src, file_exists
from cengal.text_processing.open_text_file import OpenTextFile
from shutil import copyfile


CONFIG_TEMPLATE_NAME = "codebase_agent.config.example.json"
CONFIG_NAME = "codebase_agent.config.json"
DB_NAME = "codebase_agent.jobs.sqlite3"


def get_app_name() -> str:
    return "codebase_agent"


def get_config_path(config_path: Path | str | None = None) -> Path:
    config_dir_path: Path = Path(app_dir_path(AppDirectoryType.local_config, get_app_name(), with_structure=True, ensure_dir=True))
    need_default_config_path = False
    if config_path is None:
        need_default_config_path = True
    elif isinstance(config_path, (str, Path)):
        config_path = Path(config_path)
        if config_path.is_absolute():
            return (config_path / CONFIG_NAME).resolve()
        else:
            return (config_dir_path / config_path / CONFIG_NAME).resolve()
    else:
        need_default_config_path = True

    if need_default_config_path:
        return config_dir_path / CONFIG_NAME


def get_db_path(db_path: Path | str | None = None) -> Path:
    db_dir_path: Path = Path(app_dir_path(AppDirectoryType.local_data, get_app_name(), with_structure=True, ensure_dir=True))
    need_default_db_path = False
    if db_path is None:
        need_default_db_path = True
    elif isinstance(db_path, (str, Path)):
        db_path = Path(db_path)
        if db_path.is_absolute():
            return db_path.resolve()
        else:
            return (db_dir_path / db_path).resolve()
    else:
        need_default_db_path = True

    if need_default_db_path:
        return db_dir_path / DB_NAME


def get_local_data_path() -> Path:
    return Path(app_dir_path(AppDirectoryType.local_data, get_app_name(), with_structure=True, ensure_dir=True))


def get_local_log_path() -> Path:
    return Path(app_dir_path(AppDirectoryType.local_log, get_app_name(), with_structure=True, ensure_dir=True))


def get_project_root_path() -> Path:
    return Path(path_relative_to_current_src(".."))


def get_app_root_path() -> Path:
    return Path(path_relative_to_current_src())


def get_config_template_path() -> Path:
    return get_app_root_path() / "data" / CONFIG_TEMPLATE_NAME


def ensure_config_exists(config_path: Path | None = None) -> Path:
    config_path = get_config_path(config_path)
    if not file_exists(str(config_path)):
        copyfile(get_config_template_path(), config_path)
    
    return config_path


class PluginConfig(BaseModel):
    """An OpenAI compatible local-model plugin/capability and its tool restrictions."""

    name: str = Field(..., min_length=1)
    allowed: bool = True
    denied_tools: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("plugin name must not be empty")
        
        return normalized

    @field_validator("denied_tools")
    @classmethod
    def normalize_denied_tools(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for tool_name in value:
            stripped = tool_name.strip()
            if not stripped:
                raise ValueError("denied tool names must not be empty")
            if stripped not in normalized:
                normalized.append(stripped)
        
        return normalized


class OpenAICompatibleConfig(BaseModel):
    """OpenAI compatible connection and request settings."""

    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    temperature: float = 0.2
    reasoning_allowed: bool = False
    reasoning_effort: str = ""
    max_tokens: int = Field(default=4096, gt=0)
    context_compression_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=300, gt=0)
    built_in_plugins: list[PluginConfig] = Field(default_factory=list)
    mcp_plugins: list[PluginConfig] = Field(default_factory=list)
    tool_backend: ToolBackend = "openai_tools"
    max_tool_rounds: int = Field(default=12, ge=-1)
    max_tool_response_bytes: int = Field(default=5 * 1024, ge=0)

    @field_validator("built_in_plugins", mode="before")
    @classmethod
    def normalize_built_in_plugins(cls, value: Any) -> Any:
        """Accept legacy string entries and the structured denied_tools form."""

        if value is None:
            return []
        
        if not isinstance(value, list):
            return value
        
        normalized: list[Any] = []
        for entry in value:
            if isinstance(entry, str):
                normalized.append({"name": entry})
            else:
                normalized.append(entry)
        
        return normalized

    @field_validator("mcp_plugins", mode="before")
    @classmethod
    def normalize_mcp_plugins(cls, value: Any) -> Any:
        """Accept legacy string entries and the structured denied_tools form."""

        if value is None:
            return []
        
        if not isinstance(value, list):
            return value
        
        normalized: list[Any] = []
        for entry in value:
            if isinstance(entry, str):
                normalized.append({"name": entry})
            else:
                normalized.append(entry)
        
        return normalized

    def normalized_base_url(self) -> str:
        """Return the OpenAI compatible `/v1` base URL."""

        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        
        return f"{normalized}/v1"
    
    def all_plugins(self) -> list[PluginConfig]:
        """Return all configured local-model plugins."""

        return self.built_in_plugins + self.mcp_plugins
    
    def allowed_plugins(self) -> list[PluginConfig]:
        """Return the configured local-model plugins that are allowed."""

        return [plugin for plugin in self.all_plugins() if plugin.allowed]
    
    def allowed_plugin_names(self) -> set[str]:
        """Return the configured local-model plugin names that are allowed."""

        return {plugin.name for plugin in self.allowed_plugins()}

    def all_plugin_names(self) -> set[str]:
        """Return all local-model plugin names."""

        all_plugins = self.built_in_plugins + self.mcp_plugins
        return {plugin.name for plugin in all_plugins}
    
    def all_plugin_config(self, name: str) -> PluginConfig | None:
        """Return the configured local-model plugin configuration for a given name."""

        for plugin in self.all_plugins():
            if plugin.name == name:
                return plugin
        
        return None
    
    def all_built_in_plugin_config(self, name: str) -> PluginConfig | None:
        """Return the configured built-in plugin configuration for a given name."""

        for plugin in self.built_in_plugins:
            if plugin.name == name:
                return plugin
        
        return None
    
    def all_mcp_plugin_config(self, name: str) -> PluginConfig | None:
        """Return the configured MCP plugin configuration for a given name."""

        for plugin in self.mcp_plugins:
            if plugin.name == name:
                return plugin
        
        return None
    
    def allowed_plugin_config(self, name: str) -> PluginConfig | None:
        """Return the configured local-model plugin configuration for a given name."""

        for plugin in self.allowed_plugins():
            if plugin.name == name:
                return plugin
        
        return None
    
    def allowed_built_in_plugin_config(self, name: str) -> PluginConfig | None:
        """Return the configured built-in plugin configuration for a given name."""

        for plugin in self.built_in_plugins:
            if plugin.allowed and plugin.name == name:
                return plugin
        
        return None
    
    def allowed_mcp_plugin_config(self, name: str) -> PluginConfig | None:
        """Return the configured MCP plugin configuration for a given name."""

        for plugin in self.mcp_plugins:
            if plugin.allowed and plugin.name == name:
                return plugin
        
        return None
    
    def has_allowed_plugin(self, name: str) -> bool:
        """Return whether a local-model plugin/capability is allowed."""

        return self.allowed_plugin_config(name) is not None

    def has_allowed_built_in_plugin(self, name: str) -> bool:
        """Return whether a built-in plugin/capability is allowed."""

        return self.allowed_built_in_plugin_config(name) is not None

    def has_allowed_mcp_plugin(self, name: str) -> bool:
        """Return whether an MCP plugin/capability is allowed."""

        return self.allowed_mcp_plugin_config(name) is not None

    def has_plugin(self, name: str) -> bool:
        """Return whether a local-model plugin/capability is allowed."""

        return name in self.allowed_plugin_names()

    def denied_tools_for_allowed_plugin(self, name: str) -> frozenset[str]:
        """Return disabled tool names for a configured plugin."""

        denied: set[str] = set()
        for plugin in self.allowed_plugins():
            if plugin.name == name:
                denied.update(plugin.denied_tools)
        
        return frozenset(denied)
    
    def denied_tools_for_allowed_built_in_plugin(self, name: str) -> frozenset[str]:
        """Return disabled tool names for a configured built-in plugin."""

        denied: set[str] = set()
        for plugin in self.built_in_plugins:
            if plugin.allowed and plugin.name == name:
                denied.update(plugin.denied_tools)
        
        return frozenset(denied)
    
    def denied_tools_for_allowed_mcp_plugin(self, name: str) -> frozenset[str]:
        """Return disabled tool names for a configured MCP plugin."""

        denied: set[str] = set()
        for plugin in self.mcp_plugins:
            if plugin.allowed and plugin.name == name:
                denied.update(plugin.denied_tools)
        
        return frozenset(denied)
    
    def all_denied_tools(self) -> frozenset[str]:
        """Return all disabled tool names for all configured plugins."""

        denied: set[str] = set()
        for plugin in self.allowed_plugins():
            denied.update(plugin.denied_tools)
        
        return frozenset(denied)


class LibraryConfig(BaseModel):
    """A locally available library or codebase that can be consulted."""

    name: str = Field(..., min_length=1)
    allowed: bool = True
    path: Path
    instructions: str = ""

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("library path must be absolute")
        return value


class JobConfig(BaseModel):
    """Background analysis job management settings."""

    storage_backend: Literal["sqlite"] = "sqlite"
    sqlite_path: Path = Path("codebase_agent.jobs.sqlite3")
    max_concurrent_jobs: int = Field(default=2, gt=0)
    job_ttl_seconds: int = Field(default=3600, gt=0)
    max_completed_jobs: int = Field(default=100, ge=0)
    status_update_interval_seconds: float = Field(default=1, gt=0)
    result_wait_timeout_seconds: float = Field(default=50, ge=0)
    result_poll_interval_seconds: float = Field(default=1, gt=0)


class IODebugConfig(BaseModel):
    """Optional MCP stdio IO debug logging settings."""

    enabled: bool = False
    log_path: Path = Path("codebase_agent.io.jsonl")
    client_server: bool = True
    server_plugins: bool = True


class AgentConfig(BaseModel):
    """Root project configuration."""

    openai_compatible: OpenAICompatibleConfig
    libraries: list[LibraryConfig] = Field(default_factory=list)
    jobs: JobConfig = Field(default_factory=JobConfig)
    io_debug: IODebugConfig = Field(default_factory=IODebugConfig)

    @model_validator(mode="after")
    def library_names_must_be_unique(self) -> "AgentConfig":
        names = [library.name for library in self.libraries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate library names: {', '.join(duplicates)}")
        
        return self
    
    def allowed_libraries(self) -> list[LibraryConfig]:
        """Return the configured libraries that are allowed."""

        return [library for library in self.libraries if library.allowed]

    def get_library(self, name: str) -> LibraryConfig | None:
        return next((library for library in self.libraries if library.name == name), None)
    
    def get_allowed_library(self, name: str) -> LibraryConfig | None:
        return next((library for library in self.libraries if library.name == name and library.allowed), None)


def load_config(config_path: Path | str | None = None, *, validate_paths: bool = True) -> AgentConfig:
    """Load the root JSON configuration file."""

    path = ensure_config_exists(config_path)
    if not path.exists():
        raise ConfigurationError(f"Missing configuration file: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")

    raw: dict[str, Any]
    with OpenTextFile(path, "rb") as text_file_info:
        text_file_info.text.value,
        try:
            raw = json.loads(text_file_info.text.value)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON configuration in {path}: {exc}") from exc
        except OSError as exc:
            raise ConfigurationError(f"Could not read configuration file {path}: {exc}") from exc

    try:
        config = AgentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {path}: {exc}") from exc

    config = resolve_config_paths(config)
    if validate_paths:
        validate_library_paths(config)
    
    return config


def resolve_config_paths(config: AgentConfig) -> AgentConfig:
    """Resolve relative config-managed paths against the configuration file directory."""

    sqlite_path = config.jobs.sqlite_path
    if not sqlite_path.is_absolute():
        sqlite_path = (get_local_data_path() / sqlite_path).resolve()

    io_debug_log_path = config.io_debug.log_path
    if not io_debug_log_path.is_absolute():
        io_debug_log_path = (get_local_log_path() / io_debug_log_path).resolve()

    return config.model_copy(
        update={
            "jobs": config.jobs.model_copy(update={"sqlite_path": sqlite_path}),
            "io_debug": config.io_debug.model_copy(update={"log_path": io_debug_log_path}),
        }
    )


def validate_library_paths(config: AgentConfig) -> None:
    """Ensure configured library paths exist and are directories."""

    for library in config.allowed_libraries():
        validate_library_path(library.path, library.name)


def validate_library_path(path: Path, library_name: str) -> Path:
    """Resolve and validate a single configured library root."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise InvalidLibraryPathError(
            f"Library '{library_name}' path is not accessible: {path}"
        ) from exc

    if not resolved.is_dir():
        raise InvalidLibraryPathError(
            f"Library '{library_name}' path is not a directory: {resolved}"
        )
    return resolved
