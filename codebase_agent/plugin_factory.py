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


"""Build runtime plugins from application configuration."""


from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Dict

from codebase_agent.built_in_plugins.local_fs_tools import LocalFilesystemTools
from codebase_agent.built_in_plugins.qdrant_client import QdrantClientABC, qdrant_client_factory
from codebase_agent.config import AgentConfig, PluginConfig
from codebase_agent.io_debug import IODebugLogger
from codebase_agent.mcp_stdio_plugin import StdioMCPPlugin, StdioMCPPluginConfig
from codebase_agent.types import ConfigurationError, PluginABC


TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME = "text_file_read_and_refactor_mcp"


@dataclass(frozen=True)
class PluginBundle:
    """Runtime plugins and strongly typed built-in dependencies."""

    plugins: Dict[str, PluginABC]
    qdrant_client: QdrantClientABC
    local_fs_tools: LocalFilesystemTools
    text_file_tools: StdioMCPPlugin


async def build_plugin_bundle(config: AgentConfig, exit_stack: AsyncExitStack) -> PluginBundle:
    """Create all configured runtime plugins."""

    qdrant_client = qdrant_client_factory(config)
    local_fs_tools = LocalFilesystemTools(config)
    io_debug_logger = (
        IODebugLogger(config.io_debug.log_path)
        if config.io_debug.enabled and config.io_debug.server_plugins
        else None
    )
    plugins: Dict[str, PluginABC] = {
        qdrant_client.name(): qdrant_client,
        local_fs_tools.name(): local_fs_tools,
    }

    text_file_plugin_config = config.openai_compatible.allowed_built_in_plugin_config(
        TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME
    )
    if text_file_plugin_config is not None:
        text_file_plugin = await StdioMCPPlugin.create(
            plugin_config=text_file_plugin_config,
            stdio_config=build_text_file_read_and_refactor_mcp_config(config, text_file_plugin_config),
            exit_stack=exit_stack,
            io_debug_logger=io_debug_logger,
        )
        plugins[text_file_plugin.name()] = text_file_plugin

    for plugin_config in config.openai_compatible.mcp_plugins:
        if not plugin_config.allowed:
            continue
        stdio_config = build_static_mcp_plugin_config(plugin_config)
        mcp_plugin = await StdioMCPPlugin.create(
            plugin_config=plugin_config,
            stdio_config=stdio_config,
            exit_stack=exit_stack,
            io_debug_logger=io_debug_logger,
        )
        plugins[mcp_plugin.name()] = mcp_plugin

    return PluginBundle(
        plugins=plugins,
        qdrant_client=qdrant_client,
        local_fs_tools=local_fs_tools,
        text_file_tools=text_file_plugin,
    )


def build_static_mcp_plugin_config(plugin_config: PluginConfig) -> StdioMCPPluginConfig:
    """Build stdio launch config for an externally configured MCP plugin."""

    try:
        return StdioMCPPluginConfig.model_validate(plugin_config.configuration)
    except Exception as exc:
        raise ConfigurationError(
            f"MCP plugin '{plugin_config.name}' must define valid stdio configuration"
        ) from exc


def build_text_file_read_and_refactor_mcp_config(
    config: AgentConfig,
    plugin_config: PluginConfig,
) -> StdioMCPPluginConfig:
    """Generate text-file MCP launch config from the allowed library roots."""

    configured_command = plugin_config.configuration.get("command")
    args = list()
    if configured_command:
        command = configured_command
    else:
        command = sys.executable
        args.extend(["-m", TEXT_FILE_READ_AND_REFACTOR_PLUGIN_NAME])
    
    allowed_libraries = config.allowed_libraries()
    args.extend([str(library.path) for library in allowed_libraries])
    static_options = {
        key: value
        for key, value in plugin_config.configuration.items()
        if key not in {"command", "args", "library_roots", "resolve_file_path_arguments"}
    }
    return StdioMCPPluginConfig(
        command=command,
        args=args,
        library_roots={library.name: library.path for library in allowed_libraries},
        resolve_file_path_arguments=True,
        **static_options,
    )
