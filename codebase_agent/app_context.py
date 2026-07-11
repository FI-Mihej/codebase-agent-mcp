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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any

from codebase_agent.config import AgentConfig

if TYPE_CHECKING:
    from codebase_agent.jobs import CodebaseAnalysisJobManager
    from codebase_agent.built_in_plugins.qdrant_client import QdrantClientABC
    from codebase_agent.built_in_plugins.local_fs_tools import LocalFilesystemTools
    from codebase_agent.types import (
        PluginABC,
    )
    from codebase_agent.mcp_stdio_plugin import StdioMCPPlugin

__all__ = [
    "AppContext",
]


@dataclass
class AppContext:
    """Application context with typed dependencies."""

    config: AgentConfig
    analysis_jobs: CodebaseAnalysisJobManager
    plugins: Dict[str, PluginABC]
    qdrant_client: QdrantClientABC
    local_fs_tools: LocalFilesystemTools
    text_file_tools: StdioMCPPlugin

    def allowed_tool_names(self) -> list[str]:
        """Return the names of all allowed tools."""
        allowed_tools_set: set[str] = set()
        for plugin in self.plugins.values():
            allowed_tools_set.update(plugin.allowed_tool_names())
        
        return sorted(list(allowed_tools_set))
    
    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        """Return all allowed tool definitions."""
        allowed_tool_definitions: list[dict[str, Any]] = list()
        for plugin in self.plugins.values():
            allowed_tool_definitions.extend(plugin.allowed_tool_definitions())

        return allowed_tool_definitions
    
    def plugin_by_tool_name(self, tool_name: str) -> PluginABC | None:
        """Return the plugin that provides the given tool name, or None if not found."""
        for plugin in self.plugins.values():
            if tool_name in plugin.allowed_tool_names():
                return plugin
        
        return None
