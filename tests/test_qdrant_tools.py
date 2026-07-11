from __future__ import annotations

from typing import Any, Optional

from qdrant_client.conversions.common_types import UpdateResult as QdrantUpdateResult

from codebase_agent.built_in_plugins.qdrant_client import QdrantCachedQuery, QdrantClientABC
from codebase_agent.config import PluginConfig
from codebase_agent.types import ClientRequestType, ToolResult


class InMemoryQdrantClient(QdrantClientABC):
    def __init__(self, *, denied_tools: list[str] | None = None) -> None:
        self.plugin_config = PluginConfig(name="qdrant_test", denied_tools=denied_tools or [])

    def name(self) -> str:
        return self.plugin_config.name

    def all_tools_names(self) -> list[str]:
        return ["vectordb__find_cached_queries", "vectordb__get_cached_response"]

    def allowed_tool_names(self) -> list[str]:
        denied_tools = set(self.plugin_config.denied_tools)
        return sorted(set(self.all_tools_names()) - denied_tools)

    def worker_by_tool_name(self, tool_name: str) -> Any:
        return None

    def upsert(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        response: str,
        suppress_exceptions: bool = False,
    ) -> Optional[QdrantUpdateResult]:
        return None

    def find_cached_queries(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        top_k: int = 5,
    ) -> list[QdrantCachedQuery]:
        return [QdrantCachedQuery(query=f"cached: {query}", query_id=f"top-{top_k}")]

    def get_cached_response(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query_id: str,
    ) -> Optional[str]:
        return f"response: {query_id}"


def test_qdrant_all_tool_definitions_exposes_vectordb_tools() -> None:
    plugin = InMemoryQdrantClient()

    definitions = plugin.all_tool_definitions()

    assert [tool["function"]["name"] for tool in definitions] == [
        "vectordb__find_cached_queries",
        "vectordb__get_cached_response",
    ]
    assert definitions[0]["function"]["parameters"]["required"] == ["query"]
    assert definitions[1]["function"]["parameters"]["required"] == ["query_id"]


def test_qdrant_allowed_tool_definitions_filters_denied_tools() -> None:
    plugin = InMemoryQdrantClient(denied_tools=["vectordb__get_cached_response"])

    definitions = plugin.allowed_tool_definitions()

    assert [tool["function"]["name"] for tool in definitions] == ["vectordb__find_cached_queries"]


def test_qdrant_execute_dispatches_vectordb_tools() -> None:
    plugin = InMemoryQdrantClient()

    find_result: ToolResult = plugin.execute(
        "example-lib",
        ClientRequestType.analysis,
        "vectordb__find_cached_queries",
        {"query": "How?", "top_k": 3},
    )
    response_result: ToolResult = plugin.execute(
        "example-lib",
        ClientRequestType.analysis,
        "vectordb__get_cached_response",
        {"query_id": "abc"},
    )

    assert find_result == {
        "ok": True,
        "result": {"cached_queries": [{"query": "cached: How?", "query_id": "top-3"}]},
    }
    assert response_result == {"ok": True, "result": {"response": "response: abc"}}
