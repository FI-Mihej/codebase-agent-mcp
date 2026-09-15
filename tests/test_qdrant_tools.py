from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from qdrant_client import models
from qdrant_client.conversions.common_types import UpdateResult as QdrantUpdateResult

from codebase_agent.built_in_plugins.qdrant_client import (
    QDRANT_COLLECTION_SCHEMA_OWNER,
    QDRANT_COLLECTION_SCHEMA_VERSION,
    QDRANT_DEFAULT_MODEL_NAME,
    QdrantCachedQuery,
    QdrantClientABC,
    QdrantClientFastembed,
)
from codebase_agent.config import (
    AgentConfig,
    LibraryConfig,
    OpenAICompatibleConfig,
    PluginConfig,
)
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


class RecordingQdrantClient:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.existing_collections = {"find_files__example-lib"}
        self.collection_vectors: dict[str, Any] = {
            "find_files__example-lib": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        }
        self.collection_metadata: dict[str, dict[str, Any]] = {
            "find_files__example-lib": {},
        }
        self.created_collections: list[dict[str, Any]] = []
        self.updated_collections: list[dict[str, Any]] = []
        self.deleted_collections: list[str] = []
        self.upsert_calls: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.existing_collections

    def get_embedding_size(self, model_name: str) -> int:
        assert model_name == QDRANT_DEFAULT_MODEL_NAME
        return 384

    def create_collection(self, **kwargs: Any) -> None:
        self.created_collections.append(kwargs)
        self.existing_collections.add(kwargs["collection_name"])

    def get_collection(self, collection_name: str) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=self.collection_vectors[collection_name]),
                metadata=self.collection_metadata[collection_name],
            )
        )

    def update_collection(self, **kwargs: Any) -> None:
        self.updated_collections.append(kwargs)

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)
        self.existing_collections.discard(collection_name)

    def upsert(self, **kwargs: Any) -> QdrantUpdateResult:
        self.upsert_calls.append(kwargs)
        return QdrantUpdateResult(operation_id=1, status="completed")


class LegacySchemaQdrantClient(RecordingQdrantClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.existing_collections = {
            "find_files__example-lib",
            "analysis__example-lib",
        }
        self.collection_vectors = {
            collection_name: {}
            for collection_name in self.existing_collections
        }
        self.collection_metadata = {
            collection_name: {}
            for collection_name in self.existing_collections
        }


class UnownedIncompatibleQdrantClient(RecordingQdrantClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.collection_vectors["find_files__example-lib"] = models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
        )
        self.collection_metadata["find_files__example-lib"] = {"owner": "another-application"}


def _fastembed_config(
    library_path: Path,
    *,
    recreate_incompatible_collections: bool = True,
) -> AgentConfig:
    return AgentConfig(
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            built_in_plugins=[
                PluginConfig(
                    name="qdrant_fastembed",
                    configuration={
                        "recreate_incompatible_collections": recreate_incompatible_collections,
                    },
                )
            ],
        ),
        libraries=[LibraryConfig(name="example-lib", path=library_path)],
    )


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


def test_fastembed_creates_each_missing_collection_with_vector_schema(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "codebase_agent.built_in_plugins.qdrant_client.QdrantClient",
        RecordingQdrantClient,
    )

    plugin = QdrantClientFastembed(_fastembed_config(tmp_path))

    assert plugin.qdrant_client.created_collections == [
        {
            "collection_name": "analysis__example-lib",
            "vectors_config": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
            "metadata": {
                "owner": QDRANT_COLLECTION_SCHEMA_OWNER,
                "schema_version": QDRANT_COLLECTION_SCHEMA_VERSION,
                "embedding_model": QDRANT_DEFAULT_MODEL_NAME,
                "vector_size": 384,
                "distance": models.Distance.COSINE.value,
                "vector_name": "unnamed",
                "payload_schema_version": 1,
            },
        }
    ]
    assert plugin.qdrant_client.updated_collections == [
        {
            "collection_name": "find_files__example-lib",
            "metadata": plugin._collection_metadata(384),
        }
    ]


def test_fastembed_recreates_recognized_legacy_collections(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "codebase_agent.built_in_plugins.qdrant_client.QdrantClient",
        LegacySchemaQdrantClient,
    )

    plugin = QdrantClientFastembed(_fastembed_config(tmp_path))

    assert plugin.qdrant_client.deleted_collections == [
        "find_files__example-lib",
        "analysis__example-lib",
    ]
    assert [
        call["collection_name"]
        for call in plugin.qdrant_client.created_collections
    ] == [
        "find_files__example-lib",
        "analysis__example-lib",
    ]
    assert all(
        call["metadata"] == plugin._collection_metadata(384)
        for call in plugin.qdrant_client.created_collections
    )


def test_fastembed_refuses_to_delete_unowned_collection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "codebase_agent.built_in_plugins.qdrant_client.QdrantClient",
        UnownedIncompatibleQdrantClient,
    )

    with pytest.raises(RuntimeError, match="Refusing to delete"):
        QdrantClientFastembed(_fastembed_config(tmp_path))


def test_fastembed_can_disable_automatic_legacy_recreation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "codebase_agent.built_in_plugins.qdrant_client.QdrantClient",
        LegacySchemaQdrantClient,
    )

    with pytest.raises(RuntimeError, match="automatic recreation is disabled"):
        QdrantClientFastembed(
            _fastembed_config(tmp_path, recreate_incompatible_collections=False)
        )


def test_fastembed_upsert_uses_qdrant_document_inference(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "codebase_agent.built_in_plugins.qdrant_client.QdrantClient",
        RecordingQdrantClient,
    )
    plugin = QdrantClientFastembed(_fastembed_config(tmp_path))

    plugin.upsert(
        "example-lib",
        ClientRequestType.analysis,
        "Where is the parser?",
        "It is in parser.py.",
    )

    [call] = plugin.qdrant_client.upsert_calls
    [point] = call["points"]
    assert call["collection_name"] == "analysis__example-lib"
    assert isinstance(point.vector, models.Document)
    assert point.vector.text == "Where is the parser?"
    assert point.vector.model == QDRANT_DEFAULT_MODEL_NAME
    assert point.payload == {
        "query": "Where is the parser?",
        "response": "It is in parser.py.",
    }
