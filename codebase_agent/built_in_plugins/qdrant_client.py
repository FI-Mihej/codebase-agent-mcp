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


__all__ = [
    "QdrantClientABC",
    "QdrantClientFastembed",
    "QdrantClientCloud",
    "QdrantCollectionType",
    "QdrantCachedQuery",
    "QdrantUpdateResult",
    "qdrant_client_factory",
    "QDRANT_DEFAULT_TOP_K",
]

from abc import ABC
from dataclasses import dataclass, asdict
import logging
import uuid
import json

from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import (
    UpdateResult as QdrantUpdateResult, 
    ScoredPoint as QdrantScoredPoint,
    Record as QdrantRecord,
)
from codebase_agent.types import (
    ClientRequestType,
    PluginABC,
    ToolResult,
)
from pathlib import Path
from enum import Enum
from codebase_agent.config import (
    AgentConfig, 
    load_config, 
    validate_library_path, 
    LibraryConfig, 
    PluginConfig,
    get_app_name,
)
from cengal.file_system.app_fs_structure.app_dir_path import AppDirectoryType, AppDirPath, app_dir_path
from cengal.introspection.inspect import gsodi
import os
from typing import Dict, List, Optional, Any, Set, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from codebase_agent.app_context import AppContext


logger = logging.getLogger(__name__)

QDRANT_CACHE_RELATIVE_DIR = "./models/fastembed"
QDRANT_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_COLLECTION_SCHEMA_OWNER = "codebase-agent-mcp"
QDRANT_COLLECTION_SCHEMA_VERSION = 1
COLLECTION_NAME_TEMPLATE__FIND_FILES: str = "find_files__{library_name}"
COLLECTION_NAME_TEMPLATE__ANALYSIS: str = "analysis__{library_name}"
QDRANT_PLUGIN_NAMES: List[str] = [
    "qdrant_fastembed",
    "qdrant_fastembed_gpu",
    "qdrant_cloud",
]
QDRANT_DEFAULT_TOP_K: int = 5


def ensure_qdrant_cache_dir_path(qdrant_cache_dir_path: Path | str | None = None) -> Path:
    qdrant_cache_dir_path: Path
    if (qdrant_cache_dir_path is None) or (not isinstance(qdrant_cache_dir_path, (str, Path))):
        qdrant_cache_dir = Path(app_dir_path(AppDirectoryType.local_data, get_app_name(), with_structure=True, ensure_dir=True))
        qdrant_cache_dir = (qdrant_cache_dir / QDRANT_CACHE_RELATIVE_DIR).resolve()
    elif qdrant_cache_dir_path.is_absolute():
        qdrant_cache_dir = Path(qdrant_cache_dir_path).resolve()
        qdrant_cache_dir = (qdrant_cache_dir / QDRANT_CACHE_RELATIVE_DIR).resolve()
    else:
        qdrant_cache_dir = Path(app_dir_path(AppDirectoryType.local_data, get_app_name(), with_structure=True, ensure_dir=True))
        qdrant_cache_dir = (qdrant_cache_dir / qdrant_cache_dir_path).resolve()
    
    return qdrant_cache_dir


def apply_qdrant_cache_dir_path(qdrant_cache_dir: Path | str | None = None) -> Optional[str]:
    qdrant_cache_dir = qdrant_cache_dir or ensure_qdrant_cache_dir_path()
    original_env_value = os.environ.get("FASTEMBED_CACHE_PATH", None)
    os.environ["FASTEMBED_CACHE_PATH"] = str(qdrant_cache_dir)
    return original_env_value


def ensure_qdrant_models(config: AgentConfig, qdrant_cache_dir_path: Path | str | None = None) -> None:
    qdrant_cache_dir_path = ensure_qdrant_cache_dir_path(qdrant_cache_dir_path)
    embedding_model_names: Set[str] = set()
    plugin_name: str
    for plugin_name in QDRANT_PLUGIN_NAMES:
        plugin_config: Optional[PluginConfig] = config.openai_compatible.allowed_built_in_plugin_config(plugin_name)
        if plugin_config is not None:
            embedding_model_name = plugin_config.configuration.get("model_name", QDRANT_DEFAULT_MODEL_NAME)
            embedding_model_names.add(embedding_model_name)
    
    print()
    embedding_model_names_list: List[str] = sorted(list(embedding_model_names))
    if embedding_model_names_list:
        for embedding_model_name in embedding_model_names_list:
            print(f"Ensuring Qdrant embedding model '{embedding_model_name}' is available in cache directory \"{qdrant_cache_dir_path}\"...")
            client = QdrantClient(":memory:")
            try:
                client.set_model(
                    embedding_model_name=embedding_model_name,
                )
            finally:
                client.close()

            print("Done.")
    else:
        print("No allowed Qdrant plugins with specified embedding models were found in the configuration. Skipping the model cache check.")


class QdrantCollectionType(Enum):
    find_files = "find_files"
    analysis = "analysis"


def qdrant_collection_type_from_request_type(request_type: ClientRequestType) -> QdrantCollectionType:
    if request_type == ClientRequestType.find_files:
        return QdrantCollectionType.find_files
    elif request_type == ClientRequestType.analysis:
        return QdrantCollectionType.analysis
    else:
        raise ValueError(f"Invalid request type: {request_type}")


@dataclass
class QdrantCachedQuery:
    query: str
    query_id: str


class QdrantClientABC(PluginABC, ABC):
    @staticmethod
    def collection_name(
        library_name: str, 
        client_request_type: ClientRequestType,
    ) -> str:
        collection_type = qdrant_collection_type_from_request_type(client_request_type)
        if collection_type == QdrantCollectionType.find_files:
            return COLLECTION_NAME_TEMPLATE__FIND_FILES.format(library_name=library_name)
        elif collection_type == QdrantCollectionType.analysis:
            return COLLECTION_NAME_TEMPLATE__ANALYSIS.format(library_name=library_name)
        else:
            raise ValueError(f"Invalid collection type: {collection_type}")
    
    def upsert(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        response: str,
        suppress_exceptions: bool = False,
    ) -> Optional[QdrantUpdateResult]:
        ...
    
    def find_cached_queries(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        top_k: int = 5,
    ) -> list[QdrantCachedQuery]:
        ...
    
    def get_cached_response(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query_id: str,
    ) -> Optional[str]:
        ...

    def execute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: Union[str, Dict[str, Any]],
    ) -> ToolResult:
        if tool_name not in self.allowed_tool_names():
            return {"ok": False, "error": {"code": "tool_not_enabled", "message": tool_name}}

        try:
            if tool_name == "vectordb__find_cached_queries":
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                
                cached_queries: List[QdrantCachedQuery] = self.find_cached_queries(
                    library_name=library_name,
                    client_request_type=client_request_type,
                    query=str(arguments.get("query", "")),
                    top_k=int(arguments.get("top_k", 5)),
                )
                logger.debug(f"QdrantClient found {len(cached_queries)} cached queries for library '{library_name}', request type '{client_request_type}' and arguments `{arguments}`: \n{gsodi(cached_queries)}")
                find_cached_queries_result: Dict[str, Any] = {
                    "ok": True,
                    "result": {
                        "cached_queries": [
                            {"query": item.query, "query_id": item.query_id}
                            for item in cached_queries
                        ]
                    },
                }
                return json.dumps(find_cached_queries_result, indent=2, ensure_ascii=False)
            if tool_name == "vectordb__get_cached_response":
                cached_response: str = self.get_cached_response(
                    library_name=library_name,
                    client_request_type=client_request_type,
                    query_id=str(arguments.get("query_id", "")),
                )
                logger.debug(f"QdrantClient found cached response for library '{library_name}', request type '{client_request_type}' and arguments `{arguments}`: \n{gsodi(cached_response)}")
                get_cached_response_result = {"ok": True, "result": {"response": cached_response}}
                return json.dumps(get_cached_response_result, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.exception(f"Error executing QdrantClient tool '{tool_name}' for library '{library_name}', request type '{client_request_type}' and arguments `{arguments}`: {exc}")
            return {"ok": False, "error": {"code": "vectordb_tool_error", "message": str(exc)}}

        return {"ok": False, "error": {"code": "tool_not_enabled", "message": tool_name}}

    async def aexecute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        raise NotImplementedError("QdrantClientABC does not support async execution.")
    
    def executor(self) -> Any:
        return self.execute

    def all_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "vectordb__find_cached_queries",
                    "description": (
                        "Find semantically similar cached queries previously answered for "
                        "the current library and request type."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Current user query or search intent to match against cached queries.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of cached query candidates to return.",
                                "default": 5,
                                "minimum": 1,
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
                    "name": "vectordb__get_cached_response",
                    "description": "Retrieve a previously cached response by query_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_id": {
                                "type": "string",
                                "description": "Opaque query id returned by vectordb__find_cached_queries.",
                            },
                        },
                        "required": ["query_id"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        allowed_names = set(self.allowed_tool_names())
        return [tool for tool in self.all_tool_definitions() if tool["function"]["name"] in allowed_names]


class QdrantClientFake(QdrantClientABC):
    def __init__(
        self, 
        config: AgentConfig, 
        plugin_name: str = "qdrant_fastembed",
    ):
        self._name = plugin_name
        self.plugin_config: PluginConfig = (
            config.openai_compatible.allowed_built_in_plugin_config(plugin_name)
            or PluginConfig(name=plugin_name)
        )
        self.qdrant_client = None
        self.embedding_model_name = self.plugin_config.configuration.get("model_name", QDRANT_DEFAULT_MODEL_NAME)
        self._app_context: Optional[AppContext] = None

    def set_app_context(self, app_context):
        self._app_context = app_context
    
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
        return list()
    
    def get_cached_response(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query_id: str,
    ) -> Optional[str]:
        return None
    
    def name(self) -> str:
        return self._name

    def all_tools_names(self) -> list[str]:
        return list()

    def allowed_tool_names(self) -> list[str]:
        return list()
    
    def worker_by_tool_name(self, tool_name: str) -> Optional[Any]:
        return None


class QdrantClientFastembed(QdrantClientABC):
    def __init__(
        self, 
        config: AgentConfig, 
        plugin_name: str = "qdrant_fastembed",
    ):
        self._name = plugin_name
        self.plugin_config: PluginConfig = config.openai_compatible.allowed_built_in_plugin_config(plugin_name)
        init_config = self.plugin_config.configuration.get("init", {})
        self.qdrant_client = QdrantClient(**init_config)
        self.embedding_model_name = self.plugin_config.configuration.get("model_name", QDRANT_DEFAULT_MODEL_NAME)
        self.recreate_incompatible_collections = self.plugin_config.configuration.get(
            "recreate_incompatible_collections",
            True,
        )
        if not isinstance(self.recreate_incompatible_collections, bool):
            raise ValueError("recreate_incompatible_collections must be a boolean.")

        embedding_size = self.qdrant_client.get_embedding_size(self.embedding_model_name)
        libraries = config.allowed_libraries()
        if libraries:
            library_config: LibraryConfig
            for library_config in libraries:
                library_path = library_config.path
                if library_path:
                    collection_name__find_files = self.collection_name(library_name=library_config.name, client_request_type=ClientRequestType.find_files)
                    collection_name__analysis = self.collection_name(library_name=library_config.name, client_request_type=ClientRequestType.analysis)
                    for collection_name in (
                        collection_name__find_files,
                        collection_name__analysis,
                    ):
                        self._ensure_collection(collection_name, embedding_size)
        self._app_context: Optional[AppContext] = None

    def set_app_context(self, app_context):
        self._app_context = app_context

    def _collection_metadata(self, embedding_size: int) -> Dict[str, Any]:
        return {
            "owner": QDRANT_COLLECTION_SCHEMA_OWNER,
            "schema_version": QDRANT_COLLECTION_SCHEMA_VERSION,
            "embedding_model": self.embedding_model_name,
            "vector_size": embedding_size,
            "distance": models.Distance.COSINE.value,
            "vector_name": "unnamed",
            "payload_schema_version": 1,
        }

    @staticmethod
    def _has_expected_vector_schema(vectors_config: Any, embedding_size: int) -> bool:
        return (
            isinstance(vectors_config, models.VectorParams)
            and vectors_config.size == embedding_size
            and vectors_config.distance == models.Distance.COSINE
        )

    @staticmethod
    def _has_expected_metadata(
        actual_metadata: Dict[str, Any],
        expected_metadata: Dict[str, Any],
    ) -> bool:
        return all(actual_metadata.get(key) == value for key, value in expected_metadata.items())

    def _create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        metadata: Dict[str, Any],
    ) -> None:
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_size,
                distance=models.Distance.COSINE,
            ),
            metadata=metadata,
        )

    def _ensure_collection(self, collection_name: str, embedding_size: int) -> None:
        expected_metadata = self._collection_metadata(embedding_size)
        if not self.qdrant_client.collection_exists(collection_name):
            self._create_collection(collection_name, embedding_size, expected_metadata)
            return

        collection_info = self.qdrant_client.get_collection(collection_name)
        vectors_config = collection_info.config.params.vectors
        actual_metadata = collection_info.config.metadata or {}
        vector_schema_matches = self._has_expected_vector_schema(vectors_config, embedding_size)

        if vector_schema_matches and self._has_expected_metadata(actual_metadata, expected_metadata):
            return

        # Collections created by the immediately preceding application version already have
        # the correct vector schema but no ownership metadata. Adopt them without losing cache.
        if vector_schema_matches and not actual_metadata:
            logger.info("Adding schema metadata to compatible Qdrant collection %r.", collection_name)
            self.qdrant_client.update_collection(
                collection_name=collection_name,
                metadata=expected_metadata,
            )
            return

        is_owned_collection = actual_metadata.get("owner") == QDRANT_COLLECTION_SCHEMA_OWNER
        is_known_legacy_schema = not actual_metadata and isinstance(vectors_config, dict) and not vectors_config
        if not self.recreate_incompatible_collections:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' has an incompatible schema and "
                "automatic recreation is disabled."
            )

        if not (is_owned_collection or is_known_legacy_schema):
            raise RuntimeError(
                f"Refusing to delete Qdrant collection '{collection_name}' because it is not "
                f"owned by {QDRANT_COLLECTION_SCHEMA_OWNER!r} and is not a recognized legacy schema."
            )

        logger.warning("Recreating incompatible Qdrant cache collection %r.", collection_name)
        self.qdrant_client.delete_collection(collection_name=collection_name)
        self._create_collection(collection_name, embedding_size, expected_metadata)
    
    def upsert(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        response: str,
        suppress_exceptions: bool = False,
    ) -> Optional[QdrantUpdateResult]:
        collection_type: QdrantCollectionType = qdrant_collection_type_from_request_type(client_request_type)
        if not query or not response:
            if suppress_exceptions:
                return None
            
            raise ValueError("Query and response must not be empty.")

        text: str = query
        payload: Dict[str, str] = {"query": query, "response": response}
        combined: str = f"{query} {response}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined))

        texts = [text]
        payloads = [payload]
        points_ids = [point_id]

        embeddings = [
            models.Document(text=text, model=self.embedding_model_name)
            for text in texts
        ]

        points_to_upsert = list()
        for i in range(len(points_ids)):
            points_to_upsert.append(
                models.PointStruct(
                    id=points_ids[i],
                    vector=embeddings[i],
                    payload=payloads[i]
                )
            )

        return self.qdrant_client.upsert(
            collection_name=self.collection_name(library_name=library_name, client_request_type=client_request_type), 
            points=points_to_upsert
        )
    
    def find_cached_queries(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        top_k: int = 5,
    ) -> List[QdrantCachedQuery]:
        collection_type: QdrantCollectionType = qdrant_collection_type_from_request_type(client_request_type)
        points: list[QdrantScoredPoint] = self.qdrant_client.query_points(
            collection_name=self.collection_name(library_name=library_name, client_request_type=client_request_type),
            query=models.Document(text=query, model=self.embedding_model_name),
            with_payload=True,
            limit=top_k,
        ).points
        if not points:
            return list()
        
        sorted_points = sorted(points, key=lambda p: p.score, reverse=True)
        top_k_points = sorted_points[:top_k]
        results: List[QdrantCachedQuery] = list()
        for point in top_k_points:
            results.append(QdrantCachedQuery(query=point.payload.get("query", ""), query_id=str(point.id)))
        
        return results
    
    def get_cached_response(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query_id: str,
    ) -> Optional[str]:
        collection_type: QdrantCollectionType = qdrant_collection_type_from_request_type(client_request_type)
        ids_to_check = [query_id]
        records: List[QdrantRecord] = self.qdrant_client.retrieve(
            collection_name=self.collection_name(library_name=library_name, client_request_type=client_request_type),
            ids=ids_to_check,
            with_payload=True,
            with_vectors=False
        )
        if not records:
            return None
        
        record = records[0]
        if not record.payload:
            return None
        
        return record.payload.get("response", None)
    
    def name(self) -> str:
        return self._name

    def all_tools_names(self) -> list[str]:
        return [
            "vectordb__find_cached_queries",
            "vectordb__get_cached_response",
        ]

    def allowed_tool_names(self) -> list[str]:
        denied_tools: Set[str] = set(self.plugin_config.denied_tools) or set()
        all_tools: Set[str] = set(self.all_tools_names())
        allowed_tools: Set[str] = all_tools - denied_tools
        return sorted(list(allowed_tools))
    
    def worker_by_tool_name(self, tool_name: str) -> Optional[Any]:
        worker_map = {
            "vectordb__find_cached_queries": self.find_cached_queries,
            "vectordb__get_cached_response": self.get_cached_response,
        }
        return worker_map.get(tool_name, None)

    def execute(
        self, 
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str, 
        arguments: dict[str, Any]
    ) -> ToolResult:
        return super().execute(library_name, client_request_type, tool_name, arguments)


class QdrantClientCloud(QdrantClientABC):
    def __init__(
        self, 
        config: AgentConfig, 
        plugin_name: str = "qdrant_fastembed",
    ):
        self._name = plugin_name
        self.plugin_config: PluginConfig = config.openai_compatible.allowed_built_in_plugin_config(plugin_name)
        init_config = self.plugin_config.configuration.get("init", {})
        self.qdrant_client = QdrantClient(**init_config)
        self.embedding_model_name = self.plugin_config.configuration.get("model_name", QDRANT_DEFAULT_MODEL_NAME)
        libraries = config.allowed_libraries()
        if libraries:
            library_config: LibraryConfig
            for library_config in libraries:
                library_path = library_config.path
                if library_path:
                    collection_name__find_files = self.collection_name(library_name=library_config.name, client_request_type=ClientRequestType.find_files)
                    collection_name__analysis = self.collection_name(library_name=library_config.name, client_request_type=ClientRequestType.analysis)
                    if not self.qdrant_client.collection_exists(collection_name__find_files):
                        self.qdrant_client.create_collection(
                            collection_name=collection_name__find_files,
                        )
                        self.qdrant_client.create_collection(
                            collection_name=collection_name__analysis,
                        )
        self._app_context: Optional[AppContext] = None

    def set_app_context(self, app_context):
        self._app_context = app_context
    
    def upsert(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        response: str,
        suppress_exceptions: bool = False,
    ) -> Optional[QdrantUpdateResult]:
        collection_type: QdrantCollectionType = qdrant_collection_type_from_request_type(client_request_type)
        if not query or not response:
            if suppress_exceptions:
                return None
            
            raise ValueError("Query and response must not be empty.")

        text: str = query
        payload: Dict[str, str] = {"query": query, "response": response}
        combined: str = f"{query} {response}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined))

        texts = [text]
        payloads = [payload]
        points_ids = [point_id]

        embeddings = [models.Document(text=text, model=self.embedding_model_name) for text in texts]

        points_to_upsert = list()
        for i in range(len(points_ids)):
            points_to_upsert.append(
                models.PointStruct(
                    id=points_ids[i],
                    vector=embeddings[i],
                    payload=payloads[i]
                )
            )

        return self.qdrant_client.upsert(
            collection_name=self.collection_name(library_name=library_name, client_request_type=client_request_type), 
            points=points_to_upsert
        )
    
    def find_cached_queries(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query: str,
        top_k: int = 5,
    ) -> list[QdrantCachedQuery]:
        collection_type: QdrantCollectionType = qdrant_collection_type_from_request_type(client_request_type)
        points: list[QdrantScoredPoint] = self.qdrant_client.query_points(
            collection_name=self.collection_name(library_name=library_name, client_request_type=client_request_type),
            query=models.Document(text=query, model=self.embedding_model_name),
            with_payload=True,
            limit=top_k,
        ).points
        if not points:
            return list()
        
        sorted_points = sorted(points, key=lambda p: p.score, reverse=True)
        top_k_points = sorted_points[:top_k]
        results: List[QdrantCachedQuery] = list()
        for point in top_k_points:
            results.append(QdrantCachedQuery(query=point.payload.get("query", ""), query_id=str(point.id)))
        
        return results
    
    def get_cached_response(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        query_id: str,
    ) -> Optional[str]:
        collection_type: QdrantCollectionType = qdrant_collection_type_from_request_type(client_request_type)
        ids_to_check = [query_id]
        records: List[QdrantRecord] = self.qdrant_client.retrieve(
            collection_name=self.collection_name(library_name=library_name, client_request_type=client_request_type),
            ids=ids_to_check,
            with_payload=True,
            with_vectors=False
        )
        if not records:
            return None
        
        record = records[0]
        if not record.payload:
            return None
        
        return record.payload.get("response", None)
    
    def name(self) -> str:
        return self._name

    def all_tools_names(self) -> list[str]:
        return [
            "vectordb__find_cached_queries",
            "vectordb__get_cached_response",
        ]

    def allowed_tool_names(self) -> list[str]:
        denied_tools: Set[str] = set(self.plugin_config.denied_tools) or set()
        all_tools: Set[str] = set(self.all_tools_names())
        allowed_tools: Set[str] = all_tools - denied_tools
        return sorted(list(allowed_tools))
    
    def worker_by_tool_name(self, tool_name: str) -> Optional[Any]:
        worker_map = {
            "vectordb__find_cached_queries": self.find_cached_queries,
            "vectordb__get_cached_response": self.get_cached_response,
        }
        return worker_map.get(tool_name, None)


def qdrant_client_factory(config: AgentConfig) -> QdrantClientABC:
    """Factory function to create a QdrantClientFastembed instance."""
    plugin_config: PluginConfig
    plugin_config = config.openai_compatible.allowed_built_in_plugin_config("qdrant_fastembed")
    if plugin_config is not None:
        return QdrantClientFastembed(config=config, plugin_name="qdrant_fastembed")
    
    plugin_config = config.openai_compatible.allowed_built_in_plugin_config("qdrant_fastembed_gpu")
    if plugin_config is not None:
        return QdrantClientFastembed(config=config, plugin_name="qdrant_fastembed_gpu")
    
    plugin_config = config.openai_compatible.allowed_built_in_plugin_config("qdrant_cloud")
    if plugin_config is not None:
        return QdrantClientCloud(config=config, plugin_name="qdrant_cloud")

    return QdrantClientFake(config=config, plugin_name="qdrant_fake")
