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
]

from abc import ABC
from dataclasses import dataclass
import uuid

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
import os
from typing import Dict, List, Optional, Any, Set


QDRANT_CACHE_RELATIVE_DIR = "./models/fastembed"
QDRANT_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME_TEMPLATE__FIND_FILES: str = "find_files__{library_name}"
COLLECTION_NAME_TEMPLATE__ANALYSIS: str = "analysis__{library_name}"
QDRANT_PLUGIN_NAMES: List[str] = [
    "qdrant_fastembed",
    "qdrant_fastembed_gpu",
    "qdrant_cloud",
]


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
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name not in self.allowed_tool_names():
            return {"ok": False, "error": {"code": "tool_not_enabled", "message": tool_name}}

        try:
            if tool_name == "vectordb__find_cached_queries":
                result = self.find_cached_queries(
                    library_name=library_name,
                    client_request_type=client_request_type,
                    query=str(arguments.get("query", "")),
                    top_k=int(arguments.get("top_k", 5)),
                )
                return {
                    "ok": True,
                    "result": {
                        "cached_queries": [
                            {"query": item.query, "query_id": item.query_id}
                            for item in result
                        ]
                    },
                }
            if tool_name == "vectordb__get_cached_response":
                result = self.get_cached_response(
                    library_name=library_name,
                    client_request_type=client_request_type,
                    query_id=str(arguments.get("query_id", "")),
                )
                return {"ok": True, "result": {"response": result}}
        except Exception as exc:
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

        embeddings = list(self.qdrant_client.embed(
            texts=texts,
            model=self.embedding_model_name
        ))

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
