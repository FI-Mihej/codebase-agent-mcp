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


"""OpenAI compatible chat-completions client."""


from __future__ import annotations

import json
import re
import math
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Optional, List, Dict, TYPE_CHECKING

import httpx
from openai import AsyncStream, APIError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, NotFoundError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCallUnion
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.lib.streaming.chat import ChatCompletionStreamState, ParsedChatCompletionSnapshot

from codebase_agent.config import OpenAICompatibleConfig, LibraryConfig
from codebase_agent.types import (
    ClientRequestType,
    OpenAICompatibleConnectionError,
    OpenAICompatibleContextOverflowError,
    OpenAICompatibleContentFilterError,
    OpenAICompatibleTimeoutError,
    MalformedOpenAICompatibleResponse,
    DirectErrorResponseForClientLLM,
    MissingModelError,
    ToolResult,
    ChatCompletionTokenUsage,
)
from codebase_agent.types import (
    PluginABC,
)
from codebase_agent.app_context import AppContext
from codebase_agent.prompts import (
    build_prompt_for__context_compression,
    build_prompt_for__llm_anti_stuck_mechanism,
    build_openai_response_format_for__llm_anti_stuck_mechanism,
    build_prompt_for__llm_goal_fulfillment_auditor,
    build_openai_response_format_for__llm_goal_fulfillment_auditor,
)
from codebase_agent.progress import report_subllm_progress
from cengal.introspection.inspect import gsodi, gmsodv, entity_owning_module_info_and_owning_path, is_async, is_callable
from cengal.time_management.run_time import RT
import copy
import inspect
import logging
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dataclasses import dataclass, field, asdict
if TYPE_CHECKING:
    from codebase_agent.built_in_plugins.qdrant_client import QdrantCachedQuery, QDRANT_DEFAULT_TOP_K


logger = logging.getLogger(__name__)


CompletionCreate = Callable[..., Awaitable[Any]]

response_format__for__find_files: Dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "related_files_search_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "files": {
                    "type": "array",
                    "description": "Files ranked by descending relevance to the user's request.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path relative to the configured codebase root.",
                            },
                            "score": {
                                "type": "number",
                                "description": "Relevance score in the range 0.0 to 1.0.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Short reason this file appears relevant.",
                            },
                            "evidence": {
                                "type": "array",
                                "description": "Concise supporting symbols, terms, paths, or other evidence.",
                                "items": {
                                    "type": "string",
                                },
                            },
                        },
                        "required": ["path", "score", "reason", "evidence"],
                    },
                },
                "notes": {
                    "type": "string",
                    "description": "Short coverage notes, including limitations or uncertainty.",
                },
            },
            "required": ["files", "notes"],
        },
    },
}


class LlmAntiStuckMechanismResult(BaseModel):
    is_llm_stuck: bool
    llm_agent_instructions: Optional[str] = None


class LlmTaskGoalFulfillmentAuditorResult(BaseModel):
    is_task_goal_fulfilled: bool
    llm_agent_instructions: Optional[str] = None


@dataclass
class LlmAntiStuckMechanismState:
    tool_calls_per_job: Dict[str, int] = field(default_factory=dict)


def gather_chat_completion_token_usage(completion: ChatCompletion) -> ChatCompletionTokenUsage:
    """Extract token usage statistics from a ChatCompletion response."""

    all_input_tokens: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    usage: CompletionUsage = completion.usage
    if completion.usage is not None:
        all_input_tokens = usage.prompt_tokens
        prompt_tokens_details: PromptTokensDetails = usage.prompt_tokens_details
        if prompt_tokens_details is not None:
            cached_input_tokens = prompt_tokens_details.cached_tokens or 0

        output_tokens = usage.completion_tokens

    input_tokens = all_input_tokens - cached_input_tokens

    return ChatCompletionTokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )


async def _report_subllm_progress(
    *,
    app_context: AppContext,
    job_id: str | None,
    **increments: int,
) -> None:
    """Record explicit job metrics plus the manager's lifetime aggregate."""

    await app_context.analysis_jobs.report_subllm_progress(job_id, increments)
    if job_id is None:
        # Preserve the public reporter hook for direct callers outside the job
        # manager. Job attribution itself does not rely on ambient state.
        await report_subllm_progress(**increments)


def normalize_base_url(base_url: str) -> str:
    """Normalize OpenAI compatible base URLs to an OpenAI compatible `/v1` endpoint."""

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


class OpenAICompatibleClient:
    """Client for one-shot OpenAI compatible consultations."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        completion_create: CompletionCreate | None = None,
    ) -> None:
        self._config: OpenAICompatibleConfig = config
        if completion_create is None:
            client = AsyncOpenAI(
                base_url=config.normalized_base_url(),
                api_key=config.api_key,
                timeout=httpx.Timeout(config.timeout_seconds),
            )
            self._completion_create = client.chat.completions.create
        else:
            self._completion_create = completion_create

        self.full_llm_anti_stuck_mechanism_state: LlmAntiStuckMechanismState = LlmAntiStuckMechanismState()

    async def consult(
        self,
        *,
        app_context: AppContext,
        job_id: str | None = None,
        library: LibraryConfig,
        client_request_type: ClientRequestType,
        system_prompt: str,
        query: str,
        enabled_tool_names: set[str] | None = None,
    ) -> str:
        """Run a fresh OpenAI compatible chat-completions request for one MCP tool call."""

        result: str
        response: ChatCompletion

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        if not self._enabled_tool_names(app_context=app_context):
            try:
                response = await self._create_completion(
                    app_context=app_context,
                    job_id=job_id,
                    messages=messages,
                    response_format=response_format__for__find_files if ClientRequestType.find_files == client_request_type else None,
                )
            except OpenAICompatibleContextOverflowError:
                raise DirectErrorResponseForClientLLM("The request could not be completed because the accumulated prompt exceeded the model's context window. Retry with a narrower request.")

            response_message_content: str = _extract_final_content(response)
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message_content,
                },
            )
            try:
                result = await self._compress_conversation_history_impl(
                    app_context=app_context,
                    job_id=job_id,
                    library=library,
                    openai_compatible=self._config,
                    original_prompt=query,
                    conversation_history=messages[1:],
                )
            except OpenAICompatibleContextOverflowError:
                # In this case our settings are irrelevant: the accumulated prompt exceeded the model's context window on the first request.
                raise DirectErrorResponseForClientLLM("The request could not be completed because the accumulated prompt exceeded the model's context window. Retry with a narrower request.")

            # result = _sanitize_final_response(result)
            app_context.qdrant_client.upsert(
                library_name=library.name,
                client_request_type=client_request_type,
                query=query,
                response=result,
                suppress_exceptions=True,
            )
            if ClientRequestType.find_files == client_request_type:
                return response.choices[0].message.content if response.choices else json.dumps(dict(), ensure_ascii=False)
            else:
                raise DirectErrorResponseForClientLLM(result)

        tools = self._enabled_tool_definitions(app_context=app_context, enabled_tool_names_filter=enabled_tool_names)
        tool_rounds = 0
        last_assistant_content = ""
        assistant_message: ChatCompletionMessage
        tool_calls: List[ChatCompletionMessageToolCallUnion]
        last_tool_calls: List[ChatCompletionMessageToolCallUnion]
        last_tool_call: Optional[ChatCompletionMessageToolCallUnion]
        last_tool_calls_as_dict: Dict[str, Any]
        last_tool_call_as_dict: Optional[Dict[str, Any]]
        last_tool_name: Optional[str]
        history_compression_attempts = 0

        tool_name = "vectordb__find_cached_queries"
        if tool_name in _enabled_tool_names(tools):
            from codebase_agent.built_in_plugins.qdrant_client import QDRANT_DEFAULT_TOP_K
            vectordb_arguments = {
                "query": query,
                "top_k": QDRANT_DEFAULT_TOP_K,
            }
            logger.debug(f"<<RAG>>. Searching plugin_by_tool_name for \"{tool_name}\"")
            plugin: PluginABC = app_context.plugin_by_tool_name(tool_name)
            if plugin is not None:
                logger.debug(f"<<RAG>>. Plugin found")
                vectordb_suggestions: List[QdrantCachedQuery]
                plugin_executor = plugin.executor()
                await _report_subllm_progress(
                    app_context=app_context,
                    job_id=job_id,
                    tool_calls_made_by_subllm=1,
                )

                with RT() as rt:
                    if is_async(plugin_executor):
                        logger.debug(f"<<RAG>>. Async tool call \"{tool_name}\"")
                        vectordb_suggestions = await plugin_executor(
                            library.name,
                            client_request_type,
                            tool_name,
                            vectordb_arguments,
                        )
                    elif is_callable(plugin_executor):
                        logger.debug(f"<<RAG>>. Sync tool call \"{tool_name}\"")
                        vectordb_suggestions = plugin_executor(
                            library.name,
                            client_request_type,
                            tool_name,
                            vectordb_arguments,
                        )
                    else:
                        raise RuntimeError(f"Uncallable plugin_executor. {gsodi(plugin_executor)}")

                logger.debug(f"<<RAG>>. Tool call time: {rt()}")

                tool_call_id = str(uuid.uuid4())
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "vectordb__find_cached_queries",
                                    "arguments": json.dumps({
                                        "query": query,
                                        "top_k": vectordb_arguments.get("top_k", QDRANT_DEFAULT_TOP_K),
                                    })
                                }
                            }
                        ]
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": vectordb_suggestions,
                    }
                )

        while True:
            overflow_error_occurred = False
            content_filter_error_occurred = False
            try:
                self._full_llm_anti_stuck_mechanism_increment_tool_call_count(job_id)
                messages = await self._llm_anti_stuck_mechanism(
                    app_context=app_context,
                    job_id=job_id,
                    library=library,
                    openai_compatible=self._config,
                    original_prompt=query,
                    conversation_history=messages,
                )
                response = await self._create_completion(
                    app_context=app_context,
                    job_id=job_id,
                    messages=messages,
                    tools=tools,
                    # Some OpenAI-compatible backends cannot combine tool-call
                    # grammar with structured-output grammar.
                    response_format=None,
                    raise_on_length_exceeded=False,
                )
                assistant_message = _extract_assistant_message(response)
                tool_calls = _message_tool_calls(assistant_message)
                assistant_tool_call_message = _assistant_tool_call_message(assistant_message, tool_calls)
                messages.append(assistant_tool_call_message)
                if response.choices:
                    new_choices: List[Choice] = list()
                    for choice in response.choices:
                        if "length" != choice.finish_reason:
                            new_choices.append(choice)

                    if not new_choices:
                        raise OpenAICompatibleContextOverflowError(
                            "OpenAI compatible request exceeded the model's context window (max_tokens)."
                        )

                    history_compression_attempts = 0
                    response.choices = new_choices

                    new_choices = list()
                    for choice in response.choices:
                        if "content_filter" != choice.finish_reason:
                            new_choices.append(choice)

                    if not new_choices:
                        raise OpenAICompatibleContentFilterError(
                            "OpenAI compatible request was blocked by the content filter."
                        )

                    response.choices = new_choices
                else:
                    history_compression_attempts = 0

            except OpenAICompatibleContextOverflowError:
                overflow_error_occurred = True
            except OpenAICompatibleContentFilterError:
                content_filter_error_occurred = True

            if content_filter_error_occurred:
                last_message = messages[-1]
                if "user" == last_message.get("role"):
                    raise DirectErrorResponseForClientLLM("The request could not be completed because your request was blocked by the content filter. Retry with a request that is expected to be allowed by the content filter.")
                elif "tool" == last_message.get("role"):
                    message_before_last_tool = messages[-2] if len(messages) >= 2 else None
                    if "assistant" != message_before_last_tool.get("role"):
                        raise RuntimeError("Unexpected message sequence: last message was tool, but the message before that was not assistant")

                    assistant_message_as_dict = message_before_last_tool
                    last_tool_calls_as_dict = _message_tool_calls__from__dict(assistant_message_as_dict)
                    last_tool_call_as_dict = last_tool_calls_as_dict[-1] if last_tool_calls_as_dict else None
                    last_tool_name = _tool_call_name__from__dict(last_tool_call_as_dict) if last_tool_call_as_dict else None
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id__from__dict(last_tool_call_as_dict),
                            "name": last_tool_name,
                            "content": f"Error: Your most recent request to the `{last_tool_name}` tool was blocked by the content filter. Retry with a request that is expected to produce a tool response allowed by the content filter.",
                        }
                    )
                elif "assistant" == last_message.get("role"):
                    assistant_message_as_dict = last_message
                    last_tool_calls_as_dict = _message_tool_calls__from__dict(assistant_message_as_dict)
                    last_tool_call_as_dict = last_tool_calls_as_dict[-1] if last_tool_calls_as_dict else None
                    last_tool_name = _tool_call_name__from__dict(last_tool_call_as_dict) if last_tool_call_as_dict else None
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id__from__dict(last_tool_call_as_dict),
                            "name": last_tool_name,
                            "content": f"Error: Your most recent message was blocked by the content filter. Retry with a message that is expected to be allowed by the content filter.",
                        }
                    )
                    if last_tool_call_as_dict:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _tool_call_id__from__dict(last_tool_call_as_dict),
                                "name": last_tool_name,
                                "content": f"Error: your most recent request was blocked by the content filter. Retry with a request that is expected to be allowed by the content filter.",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": app_context.config.openai_compatible.llm_guidance_role,
                                "content": f"Error \"Blocked by the context filter\": Your most recent message was blocked by the content filter. Retry with a message that is expected to be allowed by the content filter.",
                            }
                        )
                else:
                    raise RuntimeError(f"Unexpected message role in content filter handling: {last_message.get('role')}")

            compressed_history: str
            if overflow_error_occurred:
                last_message = messages[-1]
                query_message = messages[1]
                if query_message is last_message:
                    raise DirectErrorResponseForClientLLM("The request could not be completed because the accumulated prompt exceeded the model's context window. Retry with a narrower request.")
                elif "tool" == last_message.get("role"):
                    message_before_last_tool = messages[-2] if len(messages) >= 2 else None
                    # if "assistant" != message_before_last_tool.get("role"):
                    #     raise RuntimeError("Unexpected message sequence: last message was tool, but the message before that was not assistant")

                    try:
                        compressed_history = await self._compress_conversation_history_impl(
                            app_context=app_context,
                            job_id=job_id,
                            library=library,
                            openai_compatible=self._config,
                            original_prompt=query,
                            conversation_history=messages[2:-2],
                        )
                    except OpenAICompatibleContextOverflowError:
                        raise DirectErrorResponseForClientLLM("Error 0: The accumulated prompt exceeded the model's context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my (\"CodebaseAgent-MCP\") configuration by reducing the value of \"context_compression_threshold\" and, if necessary, increasing the value of \"max_tokens\" in the \"codebase_agent.config.json\" file.")

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": compressed_history,
                        },
                    ]
                    if "assistant" == message_before_last_tool.get("role"):
                        messages.append(assistant_message_as_dict)
                        assistant_message_as_dict = message_before_last_tool
                        last_tool_calls_as_dict = _message_tool_calls__from__dict(assistant_message_as_dict)
                        last_tool_call_as_dict = last_tool_calls_as_dict[-1] if last_tool_calls_as_dict else None
                        last_tool_name = _tool_call_name__from__dict(last_tool_call_as_dict) if last_tool_call_as_dict else None
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _tool_call_id__from__dict(last_tool_call_as_dict),
                                "name": last_tool_name,
                                "content": "Error: the accumulated prompt, after adding the tool's response, exceeded the model's context window. Retry with a request that is expected to produce a narrower response from the tool.",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": last_message.get("tool_call_id"),
                                "name": last_message.get("name"),
                                "content": "Error: the accumulated prompt, after adding the tool's response, exceeded the model's context window. Retry with a request that is expected to produce a narrower response from the tool.",
                            }
                        )
                elif "assistant" == last_message.get("role"):
                    assistant_message_as_dict = last_message
                    try:
                        compressed_history = await self._compress_conversation_history_impl(
                            app_context=app_context,
                            job_id=job_id,
                            library=library,
                            openai_compatible=self._config,
                            original_prompt=query,
                            conversation_history=messages[2:-1],
                        )
                    except OpenAICompatibleContextOverflowError:
                        raise DirectErrorResponseForClientLLM('Error 1: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": compressed_history,
                        },
                    ]
                    last_tool_calls_as_dict = _message_tool_calls__from__dict(assistant_message_as_dict)
                    last_tool_call_as_dict = last_tool_calls_as_dict[-1] if last_tool_calls_as_dict else None
                    last_tool_name = _tool_call_name__from__dict(last_tool_call_as_dict) if last_tool_call_as_dict else None
                    if last_tool_call_as_dict:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _tool_call_id__from__dict(last_tool_call_as_dict),
                                "name": last_tool_name,
                                "content": f"Error: your most recent request to the `{last_tool_name}` tool caused the accumulated prompt to exceed the model's context window. As a result, the conversation history was compacted. Continue your work by making requests that are expected to produce narrower responses from the tools.",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": app_context.config.openai_compatible.llm_guidance_role,
                                "content": "Error: The accumulated prompt generated by your most recent message exceeded the model's context window. As a result, the conversation history was compacted. Retry with a message that is expected to use fewer tokens.",
                            }
                        )
                elif app_context.config.openai_compatible.llm_guidance_role == last_message.get("role"):
                    developer_message = last_message
                    try:
                        compressed_history = await self._compress_conversation_history_impl(
                            app_context=app_context,
                            job_id=job_id,
                            library=library,
                            openai_compatible=self._config,
                            original_prompt=query,
                            conversation_history=messages[2:-1],
                        )
                    except OpenAICompatibleContextOverflowError:
                        raise DirectErrorResponseForClientLLM('Error 2: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": compressed_history,
                        },
                        developer_message,
                    ]
                else:
                    raise RuntimeError(f"Unexpected message role in context overflow handling: {last_message.get('role')}")

                continue  # Retry the loop with the compacted conversation history

            need_to_stop = False
            if not response.choices:
                need_to_stop = True
            else:
                first_choice: Choice = response.choices[0]
                if "tool_calls" == first_choice.finish_reason:
                    chat_completion_message: ChatCompletionMessage = first_choice.message
                    if not chat_completion_message.tool_calls:
                        need_to_stop = True  # TODO: This situation should be considered a tool-call generation error by the model. Possible solutions are: a) terminate execution with an error; b) attempt a limited number of retries by resending the same chat history to the inference server (if temperature != 0); c) provide or generate guidance to correct the tool-call generation (generating guidance is recommended for runtime experiments, allowing the auditor to suggest different options in case of failure).
                elif "stop" == first_choice.finish_reason:
                    need_to_stop = True
                else:
                    need_to_stop = False

            if 0 < self._config.max_tool_rounds:
                if tool_rounds > self._config.max_tool_rounds:
                    need_to_stop = True

            if need_to_stop:
                try:
                    is_task_goal_fulfilled, messages = await self._llm_goal_fulfillment_auditor(
                        app_context=app_context,
                        job_id=job_id,
                        library=library,
                        openai_compatible=self._config,
                        original_prompt=query,
                        client_request_type=client_request_type,
                        conversation_history=messages,
                    )
                    need_to_stop = is_task_goal_fulfilled
                except OpenAICompatibleContextOverflowError:
                    raise DirectErrorResponseForClientLLM('Error 3: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

            if need_to_stop:
                try:
                    if ClientRequestType.find_files == client_request_type:
                        result = response.choices[0].message.content if response.choices else json.dumps(dict(), ensure_ascii=False)
                    else:
                        result = response.choices[0].message.content if response.choices else str()

                    if not result:
                        result = await self._compress_conversation_history_impl(
                            app_context=app_context,
                            job_id=job_id,
                            library=library,
                            openai_compatible=self._config,
                            original_prompt=query,
                            conversation_history=messages[2:],
                        )

                    app_context.qdrant_client.upsert(
                        library_name=library.name,
                        client_request_type=client_request_type,
                        query=query,
                        response=result,
                        suppress_exceptions=True,
                    )
                    return result
                    # if ClientRequestType.find_files == client_request_type:
                    #     return response.choices[0].message.content if response.choices else json.dumps(dict(), ensure_ascii=False)
                    # else:
                    #     return str(result)
                except OpenAICompatibleContextOverflowError:
                    raise DirectErrorResponseForClientLLM('Error 4: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

            if _should_compress_context(
                response=response,
                max_tokens=app_context.config.openai_compatible.max_tokens,
                threshold=app_context.config.openai_compatible.context_compression_threshold,
            ):
                last_message = messages[-1]
                try:
                    compressed_history = await self._compress_conversation_history_impl(
                        app_context=app_context,
                        job_id=job_id,
                        library=library,
                        openai_compatible=self._config,
                        original_prompt=query,
                        conversation_history=messages[2:],
                    )
                except OpenAICompatibleContextOverflowError:
                    # TODO: In the event of insufficient space in the context window, an alternative compaction method must be called - either one that processes a portion of the chat at a time or a more aggressive one - and the desired number of tokens for the response should be passed to them (taking into account the size of the compactor's prompt).
                    raise DirectErrorResponseForClientLLM('Error 5: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": compressed_history,
                    },
                    {
                        "role": app_context.config.openai_compatible.llm_guidance_role,
                        "content": "Conversation history was compacted. Continue your work.",
                    },
                    last_message,
                ]
                messages = await self._compress_conversation_history(
                    app_context=app_context,
                    job_id=job_id,
                    library=library,
                    openai_compatible=self._config,
                    original_prompt=query,
                    conversation_history=messages,
                )

            assistant_message = _extract_assistant_message(response)
            tool_calls = _message_tool_calls(assistant_message)

            logger.debug(f"<<TOOL_CALL>>. Start.")
            tool_call: ChatCompletionMessageToolCallUnion
            for tool_call in tool_calls:
                tool_name = _tool_call_name(tool_call)
                if tool_name is None:
                    result: ToolResult = {
                        "ok": False,
                        "error": {
                            "code": "tool_name_missing",
                            "message": f"Tool name is missing in the tool call.",
                        },
                    }
                elif self._is_denied_tool_call(tool_name):
                    result: ToolResult = {
                        "ok": False,
                        "error": {
                            "code": "tool_disabled",
                            "message": f"Tool is disabled by configuration: {tool_name}",
                        },
                    }
                elif tool_name not in _enabled_tool_names(tools):
                    result: ToolResult = {
                        "ok": False,
                        "error": {
                            "code": "unknown_tool",
                            "message": f"Unknown tool: {tool_name}. Check tool name correctness.",
                        },
                    }
                else:
                    logger.debug(f"<<TOOL_CALL>>. Parsing tool call arguments for \"{tool_name}\"")
                    arguments = _parse_tool_call_arguments(tool_call, tool_name)
                    logger.debug(f"<<TOOL_CALL>>. Tool call arguments for \"{tool_name}\": {gmsodv(arguments)}")
                    logger.debug(f"<<TOOL_CALL>>. Searching plugin_by_tool_name for \"{tool_name}\"")
                    plugin: PluginABC = app_context.plugin_by_tool_name(tool_name)
                    if plugin is None:
                        logger.debug(f"<<TOOL_CALL>>. Plugin not found for \"{tool_name}\"")
                        result: ToolResult = {
                            "ok": False,
                            "error": {
                                "code": "tool_unreachable",
                                "message": f"Tool is unreachable: {tool_name}. Use other available tools.",
                            },
                        }

                    logger.debug(f"<<TOOL_CALL>>. Plugin found for \"{tool_name}\": {gsodi(plugin)}")
                    plugin_executor = plugin.executor()
                    await _report_subllm_progress(
                        app_context=app_context,
                        job_id=job_id,
                        tool_calls_made_by_subllm=1,
                    )

                    with RT() as rt:
                        if is_async(plugin_executor):
                            logger.debug(f"<<TOOL_CALL>>. Async tool call \"{tool_name}\"")
                            result = await plugin_executor(
                                library.name,
                                client_request_type,
                                tool_name,
                                arguments,
                            )
                        elif is_callable(plugin_executor):
                            logger.debug(f"<<TOOL_CALL>>. Sync tool call \"{tool_name}\"")
                            result = plugin_executor(
                                library.name,
                                client_request_type,
                                tool_name,
                                arguments,
                            )
                        else:
                            raise RuntimeError(f"Uncallable plugin_executor. {gsodi(plugin_executor)}")

                    logger.debug(f"<<TOOL_CALL>>. Tool call \"{tool_name}\" time: {rt()}")

                content: str = json.dumps(result, ensure_ascii=False)
                if self._config.max_tool_response_bytes:
                    if len(content.encode("utf-8")) > self._config.max_tool_response_bytes:
                        truncated_tool_response: str = content[:self._config.max_tool_response_bytes]
                        result = {
                            "ok": False,
                            "error": {
                                "code": "tool_response_too_large",
                                "message": f"Tool response exceeded the maximum allowed size of {self._config.max_tool_response_bytes} bytes.",
                            },
                            "llm_agent_instructions": f"The tool response was truncated to the maximum allowed size by removing its non-fitting tail; the truncated tool response has been placed in the \"truncated_tool_response\" field. Refine and narrow your request to produce shorter responses. In order for your current sub-task to be considered complete, you must recover any lost data caused by truncation through more focused requests, or generate other more focused queries that will allow you to obtain the full, accurate, and detailed information necessary to complete your main task.",
                            "truncated_tool_response": truncated_tool_response,
                        }
                        content = json.dumps(result, ensure_ascii=False)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(tool_call),
                        "name": tool_name,
                        "content": content,
                    }
                )

            tool_rounds += 1

    async def classify_without_tools(
        self,
        *,
        app_context: AppContext,
        job_id: str | None = None,
        system_prompt: str,
        query: str,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Run a no-tool classification request without persistence side effects."""

        response: ChatCompletion = await self._create_completion(
            app_context=app_context,
            job_id=job_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format=response_format,
        )
        return _extract_final_content(response)

    # async def _prepare_conversation_history(
    #     self,
    #     *,
    #     app_context: AppContext,
    #     job_id: str | None,
    #     library: LibraryConfig,
    #     openai_compatible: OpenAICompatibleConfig,
    #     original_prompt: str,
    #     conversation_history: list[dict[str, Any]],
    # ) -> str:
    #     new_conversation_history: list[dict[str, Any]] = list()
    #     for message in conversation_history:
    #         # if message.get("role") not in {app_context.config.openai_compatible.llm_guidance_role, "system"}:
    #         if message.get("role") not in {"system",}:
    #             new_conversation_history.append(message)

    #     conversation_history = new_conversation_history
    #     response: ChatCompletion = await self._create_completion(
    #         app_context=app_context,
    #         job_id=job_id,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": "You compress conversation history for continuation of an OpenAI-compatible tool-calling workflow.",
    #             },
    #             {
    #                 "role": "user",
    #                 "content": build_prompt_for__context_compression(
    #                     app_context=app_context,
    #                     library=library,
    #                     openai_compatible=openai_compatible,
    #                     original_prompt=original_prompt,
    #                     conversation_history=conversation_history,
    #                 ),
    #             },
    #         ]
    #     )
    #     result = _extract_final_content(response)
    #     await _report_subllm_progress(
    #         app_context=app_context,
    #         job_id=job_id,
    #         llm_context_compations_made_by_subharness=1,
    #     )
    #     return result

    # async def _prune_conversation_history(
    #     self,
    #     *,
    #     app_context: AppContext,
    #     job_id: str | None,
    #     library: LibraryConfig,
    #     openai_compatible: OpenAICompatibleConfig,
    #     original_prompt: str,
    #     conversation_history: list[dict[str, Any]],
    # ) -> list[dict[str, Any]]:
    #     new_conversation_history: list[dict[str, Any]] = list()
    #     for message_index, message in enumerate(conversation_history):
    #         if message.get("role") in {"tool",}:
    #             new_conversation_history.append(message)

    #         response: ChatCompletion = await self._create_completion(
    #             app_context=app_context,
    #             job_id=job_id,
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are LLM anti-stuck mechanism.",
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": build_prompt_for__llm_anti_stuck_mechanism(
    #                         app_context=app_context,
    #                         library=library,
    #                         openai_compatible=openai_compatible,
    #                         original_prompt=original_prompt,
    #                         conversation_history=conversation_history_tail,
    #                     ),
    #                 },
    #             ],
    #             response_format=build_openai_response_format_for__llm_anti_stuck_mechanism()
    #         )
    #         result = _extract_final_content(response)

    #         raw: dict[str, Any]
    #         try:
    #             raw = json.loads(result)
    #         except json.JSONDecodeError as exc:
    #             logger.exception(f"LLM anti-stuck mechanism produced invalid JSON: {result}")
    #             raise

    #         try:
    #             llm_anti_stuck_mechanism_result = LlmAntiStuckMechanismResult.model_validate(raw)
    #         except ValidationError as exc:
    #             logger.exception(f"LLM anti-stuck mechanism produced invalid result: {result}")
    #             raise

    #     conversation_history = new_conversation_history
    #     await _report_subllm_progress(
    #         app_context=app_context,
    #         job_id=job_id,
    #         llm_context_compations_made_by_subharness=1,
    #     )
    #     return result

    async def _compress_conversation_history_impl(
        self,
        *,
        app_context: AppContext,
        job_id: str | None,
        library: LibraryConfig,
        openai_compatible: OpenAICompatibleConfig,
        original_prompt: str,
        conversation_history: list[dict[str, Any]],
    ) -> str:
        new_conversation_history: list[dict[str, Any]] = list()
        for message in conversation_history:
            # if message.get("role") not in {app_context.config.openai_compatible.llm_guidance_role, "system"}:
            if message.get("role") not in {"system",}:
                new_conversation_history.append(message)

        conversation_history = new_conversation_history
        max_tokens = app_context.config.openai_compatible.max_tokens
        threshold = app_context.config.openai_compatible.context_compression_threshold
        compaction_result_approximate_size: int = math.ceil(max_tokens * (1 - threshold) * 0.4)
        response: ChatCompletion = await self._create_completion(
            app_context=app_context,
            job_id=job_id,
            messages=[
                {
                    "role": "system",
                    "content": "You compress conversation history for continuation of an OpenAI-compatible tool-calling workflow.",
                },
                {
                    "role": "user",
                    "content": build_prompt_for__context_compression(
                        app_context=app_context,
                        library=library,
                        openai_compatible=openai_compatible,
                        original_prompt=original_prompt,
                        conversation_history=conversation_history,
                        compaction_result_approximate_size=compaction_result_approximate_size,
                    ),
                },
            ]
        )
        result = _extract_final_content(response)
        await _report_subllm_progress(
            app_context=app_context,
            job_id=job_id,
            llm_context_compations_made_by_subharness=1,
        )
        return result

    async def _compress_conversation_history(
        self,
        *,
        app_context: AppContext,
        job_id: str | None,
        library: LibraryConfig,
        openai_compatible: OpenAICompatibleConfig,
        original_prompt: str,
        conversation_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        conversation_history_original = copy.copy(conversation_history)
        first_message = conversation_history[0] if conversation_history else None
        task_message = conversation_history[1] if (conversation_history and (len(conversation_history) >= 2)) else None
        conversation_history_tail = conversation_history[2:]
        compressed_history: str
        try:
            compressed_history = await self._compress_conversation_history_impl(
                app_context=app_context,
                job_id=job_id,
                library=library,
                openai_compatible=self._config,
                original_prompt=original_prompt,
                conversation_history=conversation_history_tail,
            )
        except OpenAICompatibleContextOverflowError:
            # TODO: In the event of insufficient space in the context window, an alternative compaction method must be called - either one that processes a portion of the chat at a time or a more aggressive one - and the desired number of tokens for the response should be passed to them (taking into account the size of the compactor's prompt).
            raise DirectErrorResponseForClientLLM('Error 6: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

        result_conversation_history: list[dict[str, Any]] = [
            first_message,
            task_message if task_message is not None else {"role": "user", "content": original_prompt},
            {
                "role": "assistant",
                "content": compressed_history,
            },
            {
                "role": app_context.config.openai_compatible.llm_guidance_role,
                "content": "Conversation history was compacted. Continue your work.",
            },
        ]
        return result_conversation_history

    def _full_llm_anti_stuck_mechanism_increment_tool_call_count(self, job_id: str) -> None:
        """Increment the count of tool calls for a given job ID."""
        if job_id not in self.full_llm_anti_stuck_mechanism_state.tool_calls_per_job:
            self.full_llm_anti_stuck_mechanism_state.tool_calls_per_job[job_id] = 0

        self.full_llm_anti_stuck_mechanism_state.tool_calls_per_job[job_id] += 1

    def _full_llm_anti_stuck_mechanism_reset_tool_call_count(self, job_id: str) -> None:
        """Reset the count of tool calls for a given job ID."""
        self.full_llm_anti_stuck_mechanism_state.tool_calls_per_job[job_id] = 0

    async def _llm_anti_stuck_mechanism(
        self,
        *,
        app_context: AppContext,
        job_id: str | None,
        library: LibraryConfig,
        openai_compatible: OpenAICompatibleConfig,
        original_prompt: str,
        conversation_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        conversation_history_original = copy.copy(conversation_history)
        first_message = conversation_history[0] if conversation_history else None
        task_message = conversation_history[1] if (conversation_history and (len(conversation_history) >= 2)) else None
        conversation_history_tail = conversation_history[2:]

        full_tool_call_count: int = self.full_llm_anti_stuck_mechanism_state.tool_calls_per_job[job_id]
        full_llm_anti_stuck_mechanism_tool_calls_to_activate: int = app_context.config.openai_compatible.full_anti_stuck_mechanism_tool_calls_to_activate
        tail_llm_anti_stuck_mechanism_tool_calls_to_activate: int = app_context.config.openai_compatible.tail_anti_stuck_mechanism_tool_calls_to_activate
        assistant_messages_num: int = sum(1 for message in conversation_history if message.get("role") == "assistant")
        logger.debug(f"<<llm_anti_stuck_mechanism>>. {assistant_messages_num=}; {full_tool_call_count=}")
        is_tail_only: bool = False
        if (full_tool_call_count >= full_llm_anti_stuck_mechanism_tool_calls_to_activate) and (assistant_messages_num >= 1):
            logger.debug(f"<<llm_anti_stuck_mechanism>>. 0")
            self._full_llm_anti_stuck_mechanism_reset_tool_call_count(job_id)
        else:
            logger.debug(f"<<llm_anti_stuck_mechanism>>. 1")
            if (0 == (full_tool_call_count % tail_llm_anti_stuck_mechanism_tool_calls_to_activate)) and (assistant_messages_num >= 1):
                logger.debug(f"<<llm_anti_stuck_mechanism>>. 1.0")
                conversation_history_tail = conversation_history_tail[-tail_llm_anti_stuck_mechanism_tool_calls_to_activate:]
                is_tail_only = True
            else:
                logger.debug(f"<<llm_anti_stuck_mechanism>>. 1.1")
                return conversation_history_original

        response: ChatCompletion = await self._create_completion(
            app_context=app_context,
            job_id=job_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are LLM anti-stuck mechanism.",
                },
                {
                    "role": "user",
                    "content": build_prompt_for__llm_anti_stuck_mechanism(
                        app_context=app_context,
                        library=library,
                        openai_compatible=openai_compatible,
                        original_prompt=original_prompt,
                        conversation_history=conversation_history_tail,
                        is_tail_only=is_tail_only,
                    ),
                },
            ],
            response_format=build_openai_response_format_for__llm_anti_stuck_mechanism()
        )
        result = _extract_final_content(response)

        raw: dict[str, Any]
        try:
            raw = json.loads(result)
        except json.JSONDecodeError as exc:
            logger.exception(f"LLM anti-stuck mechanism produced invalid JSON: {result}")
            raise

        try:
            llm_anti_stuck_mechanism_result = LlmAntiStuckMechanismResult.model_validate(raw)
        except ValidationError as exc:
            logger.exception(f"LLM anti-stuck mechanism produced invalid result: {result}")
            raise

        result_conversation_history = copy.copy(conversation_history_original)
        if llm_anti_stuck_mechanism_result.is_llm_stuck and llm_anti_stuck_mechanism_result.llm_agent_instructions:
            await _report_subllm_progress(
                app_context=app_context,
                job_id=job_id,
                llm_anti_stuck_mechanism_activations_by_subharness=1,
            )
            result_conversation_history.append(
                {
                    "role": app_context.config.openai_compatible.llm_guidance_role,
                    "content": f"LLM anti-stuck mechanism activated: the LLM was detected to be stuck in a loop or producing unhelpful responses. Guidance to the LLM provided to help it get unstuck: {llm_anti_stuck_mechanism_result.llm_agent_instructions}",
                }
            )

        return result_conversation_history

    async def _llm_goal_fulfillment_auditor(
        self,
        *,
        app_context: AppContext,
        job_id: str | None,
        library: LibraryConfig,
        openai_compatible: OpenAICompatibleConfig,
        original_prompt: str,
        client_request_type: ClientRequestType,
        conversation_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        conversation_history_original = copy.copy(conversation_history)
        first_message = conversation_history[0] if conversation_history else None
        task_message = conversation_history[1] if (conversation_history and (len(conversation_history) >= 2)) else None
        conversation_history_tail = conversation_history[2:]
        response: ChatCompletion = await self._create_completion(
            app_context=app_context,
            job_id=job_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are LLM task goal fulfillment auditor.",
                },
                {
                    "role": "user",
                    "content": build_prompt_for__llm_goal_fulfillment_auditor(
                        app_context=app_context,
                        library=library,
                        openai_compatible=openai_compatible,
                        original_prompt=original_prompt,
                        client_request_type=client_request_type,
                        conversation_history=conversation_history_tail,
                    ),
                },
            ],
            response_format=build_openai_response_format_for__llm_goal_fulfillment_auditor()
        )
        result = _extract_final_content(response)

        raw: dict[str, Any]
        try:
            raw = json.loads(result)
        except json.JSONDecodeError as exc:
            logger.exception(f"LLM task goal fulfillment auditor produced invalid JSON: {result}")
            raise

        try:
            llm_goal_fulfillment_auditor_result = LlmTaskGoalFulfillmentAuditorResult.model_validate(raw)
        except ValidationError as exc:
            logger.exception(f"LLM task goal fulfillment auditor produced invalid result: {result}")
            raise

        is_task_goal_fulfilled: bool = True
        result_conversation_history = copy.copy(conversation_history_original)
        if not llm_goal_fulfillment_auditor_result.is_task_goal_fulfilled:
            is_task_goal_fulfilled = False
            await _report_subllm_progress(
                app_context=app_context,
                job_id=job_id,
                llm_goal_fulfillment_auditor_activations_by_subharness=1,
            )
            result_content: str
            if llm_goal_fulfillment_auditor_result.llm_agent_instructions:
                result_content = f"LLM task goal fulfillment auditor activated: the LLM was detected to have stopped prematurely without completing its assigned task. As a result of activating the fulfillment auditor, the chat history has been compacted and restructured to save space. Continue working based on the data received from the history compaction system and on my guidance. Guidance provided to the LLM to help it achieve the goals of the assigned task: {llm_goal_fulfillment_auditor_result.llm_agent_instructions}"
            else:
                result_content = f"LLM task goal fulfillment auditor activated: the LLM was detected to have stopped prematurely without completing its assigned task. As a result of activating the fulfillment auditor, the chat history has been compacted and restructured to save space. Continue working based on the data received from the history compaction system."

            compressed_history: list[dict[str, Any]] = await self._compress_conversation_history(
                app_context=app_context,
                job_id=job_id,
                library=library,
                openai_compatible=self._config,
                original_prompt=original_prompt,
                conversation_history=result_conversation_history,
            )

            compressed_history.append(
                {
                    "role": app_context.config.openai_compatible.llm_guidance_role,
                    "content": result_content,
                }
            )
            result_conversation_history = compressed_history

        return is_task_goal_fulfilled, result_conversation_history

    async def _create_completion(
        self,
        *,
        app_context: AppContext,
        job_id: str | None = None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: Optional[Dict] = None,
        raise_on_length_exceeded: bool = True,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            # "max_tokens": self._config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        kwargs["reasoning_effort"] = self._config.reasoning_effort or "none"
        if not self._config.reasoning_allowed:
            kwargs["extra_body"] = {
                "enable_thinking": False
            }

        if response_format is not None:
            kwargs["response_format"] = response_format

        if app_context.config.openai_compatible.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {
                "include_usage": True,
            }

        try:
            logger.info(f"OpenAI compatible request; messages num: {len(messages)}, tools num: {len(tools) if tools else 0}, response_format: {response_format is not None}, raise_on_length_exceeded: {raise_on_length_exceeded}")
            logger_kwargs = copy.deepcopy(kwargs)
            if not app_context.config.logger.include_tools:
                logger_kwargs.pop("tools", None)

            logger.debug(f"OpenAI compatible request: \n{gmsodv(logger_kwargs, shift_num=1)}")
            await _report_subllm_progress(
                app_context=app_context,
                job_id=job_id,
                input_bytes_used_by_subllm=_completion_input_bytes(kwargs)
            )
            result: ChatCompletion
            with RT() as rt:
                if app_context.config.openai_compatible.stream:
                    temp_result: AsyncStream[ChatCompletionChunk]
                else:
                    temp_result: ChatCompletion
                
                temp_result = await self._completion_create(**kwargs)
                if app_context.config.openai_compatible.stream:
                    state: ChatCompletionStreamState = ChatCompletionStreamState()
                    stream: AsyncStream[ChatCompletionChunk] = temp_result
                    try:
                        chunk_history: List[ChatCompletionChunk] = list()
                        chunk: ChatCompletionChunk
                        async for chunk in stream:
                            await _report_subllm_progress(
                                app_context=app_context,
                                job_id=job_id,
                                output_bytes_generated_by_subllm=_completion_output_bytes(chunk),
                            )
                            chunk_history.append(chunk)

                            # logger.debug(f"<<ASYNC_STREAM.CHUNK>>. snapshot: {gmsodv(chunk)}")
                            state.handle_chunk(chunk)

                            snapshot: ParsedChatCompletionSnapshot = state.current_completion_snapshot
                            # logger.debug(f"<<ASYNC_STREAM>>. snapshot: {gmsodv(snapshot)}")

                            # # TODO: implement "_is_generation_stuck" method
                            # if self._is_generation_stuck(snapshot):
                            #     await stream.response.aclose()
                            #     return snapshot
                    except Exception:
                        await stream.response.aclose()
                        logger.debug(f"<<ASYNC_STREAM.Exception>>.\n\n# Request:\n\n{gmsodv(messages)}\n\n# List of chunks:\n\n{gmsodv(chunk_history)}")
                        # await _report_subllm_progress(
                        #     app_context=app_context,
                        #     job_id=job_id,
                        #     output_bytes_generated_by_subllm=_completion_output_bytes(result),
                        # )
                        raise

                    result = state.get_final_completion()
                    # logger.debug(f"<<ASYNC_STREAM>>. final_completion: {gmsodv(result)}")
                else:
                    result = temp_result

            chat_completion_token_usage: ChatCompletionTokenUsage = gather_chat_completion_token_usage(result)
            output_tokens_per_second: float = chat_completion_token_usage.output_tokens / rt()
            logger.info(f"OpenAI compatible request completed. Metrics.\nrun_time: {rt()};\ntokens_per_second: {output_tokens_per_second:.2f};\ninput_tokens: {chat_completion_token_usage.input_tokens};\ncached_input_tokens: {chat_completion_token_usage.cached_input_tokens};\noutput_tokens: {chat_completion_token_usage.output_tokens};\nchoices num: {len(result.choices)}.")
            await _report_subllm_progress(
                app_context=app_context,
                job_id=job_id,
                # output_bytes_generated_by_subllm=_completion_output_bytes(result),
                input_tokens_used_by_subllm=chat_completion_token_usage.input_tokens,
                cached_input_tokens_used_by_subllm=chat_completion_token_usage.cached_input_tokens,
                output_tokens_generated_by_subllm=chat_completion_token_usage.output_tokens,
            )
            if raise_on_length_exceeded:
                last_choice: Choice = result.choices[-1]
                if "length" == last_choice.finish_reason:
                    logger.error(f"OpenAI compatible request exceeded the model's context window (max_tokens).")
                    logger.debug(f"OpenAI compatible request exceeded the model's context window (max_tokens).:\n{gmsodv(result, shift_num=1)}")
                    raise OpenAICompatibleContextOverflowError(
                        "OpenAI compatible request exceeded the model's context window (max_tokens)."
                    )

            logger.info(f"OpenAI compatible request completed successfully; choices num: {len(result.choices)}")
            # logger.debug(f"OpenAI compatible response class owner info: {entity_owning_module_info_and_owning_path(result)}")
            # for choice in result.choices:
            #     logger.debug(f"OpenAI compatible response.choice class owner info: {entity_owning_module_info_and_owning_path(choice)}")
            #     logger.debug(f"OpenAI compatible response.choice.message class owner info: {entity_owning_module_info_and_owning_path(choice.message)}")

            logger.debug(f"OpenAI compatible request completed successfully:\n{gmsodv(result, shift_num=1)}")
            return result
        except APITimeoutError as exc:
            logger.exception(f"OpenAI compatible request timed out")
            raise OpenAICompatibleTimeoutError("OpenAI compatible request timed out") from exc
        except BadRequestError as exc:
            logger.exception(f"OpenAI compatible request failed with BadRequestError")
            if True or _is_context_overflow_error(exc):
                logger.error(f"_create_completion->BadRequestError: \n{kwargs}")
                raise OpenAICompatibleContextOverflowError(
                    _context_overflow_error_message()
                ) from exc
            raise
        except APIError as exc:
            if isinstance(exc.body, dict):
                error = exc.body

                error_type = error.get("type")
                error_code = error.get("code")

                if (400 == error_code) and ("exceed_context_size_error" == error_type):
                    logger.error(f"_create_completion->APIError(400): \n{kwargs}")
                    raise OpenAICompatibleContextOverflowError(
                        _context_overflow_error_message()
                    ) from exc

            raise
        except NotFoundError as exc:
            logger.exception(f"OpenAI compatible model is missing or unavailable: {self._config.model}")
            raise MissingModelError(f"OpenAI compatible model is missing or unavailable: {self._config.model}") from exc
        except APIConnectionError as exc:
            logger.exception(f"Could not connect to the OpenAI compatible server")
            raise OpenAICompatibleConnectionError("Could not connect to the OpenAI compatible server") from exc
        except:
            logger.exception(f"OpenAI compatible request failed with an unexpected error")
            raise

    def _enabled_tool_definitions(
        self,
        app_context: AppContext,
        enabled_tool_names_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self._config.tool_backend != "openai_tools":
            return list()

        result = app_context.allowed_tool_definitions()
        if enabled_tool_names_filter is None:
            return result
        else:
            return [tool for tool in result if tool["function"]["name"] in enabled_tool_names_filter]

    def _enabled_tool_names(
        self,
        app_context: AppContext,
    ) -> list[str]:
        if self._config.tool_backend != "openai_tools":
            return list()

        return app_context.allowed_tool_names()

    def _is_denied_tool_call(
        self,
        tool_name: str
    ) -> bool:
        return tool_name in self._config.all_denied_tools()


def _should_compress_context(*, response: Any, max_tokens: int, threshold: float) -> bool:
    prompt_tokens = _response_prompt_tokens(response)
    if prompt_tokens is None:
        return False
    if max_tokens <= 0:
        return False

    return prompt_tokens / max_tokens >= threshold


def _response_prompt_tokens(response: Any) -> int | None:
    usage = _get(response, "usage")
    prompt_tokens = _get(usage, "total_tokens")
    if prompt_tokens is None:
        return None
    try:
        return int(prompt_tokens)
    except (TypeError, ValueError):
        return None


def _completion_input_bytes(kwargs: Mapping[str, Any]) -> int:
    prompt_payload = {
        key: kwargs[key]
        for key in ("messages", "tools", "response_format")
        if key in kwargs
    }
    return _json_utf8_size(prompt_payload)


def _completion_output_bytes(response: Any) -> int:
    output_size = 0
    # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 0:")
    for choice in list(_get(response, "choices") or []):
        # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 1:")
        message = _get(choice, "message")
        if message is None:
            # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 2:")
            delta: ChoiceDelta = _get(choice, "delta")
            message = delta
            if message is None:
                # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 3:")
                continue

        content = _get(message, "content")
        if content:
            # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 4:")
            output_size += len(str(content).encode("utf-8"))

        content = _get(message, "reasoning")
        if content:
            # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 5:")
            output_size += len(str(content).encode("utf-8"))
        else:
            # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 6:")
            content = _get(message, "reasoning_content")
            if content:
                # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 7:")
                output_size += len(str(content).encode("utf-8"))

        tool_calls = _get(message, "tool_calls")
        if tool_calls:
            # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 8:")
            output_size += _json_utf8_size(tool_calls)

        function_call = _get(message, "function_call")
        if function_call:
            # logger.debug(f"<<ASYNC_STREAM.CHUNK._completion_output_bytes>>. 9:")
            output_size += _json_utf8_size(function_call)

    return output_size


def _json_utf8_size(value: Any) -> int:
    try:
        text = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)

    return len(text.encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# def _sanitize_final_response(content: str) -> str:
#     sanitized = content.strip()
#     sanitized = re.sub(
#         r"(?is)<(analysis|think|tool_call|tool_calls|tool_result|tool_results|execution_trace|internal_reasoning)>.*?</\1>",
#         "",
#         sanitized,
#     )
#     sanitized = re.sub(
#         r"(?ms)^(`{3,})[ \t]*(analysis|tool_call|tool_calls|tool_result|tool_results|execution_trace|internal_reasoning)[^\n]*\n.*?^\1[ \t]*$",
#         "",
#         sanitized,
#     )

#     internal_prefixes = (
#         "internal reasoning:",
#         "tool execution trace:",
#         "tool invocation log:",
#         "tool invocation logs:",
#         "analysis artifact:",
#         "analysis artifacts:",
#     )
#     lines = [
#         line
#         for line in sanitized.splitlines()
#         if not line.strip().lower().startswith(internal_prefixes)
#     ]
#     return "\n".join(lines).strip()


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


_CONTEXT_OVERFLOW_ERROR_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "input_too_long",
    "prompt_too_long",
    "too_many_tokens",
    "tokens_limit_exceeded",
}

_CONTEXT_OVERFLOW_MESSAGE_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context limit",
    "context size",
    "context window",
    "exceeds the context",
    "exceeded the context",
    "exceeds the available context size",
    "available context size",
    "input is too long",
    "input too long",
    "input length",
    "prompt is too long",
    "prompt too long",
    "prompt exceeds",
    "too many tokens",
    "tokens exceed",
    "token limit",
)


def _context_overflow_error_message() -> str:
    return (
        "The OpenAI-compatible model used by this tool could not process the request because "
        "the accumulated prompt exceeded its context window. The client LLM must retry with a "
        "narrower request: analyze one entity/action at a time, provide specific file paths or "
        "symbols, split broad tasks into multiple tool calls, and avoid asking the tool to load "
        "large unrelated files. If this happened after a tool call, the tool result made the next "
        "model request too large."
    )


def _is_context_overflow_error(exc: BadRequestError) -> bool:
    values = [str(exc)]
    for attr in ("code", "type", "param", "body"):
        values.extend(_flatten_error_values(_get(exc, attr)))

    normalized_values = [str(value).strip().lower() for value in values if value is not None]
    if any(value in _CONTEXT_OVERFLOW_ERROR_CODES for value in normalized_values):
        return True

    return any(
        marker in value
        for value in normalized_values
        for marker in _CONTEXT_OVERFLOW_MESSAGE_MARKERS
    )


def _flatten_error_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[Any] = []
        for key, item in value.items():
            values.append(key)
            values.extend(_flatten_error_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_error_values(item))
        return values
    return [value]


def _extract_assistant_message(response: ChatCompletion) -> ChatCompletionMessage:
    choices: List[Choice] = response.choices
    if not choices:
        raise MalformedOpenAICompatibleResponse("OpenAI compatible response had empty choices")

    message: ChatCompletionMessage = choices[0].message
    return message


def _extract_final_content(response: ChatCompletion) -> str:
    message: ChatCompletionMessage = _extract_assistant_message(response)
    content = _message_content(message).strip()
    return content


def _message_content(message: ChatCompletionMessage) -> str:
    content = message.content
    return content or str()


def _message_reasoning(message: ChatCompletionMessage) -> str:
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return "reasoning", reasoning

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return "reasoning_content", reasoning_content

    return None, None


def _message_tool_calls(message: ChatCompletionMessage) -> list[ChatCompletionMessageToolCallUnion]:
    return list(message.tool_calls or list())


def _message_tool_calls__from__dict(message: dict[str, Any]) -> list[dict[str, Any]]:
    return list(message.get("tool_calls") or list())


def _assistant_tool_call_message(message: ChatCompletionMessage, tool_calls: list[ChatCompletionMessageToolCallUnion]) -> dict[str, Any]:
    reasoning_field, reasoning_content = _message_reasoning(message)
    result = {
        "role": "assistant",
        "content": _message_content(message) or None,
        "tool_calls": [_serialize_tool_call(tool_call) for tool_call in tool_calls],
    }
    if reasoning_field is not None:
        result[reasoning_field] = reasoning_content
    
    return result


def _serialize_tool_call(tool_call: ChatCompletionMessageToolCallUnion) -> dict[str, Any]:
    function = tool_call.function
    return {
        "id": _tool_call_id(tool_call),
        "type": tool_call.type,
        "function": {
            "name": function.name,
            "arguments": function.arguments,
        },
    }


def _parse_tool_call(tool_call: ChatCompletionMessageToolCallUnion) -> tuple[str, dict[str, Any]]:
    name = _tool_call_name(tool_call)
    return name, _parse_tool_call_arguments(tool_call, name)


def _tool_call_name(tool_call: ChatCompletionMessageToolCallUnion) -> Optional[str]:
    function = tool_call.function
    name = function.name
    if not name:
        return None

    return str(name)


def _tool_call_name__from__dict(tool_call: dict[str, Any]) -> Optional[str]:
    function = tool_call.get("function") or dict()
    name = function.get("name")
    if not name:
        return None

    return str(name)


def _parse_tool_call_arguments(tool_call: ChatCompletionMessageToolCallUnion, name: str) -> dict[str, Any]:
    function = tool_call.function
    raw_arguments = function.arguments
    arguments: dict[str, Any]
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        raise MalformedOpenAICompatibleResponse(
            f"OpenAI compatible tool call for {name} had malformed JSON arguments: {raw_arguments}"
        ) from exc
    if not isinstance(arguments, dict):
        raise MalformedOpenAICompatibleResponse(
            f"OpenAI compatible tool call for {name} arguments must be a JSON object: {raw_arguments}"
        )

    return arguments


def _tool_call_id(tool_call: ChatCompletionMessageToolCallUnion) -> str:
    call_id = tool_call.id
    if not call_id:
        raise MalformedOpenAICompatibleResponse("OpenAI compatible tool call was missing id")

    return str(call_id)


def _tool_call_id__from__dict(tool_call: dict[str, Any]) -> str:
    call_id = tool_call.get("id")
    if not call_id:
        raise MalformedOpenAICompatibleResponse("OpenAI compatible tool call was missing id")

    return str(call_id)


def _enabled_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool["function"]["name"] for tool in tools}
