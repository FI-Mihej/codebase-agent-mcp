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
from collections.abc import Awaitable, Callable
from typing import Any, Optional, List, Dict

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, NotFoundError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCallUnion

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
)
from codebase_agent.types import (
    PluginABC,
)
from codebase_agent.app_context import AppContext
from codebase_agent.prompts import build_prompt_for__context_compression
import copy
import inspect
import logging


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


def is_async(entity) -> bool:
    """Check whether an entity participates in Python's async protocols.

    Args:
        entity: Object, coroutine, generator, or callable to inspect.

    Returns:
        bool: ``True`` when the entity is async-related or awaitable.
    """
    # print(inspect.iscoroutine(entity), inspect.isgenerator(entity), inspect.iscoroutinefunction(entity), inspect.isgeneratorfunction(entity), inspect.isasyncgen(entity), inspect.isasyncgenfunction(entity), inspect.isawaitable(entity))
    return inspect.iscoroutine(entity) or inspect.isgenerator(entity) or inspect.iscoroutinefunction(entity) or inspect.isgeneratorfunction(entity) or inspect.isasyncgen(entity) or inspect.isasyncgenfunction(entity) or inspect.isawaitable(entity)


def is_callable(entity) -> bool:
    """Return whether the provided entity is callable.

    Args:
        entity: Object to inspect.

    Returns:
        bool: Result of Python's built-in :func:`callable`.
    """
    return callable(entity)


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

    async def consult(
        self,
        *,
        app_context: AppContext,
        library: LibraryConfig,
        client_request_type: ClientRequestType,
        system_prompt: str,
        query: str,
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
                result = await self._compress_conversation_history(
                    app_context=app_context,
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

        tools = self._enabled_tool_definitions(app_context=app_context)
        tool_rounds = 0
        last_assistant_content = ""
        assistant_message: ChatCompletionMessage
        tool_calls: List[ChatCompletionMessageToolCallUnion]
        last_tool_calls: List[ChatCompletionMessageToolCallUnion]
        last_tool_call: Optional[ChatCompletionMessageToolCallUnion]
        last_tool_name: Optional[str]
        while True:
            overflow_error_occurred = False
            content_filter_error_occurred = False
            try:
                response = await self._create_completion(
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

                    assistant_message = message_before_last_tool
                    last_tool_calls = _message_tool_calls(assistant_message)
                    last_tool_call = last_tool_calls[-1] if last_tool_calls else None
                    last_tool_name = _tool_call_name(last_tool_call) if last_tool_call else None
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(last_tool_call),
                            "name": last_tool_name,
                            "content": f"Error: Your most recent request to the `{last_tool_name}` tool was blocked by the content filter. Retry with a request that is expected to produce a tool response allowed by the content filter.",
                        }
                    )
                elif "assistant" == last_message.get("role"):
                    assistant_message = last_message
                    last_tool_calls = _message_tool_calls(assistant_message)
                    last_tool_call = last_tool_calls[-1] if last_tool_calls else None
                    last_tool_name = _tool_call_name(last_tool_call) if last_tool_call else None
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(last_tool_call),
                            "name": last_tool_name,
                            "content": f"Error: Your most recent message was blocked by the content filter. Retry with a message that is expected to be allowed by the content filter.",
                        }
                    )
                    if last_tool_call:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _tool_call_id(last_tool_call),
                                "name": last_tool_name,
                                "content": f"Error: your most recent request was blocked by the content filter. Retry with a request that is expected to be allowed by the content filter.",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "developer",
                                "content": f"Error: Your most recent message was blocked by the content filter. Retry with a message that is expected to be allowed by the content filter.",
                            }
                        )
                else:
                    raise RuntimeError(f"Unexpected message role in content filter handling: {last_message.get('role')}")
            
            if overflow_error_occurred:
                last_message = messages[-1]
                if "user" == last_message.get("role"):
                    raise DirectErrorResponseForClientLLM("The request could not be completed because the accumulated prompt exceeded the model's context window. Retry with a narrower request.")
                elif "tool" == last_message.get("role"):
                    message_before_last_tool = messages[-2] if len(messages) >= 2 else None
                    if "assistant" != message_before_last_tool.get("role"):
                        raise RuntimeError("Unexpected message sequence: last message was tool, but the message before that was not assistant")

                    assistant_message = message_before_last_tool
                    last_tool_calls = _message_tool_calls(assistant_message)
                    last_tool_call = last_tool_calls[-1] if last_tool_calls else None
                    last_tool_name = _tool_call_name(last_tool_call) if last_tool_call else None
                    try:
                        compressed_history = await self._compress_conversation_history(
                            app_context=app_context,
                            library=library,
                            openai_compatible=self._config,
                            original_prompt=query,
                            conversation_history=messages[1:],
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
                    messages.append(message_before_last_tool)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(last_tool_call),
                            "name": last_tool_name,
                            "content": "Error: the accumulated prompt, after adding the tool's response, exceeded the model's context window. Retry with a request that is expected to produce a narrower response from the tool.",
                        }
                    )
                elif "assistant" == last_message.get("role"):
                    assistant_message = last_message
                    try:
                        compressed_history = await self._compress_conversation_history(
                            app_context=app_context,
                            library=library,
                            openai_compatible=self._config,
                            original_prompt=query,
                            conversation_history=messages[1:],
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
                    last_tool_calls = _message_tool_calls(assistant_message)
                    last_tool_call = last_tool_calls[-1] if last_tool_calls else None
                    last_tool_name = _tool_call_name(last_tool_call) if last_tool_call else None
                    if last_tool_call:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _tool_call_id(last_tool_call),
                                "name": last_tool_name,
                                "content": f"Error: your most recent request to the `{last_tool_name}` tool caused the accumulated prompt to exceed the model's context window. As a result, the conversation history was compacted. Continue your work by making requests that are expected to produce narrower responses from the tools.",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "developer",
                                "content": "Error: The accumulated prompt generated by your most recent message exceeded the model's context window. As a result, the conversation history was compacted. Retry with a message that is expected to use fewer tokens.",
                            }
                        )
                elif "developer" == last_message.get("role"):
                    raise DirectErrorResponseForClientLLM('Error 2: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')
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
                        need_to_stop = True
                else:
                    need_to_stop = True
            
            if -1 < self._config.max_tool_rounds:
                if tool_rounds >= self._config.max_tool_rounds:
                    need_to_stop = True

            if need_to_stop:
                try:
                    result = await self._compress_conversation_history(
                        app_context=app_context,
                        library=library,
                        openai_compatible=self._config,
                        original_prompt=query,
                        conversation_history=messages[1:],
                    )
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
                        return str(result)
                except OpenAICompatibleContextOverflowError:
                    raise DirectErrorResponseForClientLLM('Error 3: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

            if _should_compress_context(
                response=response,
                max_tokens=app_context.config.openai_compatible.max_tokens,
                threshold=app_context.config.openai_compatible.context_compression_threshold,
            ):
                last_message = messages[-1]
                try:
                    compressed_history = await self._compress_conversation_history(
                        app_context=app_context,
                        library=library,
                        openai_compatible=self._config,
                        original_prompt=query,
                        conversation_history=messages[1:],
                    )
                except OpenAICompatibleContextOverflowError:
                    raise DirectErrorResponseForClientLLM('Error 4: The accumulated prompt exceeded the model\'s context window. It is not possible to compact the chat history because there is not enough space available in my context window to perform this operation. You must inform the operator that they need to adjust my ("CodebaseAgent-MCP") configuration by reducing the value of "context_compression_threshold" and, if necessary, increasing the value of "max_tokens" in the "codebase_agent.config.json" file.')

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": compressed_history,
                    },
                    {
                        "role": "developer",
                        "content": "Conversation history was compacted. Continue your work.",
                    },
                    last_message,
                ]

            assistant_message = _extract_assistant_message(response)
            tool_calls = _message_tool_calls(assistant_message)

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
                    arguments = _parse_tool_call_arguments(tool_call, tool_name)
                    plugin: PluginABC = app_context.plugin_by_tool_name(tool_name)
                    if plugin is None:
                        result: ToolResult = {
                            "ok": False,
                            "error": {
                                "code": "tool_unreachable",
                                "message": f"Tool is unreachable: {tool_name}. Use other available tools.",
                            },
                        }

                    plugin_executor = plugin.executor()

                    if is_async(plugin_executor):
                        result = await plugin_executor(
                            library.name,
                            client_request_type,
                            tool_name,
                            arguments,
                        )
                    else:
                        result = plugin_executor(
                            library.name,
                            client_request_type,
                            tool_name,
                            arguments,
                        )
                
                content: str = json.dumps(result, ensure_ascii=False)
                if self._config.max_tool_response_bytes:
                    if len(content.encode("utf-8")) > self._config.max_tool_response_bytes:
                        result = {
                            "ok": False,
                            "error": {
                                "code": "tool_response_too_large",
                                "message": f"Tool response exceeded the maximum allowed size of {self._config.max_tool_response_bytes} bytes. Refine and narrow your request to produce a shorter response.",
                            },
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
        system_prompt: str,
        query: str,
    ) -> str:
        """Run a no-tool classification request without persistence side effects."""

        response: ChatCompletion = await self._create_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        )
        return _extract_final_content(response)

    async def _compress_conversation_history(
        self,
        *,
        app_context: AppContext,
        library: LibraryConfig,
        openai_compatible: OpenAICompatibleConfig,
        original_prompt: str,
        conversation_history: list[dict[str, Any]],
    ) -> str:
        new_conversation_history: list[dict[str, Any]] = list()
        for message in conversation_history:
            if message.get("role") not in {"developer", "system"}:
                new_conversation_history.append(message)
        
        conversation_history = new_conversation_history
        response: ChatCompletion = await self._create_completion(
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
                    ),
                },
            ]
        )
        return _extract_final_content(response)

    async def _create_completion(
        self, 
        *, 
        messages: list[dict[str, Any]], 
        tools: list[dict[str, Any]] | None = None,
        response_format: Optional[Dict] = None,
        raise_on_length_exceeded: bool = True,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
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

        try:
            logger.info(f"OpenAI compatible request: {kwargs}")
            result: ChatCompletion = await self._completion_create(**kwargs)
            if raise_on_length_exceeded:
                last_choice: Choice = result.choices[-1]
                if "length" == last_choice.finish_reason:
                    logger.error(f"OpenAI compatible request exceeded the model's context window (max_tokens).:\n{result}\n{kwargs}")
                    raise OpenAICompatibleContextOverflowError(
                        "OpenAI compatible request exceeded the model's context window (max_tokens)."
                    )
            
            return result
        except APITimeoutError as exc:
            raise OpenAICompatibleTimeoutError("OpenAI compatible request timed out") from exc
        except BadRequestError as exc:
            if True or _is_context_overflow_error(exc):
                logger.error(f"_create_completion->BadRequestError: \n{kwargs}")
                raise OpenAICompatibleContextOverflowError(
                    _context_overflow_error_message()
                ) from exc
            raise
        except NotFoundError as exc:
            raise MissingModelError(f"OpenAI compatible model is missing or unavailable: {self._config.model}") from exc
        except APIConnectionError as exc:
            raise OpenAICompatibleConnectionError("Could not connect to the OpenAI compatible server") from exc

    def _enabled_tool_definitions(
        self,
        app_context: AppContext,
    ) -> list[dict[str, Any]]:
        if self._config.tool_backend != "openai_tools":
            return list()
        
        return app_context.allowed_tool_definitions()

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
    prompt_tokens = _get(usage, "prompt_tokens")
    if prompt_tokens is None:
        return None
    try:
        return int(prompt_tokens)
    except (TypeError, ValueError):
        return None


def _sanitize_final_response(content: str) -> str:
    sanitized = content.strip()
    sanitized = re.sub(
        r"(?is)<(analysis|think|tool_call|tool_calls|tool_result|tool_results|execution_trace|internal_reasoning)>.*?</\1>",
        "",
        sanitized,
    )
    sanitized = re.sub(
        r"(?ms)^(`{3,})[ \t]*(analysis|tool_call|tool_calls|tool_result|tool_results|execution_trace|internal_reasoning)[^\n]*\n.*?^\1[ \t]*$",
        "",
        sanitized,
    )

    internal_prefixes = (
        "internal reasoning:",
        "tool execution trace:",
        "tool invocation log:",
        "tool invocation logs:",
        "analysis artifact:",
        "analysis artifacts:",
    )
    lines = [
        line
        for line in sanitized.splitlines()
        if not line.strip().lower().startswith(internal_prefixes)
    ]
    return "\n".join(lines).strip()


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


def _message_tool_calls(message: ChatCompletionMessage) -> list[ChatCompletionMessageToolCallUnion]:
    return list(message.tool_calls or list())


def _assistant_tool_call_message(message: ChatCompletionMessage, tool_calls: list[ChatCompletionMessageToolCallUnion]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": _message_content(message) or None,
        "tool_calls": [_serialize_tool_call(tool_call) for tool_call in tool_calls],
    }


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


def _enabled_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool["function"]["name"] for tool in tools}
