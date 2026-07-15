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


"""FastMCP stdio server entry point."""


from __future__ import annotations

import json
from pathlib import PurePath, PureWindowsPath
from typing import Any, Annotated
from pydantic import Field

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.shared.context import RequestContext, LifespanContextT

from codebase_agent import app_context
from codebase_agent.config import (
    AgentConfig, 
    load_config, 
    validate_library_path, 
    ensure_config_exists, 
    get_app_data_dir_path,
)
from codebase_agent.jobs import CodebaseAnalysisJobManager
from codebase_agent.io_debug import IODebugLogger, logged_io_streams
from codebase_agent.openai_compatible_client import OpenAICompatibleClient
from codebase_agent.plugin_factory import build_plugin_bundle
from codebase_agent.prompts import (
    QUERY_GRANULARITY_INSTRUCTION,
    build_prompt_for__codebase_start_job_related_files_search,
    build_prompt_for__codebase_start_job_analysis,
    build_prompt_for__query_granularity_validation,
)
from codebase_agent.types import (
    CodebaseAgentError,
    EmptyQueryError,
    MalformedOpenAICompatibleResponse,
    QueryGranularityError,
    UnknownLibraryError,
    DirectErrorResponseForClientLLM,
)
from codebase_agent.types import ClientRequestType
from codebase_agent.app_context import AppContext
from cengal.file_system.directory_manager import dir_exists

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
import re
from pathlib import Path

from codebase_agent.logging_config import setup_logging
setup_logging()
import logging


logger = logging.getLogger(__name__)
logger.info("MCP-server started")


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""

    async with AsyncExitStack() as exit_stack:
        config = load_config()
        analysis_jobs = CodebaseAnalysisJobManager()
        plugin_bundle = await build_plugin_bundle(config, exit_stack)
        yield AppContext(
            config=config,
            analysis_jobs=analysis_jobs,
            plugins=plugin_bundle.plugins,
            qdrant_client=plugin_bundle.qdrant_client,
            local_fs_tools=plugin_bundle.local_fs_tools,
            text_file_tools=plugin_bundle.text_file_tools,
        )


mcp = FastMCP("codebase-agent", lifespan=app_lifespan)


def codebase_list_libraries_from_config(config: AgentConfig) -> dict[str, list[str]]:
    """Return configured library names in configuration order."""

    return {"libraries": [library.name for library in config.allowed_libraries()]}


async def codebase_start_job_analysis_with_config(
    *,
    app_context: AppContext,
    config: AgentConfig,
    library_name: str,
    query: str,
    client: OpenAICompatibleClient | None = None,
) -> str:
    """Validate a request and delegate one fresh consultation to OpenAI compatible."""

    normalized_query = query.strip()
    if not normalized_query:
        raise EmptyQueryError("query must not be empty")

    library = config.get_allowed_library(library_name)
    if library is None:
        raise UnknownLibraryError(f"Unknown library name: {library_name}")

    validate_library_path(library.path, library.name)
    system_prompt = build_prompt_for__codebase_start_job_analysis(
        app_context=app_context,
        library=library,
        openai_compatible=config.openai_compatible,
        user_query=normalized_query,
    )
    openai_compatible_client = client or OpenAICompatibleClient(config.openai_compatible)
    return await openai_compatible_client.consult(
        app_context=app_context,
        library=library,
        client_request_type=ClientRequestType.analysis,
        system_prompt=system_prompt,
        query=normalized_query,
    )


async def codebase_start_job_related_files_search_with_config(
    *,
    app_context: AppContext,
    config: AgentConfig,
    library_name: str,
    query: str,
    client: OpenAICompatibleClient | None = None,
) -> dict[str, Any]:
    """Find ranked file candidates related to a task using one fresh OpenAI compatible chat."""

    normalized_query = query.strip()
    if not normalized_query:
        raise EmptyQueryError("query must not be empty")

    library = config.get_allowed_library(library_name)
    if library is None:
        raise UnknownLibraryError(f"Unknown library name: {library_name}")

    validate_library_path(library.path, library.name)
    system_prompt = build_prompt_for__codebase_start_job_related_files_search(
        app_context=app_context,
        library=library,
        openai_compatible=config.openai_compatible,
        user_query=normalized_query,
    )
    openai_compatible_client = client or OpenAICompatibleClient(config.openai_compatible)
    raw_response = await openai_compatible_client.consult(
        app_context=app_context,
        library=library,
        client_request_type=ClientRequestType.find_files,
        system_prompt=system_prompt,
        query=normalized_query,
    )
    return _parse_codebase_start_job_related_files_search_response(raw_response)


async def codebase_start_job_related_files_search_job_with_config(
    *,
    app_context: AppContext,
    config: AgentConfig,
    library_name: str,
    query: str,
    client: OpenAICompatibleClient | None = None,
) -> str:
    """Run related-file discovery and serialize the structured result for job storage."""

    normalized_query = _validate_start_job_inputs(config=config, library_name=library_name, query=query)
    openai_compatible_client = client or OpenAICompatibleClient(config.openai_compatible)
    result: dict[str, Any]
    try:
        await _validate_query_granularity(
            client=openai_compatible_client,
            config=config,
            tool_name="codebase_start_job_related_files_search",
            query=normalized_query,
        )
        result = await codebase_start_job_related_files_search_with_config(
            app_context=app_context,
            config=config,
            library_name=library_name,
            query=normalized_query,
            client=openai_compatible_client,
        )
    except DirectErrorResponseForClientLLM as ex:
        message = ex.message
        result = {
            "error": message,
        }
    
    return json.dumps(result, ensure_ascii=False)


async def codebase_start_job_analysis_job_with_config(
    *,
    app_context: AppContext,
    config: AgentConfig,
    library_name: str,
    query: str,
    client: OpenAICompatibleClient | None = None,
) -> str:
    """Run analysis after validating that the request is granular enough."""

    normalized_query = _validate_start_job_inputs(config=config, library_name=library_name, query=query)
    openai_compatible_client = client or OpenAICompatibleClient(config.openai_compatible)
    result: dict[str, Any]
    try:
        await _validate_query_granularity(
            client=openai_compatible_client,
            config=config,
            tool_name="codebase_start_job_analysis",
            query=normalized_query,
        )
        result = await codebase_start_job_analysis_with_config(
            app_context=app_context,
            config=config,
            library_name=library_name,
            query=normalized_query,
            client=openai_compatible_client,
        )
    except DirectErrorResponseForClientLLM as ex:
        message = ex.message
        result = json.dumps({
            "error": message,
        })
    
    return result


def _validate_start_job_inputs(*, config: AgentConfig, library_name: str, query: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise EmptyQueryError("query must not be empty")

    library = config.get_allowed_library(library_name)
    if library is None:
        raise UnknownLibraryError(f"Unknown library name: {library_name}")

    validate_library_path(library.path, library.name)
    return normalized_query


@mcp.tool()
def codebase_list_libraries() -> dict[str, list[str]]:
    """Return the public names of local libraries/codebases available for analysis."""

    try:
        return codebase_list_libraries_from_config(load_config())
    except CodebaseAgentError as exc:
        _raise_mcp_error(exc)


@mcp.tool()
async def codebase_start_job_related_files_search(
    library_name: Annotated[
        str,
        Field(description="Public library name returned by list_libraries."),
    ],
    query: Annotated[
        str,
        Field(description=(
            "Focused, detailed request to find files related to one entity/action. "
            "Include all relevant context (purpose, paths, symbols, imports, etc.). "
            "One topic or one context per request! Instead \"Find X, Y, Z, etc.\" you MUST: \"Find X.\", wait result, \"Find Y.\", wait for result, etc!"
        )),
    ],
    ctx: Context[ServerSession, AppContext],
) -> dict[str, Any]:
    """**Start async related-file discovery. Find files relevant to single entity/action before `codebase_start_job_analysis`. Provide all relevant context (files, paths, symbols, imports, etc.). One topic or one context per request → wait for the result → send the next request! Instead \"Find X, Y, Z, etc.\" you MUST: \"Find X.\", wait result, \"Find Y.\", wait for result, etc!**"""

    app_context: AppContext = ctx.request_context.lifespan_context
    try:
        result = await app_context.analysis_jobs.start_job(
            app_context=app_context,
            config=load_config(),
            library_name=library_name,
            query=query,
            runner=codebase_start_job_related_files_search_job_with_config,
        )
        result["message"] = (
            "codebase_start_job_related_files_search starts an asynchronous job. Use "
            "codebase_get_job_status and codebase_get_job_result with the returned job_id. "
            "The completed result is a JSON string containing files and notes."
        )
        return result
    except CodebaseAgentError as exc:
        _raise_mcp_error(exc)


@mcp.tool()
async def codebase_start_job_analysis(
    library_name: Annotated[
        str,
        Field(description="Public library name returned by list_libraries."),
    ],
    query: Annotated[
        str,
        Field(description=(
            "Focused, detailed request to analyze one entity/action. "
            "Include all relevant context (purpose, paths, symbols, imports, etc.). "
            "One topic or one context per request. Instead \"Find X, Y, Z, etc.\" you MUST: \"Find X.\", wait result, \"Find Y.\", wait for result, etc!"
        )),
    ],
    ctx: Context[ServerSession, AppContext],
) -> dict[str, Any]:
    """**Start async codebase analysis. Analyze single entity/action only. Return detailed analysis, recommendations, implementation guidance, usage examples. Prefer `codebase_start_job_related_files_search` first. Provide all relevant context (files, paths, symbols, imports, etc.). One topic or one context per request → wait for the result → send the next request! Instead \"Find X, Y, Z, etc.\" you MUST: \"Find X.\", wait result, \"Find Y.\", wait for result, etc!**"""

    app_context: AppContext = ctx.request_context.lifespan_context
    try:
        result = await app_context.analysis_jobs.start_job(
            app_context=app_context,
            config=load_config(),
            library_name=library_name,
            query=query,
            runner=codebase_start_job_analysis_job_with_config,
        )
        result["message"] = (
            "codebase_start_job_analysis starts an asynchronous job. Use "
            "codebase_get_job_status and codebase_get_job_result with the returned job_id."
        )
        return result
    except CodebaseAgentError as exc:
        _raise_mcp_error(exc)


@mcp.tool()
async def codebase_get_job_status(
    job_id: Annotated[
        str,
        Field(description="Job id returned by either `codebase_start_job_related_files_search` or `codebase_start_job_analysis`."),
    ],
    ctx: Context[ServerSession, AppContext],
) -> dict[str, Any]:
    """**Get async analysis job status. Check either `codebase_start_job_related_files_search` or `codebase_start_job_analysis` job progress. Poll until `success`/`failure`. Wait 50s/request. Do not assume failure before terminal status.**"""

    app_context: AppContext = ctx.request_context.lifespan_context
    try:
        return await app_context.analysis_jobs.get_status(job_id, load_config().jobs)
    except CodebaseAgentError as exc:
        _raise_mcp_error(exc)


@mcp.tool()
async def codebase_get_job_result(
    job_id: Annotated[
        str,
        Field(description="Job id returned by either `codebase_start_job_related_files_search` or `codebase_start_job_analysis`."),
    ],
    ctx: Context[ServerSession, AppContext],
) -> dict[str, Any]:
    """Return the final result, error, or latest partial output for an analysis job."""

    app_context: AppContext = ctx.request_context.lifespan_context
    try:
        return await app_context.analysis_jobs.get_result(job_id, load_config().jobs)
    except CodebaseAgentError as exc:
        _raise_mcp_error(exc)


@mcp.tool()
async def codebase_cancel_job(
    job_id: Annotated[
        str,
        Field(description="Job id returned by either `codebase_start_job_related_files_search` or `codebase_start_job_analysis`."),
    ],
    ctx: Context[ServerSession, AppContext],
) -> dict[str, Any]:
    """Request cancellation of a queued or running analysis job."""

    app_context: AppContext = ctx.request_context.lifespan_context
    try:
        return await app_context.analysis_jobs.cancel(job_id, load_config().jobs)
    except CodebaseAgentError as exc:
        _raise_mcp_error(exc)


async def _validate_query_granularity(
    *,
    client: OpenAICompatibleClient,
    config: AgentConfig,
    tool_name: str,
    query: str,
) -> None:
    """Ask the local model whether the start-tool request is appropriately scoped."""

    raw_response = await client.classify_without_tools(
        system_prompt=build_prompt_for__query_granularity_validation(
            tool_name=tool_name,
            openai_compatible=config.openai_compatible,
        ),
        query=query.strip(),
    )
    validation = _parse_query_granularity_validation_response(raw_response)
    if validation["valid"]:
        return

    reason = validation["reason"]
    raise QueryGranularityError(_query_granularity_error_message(tool_name=tool_name, reason=reason))


def _query_granularity_error_message(*, tool_name: str, reason: str) -> str:
    reason_text = reason.strip() or "The request contains more than one entity/action."
    return (
        f"{tool_name} rejected the request because it is too broad: {reason_text}\n\n"
        f"Client LLM instruction: {QUERY_GRANULARITY_INSTRUCTION}"
    )


def _parse_query_granularity_validation_response(raw_response: str) -> dict[str, Any]:
    """Parse the strict JSON query-granularity validation response."""

    payload = _strip_json_response_fence(raw_response)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MalformedOpenAICompatibleResponse(
            "query granularity validation expected OpenAI compatible to return strict JSON"
        ) from exc

    if not isinstance(parsed, dict):
        raise MalformedOpenAICompatibleResponse("query granularity validation response must be a JSON object")

    valid = parsed.get("valid")
    if not isinstance(valid, bool):
        raise MalformedOpenAICompatibleResponse("query granularity validation response must include boolean valid")

    reason = parsed.get("reason", "")
    if reason is not None and not isinstance(reason, str):
        raise MalformedOpenAICompatibleResponse("query granularity validation reason must be a string")

    return {"valid": valid, "reason": reason or ""}


def _raise_mcp_error(error: CodebaseAgentError) -> None:
    """Raise an MCP-friendly structured error when the SDK exposes that API."""

    message = f"{error.code}: {error}"
    try:
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

        invalid_param_codes = {"unknown_library", "empty_query", "invalid_library_path", "invalid_job_id"}
        code = INVALID_PARAMS if error.code in invalid_param_codes else INTERNAL_ERROR
        raise McpError(ErrorData(code=code, message=message))
    except ImportError:
        raise RuntimeError(message) from error


def _parse_codebase_start_job_related_files_search_response(raw_response: str) -> dict[str, Any]:
    """Parse the local model's strict JSON file-finder response."""

    payload = _strip_json_response_fence(raw_response)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return {"payload": payload, "files": list(), "notes": str(), "warning": "Related files search result is not strict JSON; returning raw payload."}

    if not isinstance(parsed, dict):
        raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search response must be a JSON object")

    files = parsed.get("files")
    if not isinstance(files, list):
        raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search response must include a files list")
    for file_entry in files:
        if not isinstance(file_entry, dict):
            raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search files must be JSON objects")
        path = file_entry.get("path")
        if not isinstance(path, str) or not path:
            raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search file entries must include path")
        parsed_path = PurePath(path)
        windows_path = PureWindowsPath(path)
        if (
            parsed_path.is_absolute()
            or windows_path.is_absolute()
            or ".." in parsed_path.parts
            or ".." in windows_path.parts
        ):
            raise MalformedOpenAICompatibleResponse(
                "codebase_start_job_related_files_search file paths must be relative to the library root"
            )
        score = file_entry.get("score")
        if not isinstance(score, int | float):
            raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search file entries must include numeric score")
        reason = file_entry.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search file entry reason must be a string")
        evidence = file_entry.get("evidence", [])
        if not isinstance(evidence, list):
            raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search file entry evidence must be a list")

    notes = parsed.get("notes", "")
    if notes is not None and not isinstance(notes, str):
        raise MalformedOpenAICompatibleResponse("codebase_start_job_related_files_search notes must be a string")

    return {"files": files, "notes": notes or ""}


def _strip_json_response_fence(raw_response: str) -> str:
    payload = raw_response.strip()
    if payload.startswith("`"):
        lines = payload.splitlines()
        first_line = lines[0].strip()
        longest_backtick_run = max((len(match.group(0)) for match in re.finditer(r"`+", first_line)), default=0)
        fence = "`" * longest_backtick_run
        if first_line.startswith(f"{fence}json"):
            first_line = first_line[len(fence) + len("json") :]
            last_line = lines[-1].strip()
            if last_line.endswith(fence):
                last_line = last_line[: -len(fence)]
            
            lines = [first_line] + lines[1:-1] + [last_line]
            payload = "\n".join(lines).strip()
        elif first_line.startswith(fence):
            first_line = first_line[len(fence) :]
            last_line = lines[-1].strip()
            if last_line.endswith(fence):
                last_line = last_line[: -len(fence)]
            
            lines = [first_line] + lines[1:-1] + [last_line]
            payload = "\n".join(lines).strip()
    
    return payload


async def _run_stdio_with_io_debug(config: AgentConfig) -> None:
    """Run FastMCP stdio with client/server IO logging enabled."""

    logger = IODebugLogger(config.io_debug.log_path)
    async with stdio_server() as (read_stream, write_stream):
        async with logged_io_streams(
            read_stream,
            write_stream,
            logger=logger,
            path="client-server",
            inbound_direction="client_to_server",
            outbound_direction="server_to_client",
        ) as (logged_read_stream, logged_write_stream):
            await mcp._mcp_server.run(
                logged_read_stream,
                logged_write_stream,
                mcp._mcp_server.create_initialization_options(),
            )


def console_script__init() -> None:
    """Console script entry point for MCP stdio server."""

    config_path: Path = ensure_config_exists()
    print(f"The configuration file is located at \"{config_path}\". Please review and edit it according to your needs.")


def console_script__ensure_qdrant_models() -> None:
    """Console script entry point for initializing Qdrant embeddings."""

    config_path: Path = ensure_config_exists()

    from codebase_agent.built_in_plugins.qdrant_client import (
        apply_qdrant_cache_dir_path, 
        ensure_qdrant_cache_dir_path,
        ensure_qdrant_models,
    )

    console_script__init()

    apply_qdrant_cache_dir_path(ensure_qdrant_cache_dir_path())
    config: AgentConfig = load_config()
    ensure_qdrant_models(config)


def console_script__install_skills_to_current_dir() -> None:
    config_path: Path = ensure_config_exists()

    harness_names = [
        'antigravity',
        'claude',
        'codex',
        'cursor',
        'hermes',
        'opencode',
        'pi_agent',
    ]

    integration_dir: Path = (get_app_data_dir_path() / "integration_to").resolve()
    for harness_name in harness_names:
        harness_dir: Path = integration_dir / harness_name
        if not dir_exists(str(harness_dir)):
            print(f"Integration harness '{harness_name}' is not available in the current installation.")
            continue

        target_dir: Path = Path.cwd()
        try:
            import shutil
            shutil.copytree(harness_dir, target_dir, dirs_exist_ok=True)
            print(f"Integration harness '{harness_name}' has been installed to the current directory.")
        except Exception as ex:
            print(f"Failed to install integration harness '{harness_name}': {ex}")


def console_script__sanitize_library_codebases() -> None:
    print("Functionality will be added soon - stay tuned for updates.")
    config_path: Path = ensure_config_exists()

def console_script__index_dependency_libraries() -> None:
    print("Functionality will be added soon - stay tuned for updates.")
    config_path: Path = ensure_config_exists()

def main() -> None:
    """Run the MCP server over stdio."""

    config_path: Path = ensure_config_exists()

    from codebase_agent.built_in_plugins.qdrant_client import apply_qdrant_cache_dir_path, ensure_qdrant_cache_dir_path

    apply_qdrant_cache_dir_path(ensure_qdrant_cache_dir_path())

    config = load_config()
    if config.io_debug.enabled and config.io_debug.client_server:
        anyio.run(_run_stdio_with_io_debug, config)
        return

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
