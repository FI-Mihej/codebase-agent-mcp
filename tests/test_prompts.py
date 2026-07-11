from __future__ import annotations

from pathlib import Path

from codebase_agent.config import AgentConfig, OpenAICompatibleConfig, LibraryConfig
from codebase_agent.app_context import AppContext
from codebase_agent.jobs import CodebaseAnalysisJobManager
from codebase_agent.prompts import (
    build_prompt_for__codebase_start_job_related_files_search,
    build_prompt_for__codebase_start_job_analysis,
    build_prompt_for__query_granularity_validation,
)
from codebase_agent.built_in_plugins.local_fs_tools import LocalFilesystemTools
from codebase_agent.built_in_plugins.qdrant_client import QdrantClientFake


def _app_context(tmp_path: Path, openai_compatible: OpenAICompatibleConfig) -> AppContext:
    config = AgentConfig(openai_compatible=openai_compatible, libraries=[LibraryConfig(name="example-lib", path=tmp_path)])
    qdrant_client = QdrantClientFake(config)
    local_fs_tools = LocalFilesystemTools(config=config)
    return AppContext(
        config=config,
        analysis_jobs=CodebaseAnalysisJobManager(),
        plugins={qdrant_client.name(): qdrant_client, local_fs_tools.name(): local_fs_tools},
        qdrant_client=qdrant_client,
        local_fs_tools=local_fs_tools,
        text_file_tools=None,
    )


def test_codebase_start_job_related_files_search_prompt_requests_ranked_json(tmp_path: Path) -> None:
    library = LibraryConfig(
        name="example-lib",
        path=tmp_path,
        instructions="Prefer tests.",
    )
    openai_compatible = OpenAICompatibleConfig(
        base_url="http://localhost:1234",
        model="local-model",
        max_tokens=65536,
        api_key="test-api-key",
        built_in_plugins=["built_in_fs"],
        tool_backend="openai_tools",
    )

    prompt = build_prompt_for__codebase_start_job_related_files_search(
        app_context=_app_context(tmp_path, openai_compatible),
        library=library,
        openai_compatible=openai_compatible,
        user_query="Where is token refresh implemented?",
    )

    assert "65536" in prompt
    assert "locate files" in prompt
    assert "Return **strict JSON only**" in prompt
    assert "path/relative/to/codebase/root.py" in prompt
    assert "Do **not** solve the user's task" in prompt
    assert "fs__list_files" in prompt
    assert "fs__read_text_file" in prompt
    assert "fs__search_text_in_files" in prompt
    assert "Where is token refresh implemented?" in prompt


def test_prompt_construction_includes_required_context(tmp_path: Path) -> None:
    library = LibraryConfig(
        name="example-lib",
        path=tmp_path,
        instructions="Prefer public APIs.",
    )
    openai_compatible = OpenAICompatibleConfig(
        base_url="http://localhost:1234",
        model="local-model",
        max_tokens=65536,
        api_key="test-api-key",
        built_in_plugins=["built_in_fs"],
        tool_backend="openai_tools",
    )

    prompt = build_prompt_for__codebase_start_job_analysis(
        app_context=_app_context(tmp_path, openai_compatible),
        library=library,
        openai_compatible=openai_compatible,
        user_query="How do I implement X?",
    )

    assert "65536" in prompt
    assert "example-lib" in prompt
    assert str(tmp_path) in prompt
    assert "Prefer public APIs." in prompt
    assert "fs__list_files" in prompt
    assert "fs__read_text_file" in prompt
    assert "fs__search_text_in_files" in prompt
    assert "How do I implement X?" in prompt
    assert "Clearly distinguish **verified facts** from **inferred conclusions**." in prompt


def test_prompt_construction_includes_denied_tools(tmp_path: Path) -> None:
    library = LibraryConfig(
        name="example-lib",
        path=tmp_path,
        instructions="Prefer public APIs.",
    )
    openai_compatible = OpenAICompatibleConfig(
        base_url="http://localhost:1234",
        model="local-model",
        api_key="test-api-key",
        built_in_plugins=[
            {
                "name": "built_in_fs",
                "denied_tools": ["fs__read_text_file"],
            }
        ],
        tool_backend="openai_tools",
    )

    prompt = build_prompt_for__codebase_start_job_analysis(
        app_context=_app_context(tmp_path, openai_compatible),
        library=library,
        openai_compatible=openai_compatible,
        user_query="How do I implement X?",
    )

    assert "Use the following tools: fs__list_files, fs__search_text_in_files" in prompt
    assert "fs__read_text_file" not in prompt


def test_query_granularity_validation_prompt_includes_strict_instruction() -> None:
    openai_compatible = OpenAICompatibleConfig(
        base_url="http://localhost:1234",
        model="local-model",
        api_key="test-api-key",
    )

    prompt = build_prompt_for__query_granularity_validation(
        tool_name="codebase_start_job_analysis",
        openai_compatible=openai_compatible,
    )

    assert "strict request gatekeeper" in prompt
    assert "One topic or one context per request" in prompt
    assert '"valid": true' in prompt
    assert "Analyze X and Y" in prompt
