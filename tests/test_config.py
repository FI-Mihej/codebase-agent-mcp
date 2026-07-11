from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic import ValidationError

from codebase_agent.config import OpenAICompatibleConfig, load_config
from codebase_agent.openai_compatible_client import normalize_base_url


def _write_config(path: Path, library_root: Path, *, tool_backend: str = "openai_tools") -> Path:
    config_path = path / "codebase_agent.config.json"
    config_path.write_text(
        json.dumps(
            {
                "openai_compatible": {
                    "base_url": "http://localhost:1234",
                    "model": "local-model",
                    "api_key": "test-api-key",
                    "tool_backend": tool_backend,
                    "built_in_plugins": ["built_in_fs"],
                },
                "libraries": [
                    {
                        "name": "example-lib",
                        "path": str(library_root),
                        "instructions": "Check examples first.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_config_with_plugin_restrictions(path: Path, library_root: Path) -> Path:
    config_path = path / "codebase_agent.config.json"
    config_path.write_text(
        json.dumps(
            {
                "openai_compatible": {
                    "base_url": "http://localhost:1234",
                    "model": "local-model",
                    "api_key": "test-api-key",
                    "tool_backend": "openai_tools",
                    "built_in_plugins": [
                        {
                            "name": "built_in_fs",
                            "denied_tools": ["fs__read_text_file", "fs__search_text_in_files"],
                        }
                    ],
                },
                "libraries": [
                    {
                        "name": "example-lib",
                        "allowed": True,
                        "path": str(library_root),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_config(tmp_path: Path) -> None:
    library_root = tmp_path / "example"
    library_root.mkdir()
    config_path = _write_config(tmp_path, library_root)

    config = load_config(config_path)

    assert config.openai_compatible.model == "local-model"
    assert config.openai_compatible.tool_backend == "openai_tools"
    assert config.allowed_libraries()[0].name == "example-lib"
    assert config.jobs.max_concurrent_jobs == 2
    assert config.jobs.storage_backend == "sqlite"
    assert config.jobs.sqlite_path == tmp_path / "codebase_agent.jobs.sqlite3"
    assert config.jobs.sqlite_path.name == "codebase_agent.jobs.sqlite3"
    assert config.jobs.job_ttl_seconds == 3600
    assert config.jobs.max_completed_jobs == 100
    assert config.jobs.result_wait_timeout_seconds == 50
    assert config.jobs.result_poll_interval_seconds == 1
    assert config.io_debug.enabled is False
    assert config.io_debug.client_server is True
    assert config.io_debug.server_plugins is True
    assert config.io_debug.log_path == tmp_path / "codebase_agent.io.jsonl"


def test_load_config_accepts_plugin_denied_tools(tmp_path: Path) -> None:
    library_root = tmp_path / "example"
    library_root.mkdir()
    config = load_config(_write_config_with_plugin_restrictions(tmp_path, library_root))

    assert config.openai_compatible.has_allowed_built_in_plugin("built_in_fs")
    assert config.openai_compatible.allowed_plugins()[0].name == "built_in_fs"
    assert config.openai_compatible.denied_tools_for_allowed_built_in_plugin("built_in_fs") == frozenset(
        {"fs__read_text_file", "fs__search_text_in_files"}
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:1234", "http://localhost:1234/v1"),
        ("http://localhost:1234/", "http://localhost:1234/v1"),
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),
        ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
    ],
)
def test_base_url_normalization(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_context_compression_threshold_validates_range(threshold: float) -> None:
    with pytest.raises(ValidationError):
        OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            context_compression_threshold=threshold,
        )


def test_tool_backend_none(tmp_path: Path) -> None:
    library_root = tmp_path / "example"
    library_root.mkdir()
    config = load_config(_write_config(tmp_path, library_root, tool_backend="none"))

    assert config.openai_compatible.tool_backend == "none"


def test_tool_backend_openai_tools(tmp_path: Path) -> None:
    library_root = tmp_path / "example"
    library_root.mkdir()
    config = load_config(_write_config(tmp_path, library_root, tool_backend="openai_tools"))

    assert config.openai_compatible.tool_backend == "openai_tools"


def test_relative_io_debug_log_path_resolves_against_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "project"
    library_root = tmp_path / "example"
    config_dir.mkdir()
    library_root.mkdir()
    config_path = config_dir / "codebase_agent.config.json"
    config_path.write_text(
        json.dumps(
            {
                "openai_compatible": {
                    "base_url": "http://localhost:1234",
                    "model": "local-model",
                    "api_key": "test-api-key",
                },
                "libraries": [
                    {
                        "name": "example-lib",
                        "allowed": True,
                        "path": str(library_root),
                    }
                ],
                "io_debug": {"enabled": True, "log_path": "logs/io.jsonl"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.io_debug.enabled is True
    assert config.io_debug.log_path == config_dir / "logs" / "io.jsonl"


def test_relative_sqlite_path_resolves_against_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "project"
    process_cwd = tmp_path / "host-cwd"
    library_root = tmp_path / "example"
    config_dir.mkdir()
    process_cwd.mkdir()
    library_root.mkdir()
    config_path = config_dir / "codebase_agent.config.json"
    config_path.write_text(
        json.dumps(
            {
                "openai_compatible": {
                    "base_url": "http://localhost:1234",
                    "model": "local-model",
                    "api_key": "test-api-key",
                },
                "libraries": [
                    {
                        "name": "example-lib",
                        "allowed": True,
                        "path": str(library_root),
                    }
                ],
                "jobs": {"sqlite_path": "jobs.sqlite3"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(process_cwd)

    config = load_config(config_path)

    assert config.jobs.sqlite_path == config_dir / "jobs.sqlite3"
