from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

from codebase_agent.config import AgentConfig, JobConfig, OpenAICompatibleConfig, LibraryConfig
from codebase_agent.app_context import AppContext
from codebase_agent.jobs import CodebaseAnalysisJobManager
from codebase_agent.built_in_plugins.qdrant_client import QdrantClientABC
from codebase_agent.built_in_plugins.local_fs_tools import LocalFilesystemTools
from codebase_agent.types import (
    ClientRequestType,
    ConcurrentJobLimitError,
    InvalidJobIdError,
    OpenAICompatibleContextOverflowError,
    QueryGranularityError,
    ToolResult,
)


def _config(tmp_path: Path, **job_overrides: Any) -> AgentConfig:
    library_root = tmp_path / "lib"
    library_root.mkdir(exist_ok=True)
    jobs = {
        "sqlite_path": tmp_path / "jobs.sqlite3",
        "result_wait_timeout_seconds": 0.01,
        "result_poll_interval_seconds": 0.005,
    }
    jobs.update(job_overrides)
    return AgentConfig(
        openai_compatible=OpenAICompatibleConfig(
            base_url="http://localhost:1234",
            model="local-model",
            api_key="test-api-key",
            tool_backend="none",
        ),
        libraries=[
            LibraryConfig(
                name="example-lib",
                path=library_root,
                instructions="Use tests as evidence.",
            )
        ],
        jobs=JobConfig(**jobs),
    )


class FakeQdrantClient(QdrantClientABC):
    def name(self) -> str:
        return "qdrant_fake"

    def all_tools_names(self) -> list[str]:
        return []

    def allowed_tool_names(self) -> list[str]:
        return []

    def worker_by_tool_name(self, tool_name: str) -> Any:
        return None

    def execute(
        self,
        library_name: str,
        client_request_type: ClientRequestType,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        return {"ok": False, "error": {"code": "tool_not_enabled", "message": tool_name}}

    def all_tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def allowed_tool_definitions(self) -> list[dict[str, Any]]:
        return []


def _app_context(config: AgentConfig, manager: CodebaseAnalysisJobManager) -> AppContext:
    qdrant_client = FakeQdrantClient()
    local_fs_tools = LocalFilesystemTools(config=config)
    return AppContext(
        config=config,
        analysis_jobs=manager,
        plugins={qdrant_client.name(): qdrant_client, local_fs_tools.name(): local_fs_tools},
        qdrant_client=qdrant_client,
        local_fs_tools=local_fs_tools,
        text_file_tools=None,
    )

def test_background_job_completes_with_result(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)

        async def fake_runner(**kwargs: Any) -> str:
            await asyncio.sleep(0)
            return f"answer for {kwargs['library_name']}"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        result = await manager.get_result(started["job_id"])

        assert started["status"] == "running"
        assert result["status"] == "done"
        assert result["result"] == "answer for example-lib"

    asyncio.run(run())


def test_background_job_preserves_structured_error(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)

        async def fake_runner(**kwargs: Any) -> str:
            raise ValueError("boom")

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        result = await manager.get_result(started["job_id"])

        assert result["status"] == "error"
        assert result["error"] == {
            "code": "unexpected_analysis_error",
            "message": "boom",
        }

    asyncio.run(run())


def test_background_job_returns_actionable_context_overflow_error(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)

        async def fake_runner(**kwargs: Any) -> str:
            raise OpenAICompatibleContextOverflowError(
                "The OpenAI-compatible model used by this tool could not process the request because "
                "the accumulated prompt exceeded its context window. The client LLM must retry with a "
                "narrower request."
            )

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        result = await manager.get_result(started["job_id"])

        assert result["status"] == "error"
        assert result["error"]["code"] == "openai_compatible_context_overflow"
        assert "must retry with a narrower request" in result["error"]["message"]

    asyncio.run(run())


def test_background_job_returns_query_granularity_error_via_polling(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)

        async def fake_runner(**kwargs: Any) -> str:
            raise QueryGranularityError(
                "codebase_start_job_analysis rejected the request because it is too broad.\n\n"
                "Client LLM instruction: One topic or one context per request -> wait for the result -> send the next request!"
            )

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="Analyze X and Y.",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        result = await manager.get_result(started["job_id"])

        assert started["status"] == "running"
        assert result["status"] == "error"
        assert result["error"]["code"] == "query_granularity_violation"
        assert "Client LLM instruction: One topic or one context per request" in result["error"]["message"]

    asyncio.run(run())


def test_start_job_rejects_when_concurrent_job_limit_is_reached(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path, max_concurrent_jobs=1)
        release = asyncio.Event()

        async def fake_runner(**kwargs: Any) -> str:
            await release.wait()
            return "done"

        results = await asyncio.gather(
            manager.start_job(
                app_context=_app_context(config, manager),
                config=config,
                library_name="example-lib",
                query="first",
                runner=fake_runner,
            ),
            manager.start_job(
                app_context=_app_context(config, manager),
                config=config,
                library_name="example-lib",
                query="second",
                runner=fake_runner,
            ),
            return_exceptions=True,
        )
        started_jobs = [result for result in results if isinstance(result, dict)]
        errors = [result for result in results if isinstance(result, ConcurrentJobLimitError)]

        with sqlite3.connect(config.jobs.sqlite_path) as connection:
            active_count = connection.execute(
                "SELECT COUNT(*) FROM analysis_jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0]

        assert len(started_jobs) == 1
        assert len(errors) == 1
        started = started_jobs[0]
        release.set()
        await manager.get_result(started["job_id"], config.jobs)

        assert "Concurrent codebase job limit reached" in str(errors[0])
        assert "must wait, continuing to poll, until already started" in str(errors[0])
        assert started["status"] == "running"
        assert active_count == 1

    asyncio.run(run())


def test_cancel_background_job(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)

        async def fake_runner(**kwargs: Any) -> str:
            await asyncio.sleep(10)
            return "too late"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )

        cancelled = await manager.cancel(started["job_id"], config.jobs)
        result = await manager.get_result(started["job_id"], config.jobs)

        assert cancelled["status"] == "cancelled"
        assert result["status"] == "cancelled"

    asyncio.run(run())


def test_job_cleanup_respects_max_completed_jobs(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()

        async def fake_runner(**kwargs: Any) -> str:
            return kwargs["query"]

        config = _config(tmp_path, max_completed_jobs=1)
        first = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="first",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)
        second = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="second",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        await manager.get_status(second["job_id"], config.jobs)

        assert (await manager.get_result(second["job_id"], config.jobs))["status"] == "done"
        assert (await manager.get_result(first["job_id"], config.jobs))["status"] == "not_found"

    asyncio.run(run())


def test_job_lookup_returns_not_found_for_missing_job(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)
        missing_job_id = "0" * 32
        expected_payload = {
            "job_id": missing_job_id,
            "status": "not_found",
            "message": (
                "No job was found for the provided job_id. The client LLM must re-check "
                "that it is using the exact job_id returned by the corresponding start tool "
                "and retry the request with the correct identifier."
            ),
        }

        assert await manager.get_status(missing_job_id, config.jobs) == expected_payload
        assert await manager.get_result(missing_job_id, config.jobs) == expected_payload
        assert await manager.cancel(missing_job_id, config.jobs) == expected_payload

    asyncio.run(run())


def test_job_lookup_rejects_malformed_job_id(tmp_path: Path) -> None:
    async def run() -> None:
        manager = CodebaseAnalysisJobManager()
        config = _config(tmp_path)

        for lookup in (manager.get_status, manager.get_result, manager.cancel):
            try:
                await lookup("58c45ca50ab24fb3bcc3c7b3b19e3", config.jobs)
            except InvalidJobIdError as exc:
                assert "exact 32-character lowercase hex id" in str(exc)
            else:
                raise AssertionError("Expected InvalidJobIdError")

    asyncio.run(run())


def test_sqlite_job_creation_and_retrieval(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path)
        manager = CodebaseAnalysisJobManager()

        async def fake_runner(**kwargs: Any) -> str:
            return "stored answer"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        with sqlite3.connect(config.jobs.sqlite_path) as connection:
            row = connection.execute(
                "SELECT library_name, query, status, result FROM analysis_jobs WHERE job_id = ?",
                (started["job_id"],),
            ).fetchone()

        assert row == ("example-lib", "How?", "done", "stored answer")

    asyncio.run(run())


def test_completed_result_survives_registry_reinitialization(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path)
        first_manager = CodebaseAnalysisJobManager()

        async def fake_runner(**kwargs: Any) -> str:
            return "durable answer"

        started = await first_manager.start_job(
            app_context=_app_context(config, first_manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        second_manager = CodebaseAnalysisJobManager()
        result = await second_manager.get_result(started["job_id"], config.jobs)

        assert result["status"] == "done"
        assert result["result"] == "durable answer"

    asyncio.run(run())


def test_failed_job_survives_registry_reinitialization(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path)
        first_manager = CodebaseAnalysisJobManager()

        async def fake_runner(**kwargs: Any) -> str:
            raise ValueError("durable boom")

        started = await first_manager.start_job(
            app_context=_app_context(config, first_manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)

        second_manager = CodebaseAnalysisJobManager()
        result = await second_manager.get_result(started["job_id"], config.jobs)

        assert result["status"] == "error"
        assert result["error"]["message"] == "durable boom"

    asyncio.run(run())


def test_stale_running_job_from_previous_process_is_converted_to_error(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path)
        seed_manager = CodebaseAnalysisJobManager()
        await seed_manager.get_status("0" * 32, config.jobs)
        stale_job_id = "1" * 32
        now = time.time()
        with sqlite3.connect(config.jobs.sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    job_id, library_name, query, status, progress_json,
                    cancel_requested, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stale_job_id, "example-lib", "How?", "running", "{}", 0, now, now),
            )

        restarted_manager = CodebaseAnalysisJobManager()
        result = await restarted_manager.get_result(stale_job_id, config.jobs)

        assert result["status"] == "error"
        assert result["error"]["code"] == "interrupted_analysis_job"
        assert "interrupted" in result["error"]["message"]

    asyncio.run(run())


def test_get_status_waits_when_job_is_running(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(
            tmp_path,
            result_wait_timeout_seconds=0.05,
            result_poll_interval_seconds=0.01,
        )
        manager = CodebaseAnalysisJobManager()
        release = asyncio.Event()

        async def fake_runner(**kwargs: Any) -> str:
            await release.wait()
            return "done"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        started_at = time.monotonic()
        status = await manager.get_status(started["job_id"], config.jobs)
        elapsed = time.monotonic() - started_at
        release.set()
        await manager.cancel(started["job_id"], config.jobs)

        assert status["status"] == "running"
        assert elapsed >= 0.04

    asyncio.run(run())


def test_get_result_waits_when_result_is_not_ready(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(
            tmp_path,
            result_wait_timeout_seconds=0.05,
            result_poll_interval_seconds=0.01,
        )
        manager = CodebaseAnalysisJobManager()
        release = asyncio.Event()

        async def fake_runner(**kwargs: Any) -> str:
            await release.wait()
            return "done"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0)
        started_at = time.monotonic()
        result = await manager.get_result(started["job_id"], config.jobs)
        elapsed = time.monotonic() - started_at
        release.set()
        await manager.cancel(started["job_id"], config.jobs)

        assert result["status"] == "running"
        assert elapsed >= 0.04

    asyncio.run(run())


def test_status_and_result_return_immediately_when_job_completes_during_wait(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(
            tmp_path,
            result_wait_timeout_seconds=1,
            result_poll_interval_seconds=0.2,
        )
        manager = CodebaseAnalysisJobManager()

        async def fake_runner(**kwargs: Any) -> str:
            await asyncio.sleep(0.03)
            return "quick finish"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        started_at = time.monotonic()
        status = await manager.get_status(started["job_id"], config.jobs)
        elapsed_status = time.monotonic() - started_at
        started_at = time.monotonic()
        result = await manager.get_result(started["job_id"], config.jobs)
        elapsed_result = time.monotonic() - started_at

        assert status["status"] == "done"
        assert result["status"] == "done"
        assert result["result"] == "quick finish"
        assert elapsed_status < 0.5
        assert elapsed_result < 0.1

    asyncio.run(run())


def test_result_returns_after_timeout_with_latest_partial_state(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(
            tmp_path,
            result_wait_timeout_seconds=0.05,
            result_poll_interval_seconds=0.01,
        )
        manager = CodebaseAnalysisJobManager()
        release = asyncio.Event()

        async def fake_runner(**kwargs: Any) -> str:
            await release.wait()
            return "done"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0)
        result = await manager.get_result(started["job_id"], config.jobs)
        release.set()
        await manager.cancel(started["job_id"], config.jobs)

        assert result["status"] == "running"
        assert result["partial_result"] == "Job running"

    asyncio.run(run())


def test_expired_completed_jobs_are_cleaned_up_according_to_ttl(tmp_path: Path) -> None:
    async def run() -> None:
        config = _config(tmp_path, job_ttl_seconds=1)
        manager = CodebaseAnalysisJobManager()

        async def fake_runner(**kwargs: Any) -> str:
            return "old answer"

        started = await manager.start_job(
            app_context=_app_context(config, manager),
            config=config,
            library_name="example-lib",
            query="How?",
            runner=fake_runner,
        )
        await asyncio.sleep(0.01)
        old_timestamp = time.time() - 10
        with sqlite3.connect(config.jobs.sqlite_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET completed_at = ?, updated_at = ? WHERE job_id = ?",
                (old_timestamp, old_timestamp, started["job_id"]),
            )
        manager = CodebaseAnalysisJobManager()

        result = await manager.get_result(started["job_id"], config.jobs)

        assert result["status"] == "not_found"

    asyncio.run(run())
