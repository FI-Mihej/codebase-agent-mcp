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


"""Persistent background job lifecycle for codebase analysis."""


from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codebase_agent.config import (
    AgentConfig,
    JobConfig,
    validate_library_path,
    get_local_data_path,
)
from codebase_agent.app_context import AppContext
from codebase_agent.types import (
    CodebaseAgentError,
    ConcurrentJobLimitError,
    EmptyQueryError,
    InvalidJobIdError,
    UnknownLibraryError,
)
from codebase_agent.progress import (
    SUBLLM_PROGRESS_COUNTERS,
    SUBLLM_STREAM_PROGRESS_COUNTERS,
)
import traceback
import math
import string
from cengal.introspection.inspect import gmsodv
import logging


logger = logging.getLogger(__name__)


JobStatus = Literal["queued", "running", "done", "error", "cancelled", "not_found"]
TerminalJobStatus = Literal["done", "error", "cancelled"]
Progress = dict[str, Any]
AnalysisRunner = Callable[..., Awaitable[str]]

ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"done", "error", "cancelled"}
INTERRUPTED_ERROR = {
    "code": "interrupted_analysis_job",
    "message": "Analysis job was interrupted by MCP server shutdown or restart before completion.",
}
USE_URLSAFE_JOB_ID = True
_TOKEN_URLSAFE_ALPHABET = frozenset(string.ascii_letters + string.digits + "-_")
JOB_ID_TOKEN_URLSAFE_NBYTES = 8
JOB_ID_TOKEN_URLSAFE_EXPECTED_LENGTH = math.ceil(JOB_ID_TOKEN_URLSAFE_NBYTES * 4 / 3)


def gen_job_id() -> str:
    if USE_URLSAFE_JOB_ID:
        return secrets.token_urlsafe(8)  # Shorter, URL-safe job ID
    else:
        return uuid.uuid4().hex


def get_db_dir_path() -> Path:
    """Return the directory path for storing SQLite database files."""
    return (get_local_data_path() / "db").resolve()


@dataclass
class AnalysisJob:
    """Mutable state for a single analysis job."""

    job_id: str
    job_type: str | None
    library_name: str
    query: str
    status: JobStatus
    progress: Progress
    result: str | None = None
    partial_result: str | None = None
    error: dict[str, str] | None = None
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False, compare=False)

    def touch(self) -> None:
        self.updated_at = time.time()

    def mark_terminal(self, status: TerminalJobStatus) -> None:
        self.status = status
        self.completed_at = time.time()
        self.touch()


class CodebaseAnalysisJobManager:
    """Manage bounded, expiring background analysis jobs backed by SQLite."""

    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisJob] = {}
        self._last_settings = JobConfig()
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_limit: int | None = None
        self._storage_path: Path | None = None
        self._initialized_storage_paths: set[Path] = set()
        self._conditions: dict[str, asyncio.Condition] = {}
        self._start_lock = asyncio.Lock()
        self._global_progress_counters: dict[str, int] = {
            counter_name: 0 for counter_name in SUBLLM_PROGRESS_COUNTERS
        }

    @property
    def global_progress_counters(self) -> dict[str, int]:
        """Return aggregate sub-LLM counters for this manager instance."""

        return dict(self._global_progress_counters)

    async def report_subllm_progress(
        self,
        job_id: str | None,
        increments: Mapping[str, int],
    ) -> None:
        """Attribute increments to one job and to the server-lifetime aggregate."""

        normalized_increments: dict[str, int] = {}
        for counter_name in SUBLLM_PROGRESS_COUNTERS:
            raw_increment = increments.get(counter_name)
            if raw_increment is None:
                continue

            try:
                increment = int(raw_increment)
            except (TypeError, ValueError):
                continue

            if increment > 0:
                normalized_increments[counter_name] = increment

        if not normalized_increments:
            return

        global_counters_updated_num: int = 0
        for counter_name, increment in normalized_increments.items():
            self._global_progress_counters[counter_name] += increment
            if counter_name not in SUBLLM_STREAM_PROGRESS_COUNTERS:
                global_counters_updated_num += 1

        if global_counters_updated_num:
            logger.info(f"Global progress conters: \n{gmsodv(self._global_progress_counters, shift_num=1)}")
        
        if job_id is None:
            return

        job = self._jobs.get(job_id)
        if job is None or job.status not in ACTIVE_STATUSES:
            return

        job_counters_updated_num: int = 0
        for counter_name, increment in normalized_increments.items():
            job.progress[counter_name] = (
                _progress_counter_value(job.progress, counter_name) + increment
            )
            if counter_name not in SUBLLM_STREAM_PROGRESS_COUNTERS:
                job_counters_updated_num += 1

        if job_counters_updated_num:
            logger.info(f"Job \"{job.job_id}\" progress conters: \n{gmsodv(job.progress, shift_num=1)}")

        job.touch()
        self._save_job(job)
        await self._notify(job_id)

    async def start_job(
        self,
        *,
        app_context: AppContext,
        config: AgentConfig,
        library_name: str,
        query: str,
        runner: AnalysisRunner,
        job_type: str | None = None,
    ) -> dict[str, Any]:
        """Create a job, persist it, and schedule analysis outside the MCP request path."""

        self._last_settings = config.jobs
        self._ensure_storage(config.jobs)
        self._cleanup(config.jobs)

        normalized_query = query.strip()
        if not normalized_query:
            raise EmptyQueryError("query must not be empty")
        
        library = config.get_allowed_library(library_name)
        if library is None:
            raise UnknownLibraryError(f"Unknown library name: {library_name}")
        
        validate_library_path(library.path, library.name)
        async with self._start_lock:
            active_job_count = self._active_job_count()
            max_concurrent_jobs = config.jobs.max_concurrent_jobs
            if active_job_count >= max_concurrent_jobs:
                raise ConcurrentJobLimitError(
                    "Concurrent codebase job limit reached: "
                    f"{active_job_count} active job(s), max_concurrent_jobs={max_concurrent_jobs}. "
                    "The client LLM must wait, continuing to poll, until already started related-file search "
                    "or analysis jobs reach a terminal state before starting another job."
                )

            now = time.time()
            job_id: str
            if USE_URLSAFE_JOB_ID:
                job_id = secrets.token_urlsafe(8)  # Shorter, URL-safe job ID
            else:
                job_id = uuid.uuid4().hex

            job = AnalysisJob(
                job_id=job_id,
                job_type=job_type,
                library_name=library_name,
                query=normalized_query,
                status="running",
                progress=_progress_payload(
                    phase="queued",
                    message="Job created",
                    completed_steps=0,
                    total_steps=2,
                ),
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            self._save_job(job)
            job.task = asyncio.create_task(
                self._run_job(
                    app_context=app_context,
                    job=job,
                    config=config,
                    library_name=library_name,
                    query=normalized_query,
                    runner=runner,
                )
            )

        await self._notify(job_id)
        return {"job_id": job_id, "status": job.status}

    async def get_status(
        self,
        job_id: str,
        settings: JobConfig | None = None,
    ) -> dict[str, Any]:
        """Return current job status and progress, waiting briefly for active jobs."""

        settings = settings or self._last_settings
        self._last_settings = settings
        _validate_job_id(job_id)
        self._ensure_storage(settings)
        self._cleanup(settings)

        job = self._get_job(job_id)
        if job is None:
            return _not_found_payload(job_id)

        job = await self._wait_for_terminal(job, settings)
        return self._status_payload(job)

    async def get_result(
        self,
        job_id: str,
        settings: JobConfig | None = None,
    ) -> dict[str, Any]:
        """Return completed result, error, or current partial result after a bounded wait."""

        settings = settings or self._last_settings
        self._last_settings = settings
        _validate_job_id(job_id)
        self._ensure_storage(settings)
        self._cleanup(settings)

        job = self._get_job(job_id)
        if job is None:
            return _not_found_payload(job_id)

        if job.status in ACTIVE_STATUSES and job.result is None:
            job = await self._wait_for_terminal(job, settings)
        return self._result_payload(job)

    async def get_job_type(
        self,
        job_id: str,
        settings: JobConfig | None = None,
    ) -> str | None:
        """Return the stored type for a job, if the job exists and has one."""

        settings = settings or self._last_settings
        self._last_settings = settings
        _validate_job_id(job_id)
        self._ensure_storage(settings)
        self._cleanup(settings)

        job = self._get_job(job_id)
        if job is None:
            return None

        return job.job_type

    async def cancel(
        self,
        job_id: str,
        settings: JobConfig | None = None,
    ) -> dict[str, Any]:
        """Request cancellation for a queued or running job."""

        settings = settings or self._last_settings
        self._last_settings = settings
        _validate_job_id(job_id)
        self._ensure_storage(settings)
        self._cleanup(settings)

        job = self._get_job(job_id)
        if job is None:
            return _not_found_payload(job_id)

        if job.status not in ACTIVE_STATUSES:
            return {"job_id": job_id, "status": job.status}

        job.cancel_requested = True
        job.progress = _progress_payload(
            phase="cancelled",
            message="Cancellation requested",
            source_progress=job.progress,
        )
        self._mark_terminal(job, "cancelled")
        if job.task is not None and not job.task.done():
            job.task.cancel()
        await self._notify(job_id)
        return {"job_id": job_id, "status": "cancelled"}

    async def _run_job(
        self,
        *,
        app_context: AppContext,
        job: AnalysisJob,
        config: AgentConfig,
        library_name: str,
        query: str,
        runner: AnalysisRunner,
    ) -> None:
        semaphore = self._semaphore_for(config.jobs.max_concurrent_jobs)
        if semaphore.locked():
            job.status = "queued"
            job.progress = _progress_payload(
                phase="queued",
                message="Waiting for an analysis slot",
                completed_steps=0,
                total_steps=2,
                source_progress=job.progress,
            )
            job.touch()
            self._save_job(job)
            await self._notify(job.job_id)

        try:
            async with semaphore:
                if job.cancel_requested:
                    self._mark_terminal(job, "cancelled")
                    await self._notify(job.job_id)
                    return

                job.status = "running"
                job.started_at = time.time()
                job.progress = _progress_payload(
                    phase="running",
                    message="Job running",
                    completed_steps=1,
                    total_steps=2,
                    source_progress=job.progress,
                )
                job.partial_result = "Job running"
                job.touch()
                self._save_job(job)
                await self._notify(job.job_id)

                result = await runner(
                    app_context=app_context,
                    config=config,
                    library_name=library_name,
                    query=query,
                    job_id=job.job_id,
                )

                if job.cancel_requested:
                    self._mark_terminal(job, "cancelled")
                    await self._notify(job.job_id)
                    return

                job.result = result
                # job.partial_result = result
                job.partial_result = "Job done, result available."
                job.progress = _progress_payload(
                    phase="complete",
                    message="Analysis completed",
                    completed_steps=2,
                    total_steps=2,
                    source_progress=job.progress,
                )
                self._mark_terminal(job, "done")
                await self._notify(job.job_id)
        except asyncio.CancelledError:
            self._mark_terminal(job, "cancelled")
            await self._notify(job.job_id)
            raise
        except Exception as exc:
            job.error = _serialize_error(exc)
            job.progress = _progress_payload(
                phase="error",
                message="Job failed",
                completed_steps=1,
                total_steps=2,
                source_progress=job.progress,
            )
            self._mark_terminal(job, "error")
            await self._notify(job.job_id)

    async def _wait_for_terminal(self, job: AnalysisJob, settings: JobConfig) -> AnalysisJob:
        timeout = settings.result_wait_timeout_seconds
        if timeout <= 0 or job.status not in ACTIVE_STATUSES:
            return job

        deadline = time.monotonic() + timeout
        interval = settings.result_poll_interval_seconds
        while job.status in ACTIVE_STATUSES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            condition = self._condition_for(job.job_id)
            try:
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=min(interval, remaining))
            except TimeoutError:
                pass

            refreshed = self._get_job(job.job_id)
            if refreshed is None:
                break
            job = refreshed

        return job

    def _ensure_storage(self, settings: JobConfig) -> None:
        storage_path = self._resolve_storage_path(settings.sqlite_path)
        self._last_settings = settings
        if self._storage_path != storage_path:
            self._storage_path = storage_path

        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT,
                    library_name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_json TEXT,
                    result TEXT,
                    partial_result TEXT,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                )
                """
            )
            self._ensure_column(connection, "analysis_jobs", "job_type", "TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_updated_at ON analysis_jobs(updated_at)"
            )

            if storage_path not in self._initialized_storage_paths:
                now = time.time()
                connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status = ?,
                        progress_json = ?,
                        error_json = ?,
                        updated_at = ?,
                        completed_at = COALESCE(completed_at, ?)
                    WHERE status IN ('queued', 'running')
                    """,
                    (
                        "error",
                        _dump_json({"phase": "error", "message": INTERRUPTED_ERROR["message"]}),
                        _dump_json(INTERRUPTED_ERROR),
                        now,
                        now,
                    ),
                )
                self._initialized_storage_paths.add(storage_path)

    def _resolve_storage_path(self, raw_path: Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        
        return (get_db_dir_path() / path).resolve()

    def _connect(self) -> sqlite3.Connection:
        if self._storage_path is None:
            raise RuntimeError("job storage has not been initialized")

        connection = sqlite3.connect(self._storage_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _save_job(self, job: AnalysisJob) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    job_id,
                    job_type,
                    library_name,
                    query,
                    status,
                    progress_json,
                    result,
                    partial_result,
                    error_json,
                    cancel_requested,
                    created_at,
                    updated_at,
                    started_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type = excluded.job_type,
                    library_name = excluded.library_name,
                    query = excluded.query,
                    status = excluded.status,
                    progress_json = excluded.progress_json,
                    result = excluded.result,
                    partial_result = excluded.partial_result,
                    error_json = excluded.error_json,
                    cancel_requested = excluded.cancel_requested,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at
                """,
                (
                    job.job_id,
                    job.job_type,
                    job.library_name,
                    job.query,
                    job.status,
                    _dump_json(job.progress),
                    job.result,
                    job.partial_result,
                    _dump_json(job.error),
                    int(job.cancel_requested),
                    job.created_at,
                    job.updated_at,
                    job.started_at,
                    job.completed_at,
                ),
            )

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        existing_columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )

    def _get_job(self, job_id: str) -> AnalysisJob | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None

        job = _job_from_row(row)
        self._jobs[job_id] = job
        return job

    def _mark_terminal(self, job: AnalysisJob, status: TerminalJobStatus) -> None:
        job.mark_terminal(status)
        self._save_job(job)

    def _semaphore_for(self, max_concurrent_jobs: int) -> asyncio.Semaphore:
        if self._semaphore is None or self._semaphore_limit != max_concurrent_jobs:
            self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
            self._semaphore_limit = max_concurrent_jobs
        return self._semaphore

    def _active_job_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS active_count
                FROM analysis_jobs
                WHERE status IN ('queued', 'running')
                """
            ).fetchone()
        return int(row["active_count"])

    def _condition_for(self, job_id: str) -> asyncio.Condition:
        condition = self._conditions.get(job_id)
        if condition is None:
            condition = asyncio.Condition()
            self._conditions[job_id] = condition
        return condition

    async def _notify(self, job_id: str) -> None:
        condition = self._conditions.get(job_id)
        if condition is None:
            return
        async with condition:
            condition.notify_all()

    def _cleanup(self, settings: JobConfig) -> None:
        now = time.time()
        cutoff = now - settings.job_ttl_seconds
        expired_ids: list[str] = []
        overflow_ids: list[str] = []
        with self._connect() as connection:
            expired_rows = connection.execute(
                """
                SELECT job_id FROM analysis_jobs
                WHERE status IN ('done', 'error', 'cancelled')
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff,),
            ).fetchall()
            expired_ids = [str(row["job_id"]) for row in expired_rows]
            if expired_ids:
                connection.executemany(
                    "DELETE FROM analysis_jobs WHERE job_id = ?",
                    [(job_id,) for job_id in expired_ids],
                )

            if settings.max_completed_jobs == 0:
                overflow_rows = connection.execute(
                    """
                    SELECT job_id FROM analysis_jobs
                    WHERE status IN ('done', 'error', 'cancelled')
                      AND completed_at IS NOT NULL
                    """
                ).fetchall()
                overflow_ids = [str(row["job_id"]) for row in overflow_rows]
            else:
                overflow_rows = connection.execute(
                    """
                    SELECT job_id FROM analysis_jobs
                    WHERE status IN ('done', 'error', 'cancelled')
                      AND completed_at IS NOT NULL
                    ORDER BY completed_at DESC, updated_at DESC, created_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (settings.max_completed_jobs,),
                ).fetchall()
                overflow_ids = [str(row["job_id"]) for row in overflow_rows]

            if overflow_ids:
                connection.executemany(
                    "DELETE FROM analysis_jobs WHERE job_id = ?",
                    [(job_id,) for job_id in overflow_ids],
                )

        for job_id in set(expired_ids + overflow_ids):
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                self._jobs.pop(job_id, None)

    def _status_payload(self, job: AnalysisJob) -> dict[str, Any]:
        result = {
            "job_id": job.job_id,
            "status": job.status,
            # "progress": job.progress,
            "progress": _progress_payload_short(job.progress),
            # "error": job.error,
        }
        if job.error:
            result["error"] = job.error
        
        if "running" == job.status:
            # result["llm_agent_instructions"] = "You must poll using `codebase_get_job_status` tool until `success`/`failure`. `codebase_get_job_status` tool internally waits up to 50s/request. You are forbidden to finish response while polling is running. If `codebase_get_job_status` returns 'running', continue polling; even if the `progress` field does not change for several calls, assume the process is working unless you receive a terminal error."
            result["llm_agent_instructions"] = "You must poll using `codebase_get_job_status` tool until `success`/`failure`. `codebase_get_job_status` tool internally waits up to 50s/request. You are forbidden to finish response untill you receive a finished result or terminal error."
        elif "done" == job.status:
            result["llm_agent_instructions"] = "The analysis job has completed successfully. You must now retrieve the result using `codebase_get_job_result` tool."

        return result

    def _result_payload(self, job: AnalysisJob) -> dict[str, Any]:
        result = {
            "job_id": job.job_id,
            "status": job.status,
            "result": job.result,
            # "partial_result": job.partial_result,
            # "error": job.error,
        }
        if job.error:
            result["error"] = job.error
        
        if "running" == job.status:
            result["llm_agent_instructions"] = "You must poll using `codebase_get_job_status` tool until `success`/`failure`. `codebase_get_job_result` tool internally waits up to 50s/request. You are forbidden to finish response while polling is running."

        return result


def _progress_payload(
    *,
    phase: str,
    message: str,
    completed_steps: int | None = None,
    total_steps: int | None = None,
    source_progress: Mapping[str, Any] | None = None,
) -> Progress:
    progress: Progress = {
        "phase": phase,
        "message": message,
    }
    if completed_steps is not None:
        progress["completed_steps"] = completed_steps
    if total_steps is not None:
        progress["total_steps"] = total_steps

    for counter_name in SUBLLM_PROGRESS_COUNTERS:
        progress[counter_name] = _progress_counter_value(source_progress, counter_name)

    return progress


def _progress_payload_short(
    source_progress: Mapping[str, Any] | None = None,
) -> Progress:
    progress: Progress = dict()
    # if source_progress is not None:
    #     if "phase" in source_progress:
    #         if source_progress["phase"] is not None:
    #             progress["phase"] = source_progress["phase"]
    
    progress["input_tokens_processed"] = _progress_counter_value(source_progress, "input_tokens_used_by_subllm")
    # progress["output_tokens_generated"] = _progress_counter_value(source_progress, "output_tokens_generated_by_subllm")
    progress["output_bytes_generated"] = _progress_counter_value(source_progress, "output_bytes_generated_by_subllm")
    progress["context_compations_made"] = _progress_counter_value(source_progress, "llm_context_compations_made_by_subharness")

    return progress


def _progress_counter_value(
    progress: Mapping[str, Any] | None,
    counter_name: str,
) -> int:
    if progress is None:
        return 0

    try:
        return int(progress.get(counter_name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _job_from_row(row: sqlite3.Row) -> AnalysisJob:
    return AnalysisJob(
        job_id=str(row["job_id"]),
        job_type=row["job_type"],
        library_name=str(row["library_name"]),
        query=str(row["query"]),
        status=row["status"],
        progress=_load_json(row["progress_json"], default={}),
        result=row["result"],
        partial_result=row["partial_result"],
        error=_load_json(row["error_json"], default=None),
        cancel_requested=bool(row["cancel_requested"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=_optional_float(row["started_at"]),
        completed_at=_optional_float(row["completed_at"]),
    )


def _dump_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(value: str | None, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _not_found_payload(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "not_found",
        "message": (
            "No job was found for the provided job_id. The client LLM must re-check "
            "that it is using the exact job_id returned by the corresponding start tool "
            "and retry the request with the correct identifier."
        ),
    }


if USE_URLSAFE_JOB_ID:
    def _validate_job_id(job_id: str) -> None:
        if (
            len(job_id) != JOB_ID_TOKEN_URLSAFE_EXPECTED_LENGTH
            or any(ch not in _TOKEN_URLSAFE_ALPHABET for ch in job_id)
        ):
            raise InvalidJobIdError(
                f"job_id must be the exact {JOB_ID_TOKEN_URLSAFE_EXPECTED_LENGTH}-character value "
                f"returned by secrets.token_urlsafe({JOB_ID_TOKEN_URLSAFE_NBYTES}); "
                f"received {job_id!r} ({len(job_id)} characters)."
            )
else:
    def _validate_job_id(job_id: str) -> None:
        if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise InvalidJobIdError(
                "job_id must be the exact 32-character lowercase hex id returned by the start tool; "
                f"received {job_id!r} ({len(job_id)} characters)."
            )


def _serialize_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, CodebaseAgentError):
        return {"code": exc.code, "message": str(exc)}
    
    return {"type": type(exc).__name__, "code": "unexpected_analysis_error", "message": str(exc), "traceback": ''.join(traceback.TracebackException.from_exception(exc).format())}
