#!/usr/bin/env python
# coding=utf-8

"""Async-scoped progress reporting for background analysis jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token


SUBLLM_PROGRESS_COUNTERS = (
    "input_bytes_used_by_subllm",
    "output_bytes_generated_by_subllm",
    "tool_calls_made_by_subllm",
    "llm_context_compations_made_by_subharness",
    "input_tokens_used_by_subllm",
    "cached_input_tokens_used_by_subllm",
    "output_tokens_generated_by_subllm",
    "llm_anti_stuck_mechanism_activations_by_subharness",
    "llm_goal_fulfillment_auditor_activations_by_subharness",
)

SUBLLM_STREAM_PROGRESS_COUNTERS = {
    "output_bytes_generated_by_subllm",
}


ProgressReporter = Callable[[Mapping[str, int]], Awaitable[None]]

_current_progress_reporter: ContextVar[ProgressReporter | None] = ContextVar(
    "current_subllm_progress_reporter",
    default=None,
)


def set_progress_reporter(reporter: ProgressReporter) -> Token[ProgressReporter | None]:
    return _current_progress_reporter.set(reporter)


def reset_progress_reporter(token: Token[ProgressReporter | None]) -> None:
    _current_progress_reporter.reset(token)


async def report_subllm_progress(**increments: int) -> None:
    reporter = _current_progress_reporter.get()
    if reporter is None:
        return

    await reporter(increments)
