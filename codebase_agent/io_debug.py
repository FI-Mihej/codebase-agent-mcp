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


"""JSONL IO debug logging for MCP stdio streams."""


from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


class IODebugLogger:
    """Append MCP stream messages to a JSONL debug file."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._lock = anyio.Lock()

    async def log(self, *, path: str, direction: str, payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": path,
            "direction": direction,
            "payload": _payload_to_json(payload),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        async with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


@asynccontextmanager
async def logged_io_streams(
    read_stream: MemoryObjectReceiveStream[Any],
    write_stream: MemoryObjectSendStream[Any],
    *,
    logger: IODebugLogger,
    path: str,
    inbound_direction: str,
    outbound_direction: str,
) -> AsyncIterator[tuple[MemoryObjectReceiveStream[Any], MemoryObjectSendStream[Any]]]:
    """Proxy memory streams while logging inbound and outbound messages."""

    logged_read_writer, logged_read = anyio.create_memory_object_stream[Any](0)
    logged_write, logged_write_reader = anyio.create_memory_object_stream[Any](0)

    async def read_pump() -> None:
        try:
            async with logged_read_writer:
                async for item in read_stream:
                    await logger.log(path=path, direction=inbound_direction, payload=item)
                    await logged_read_writer.send(item)
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async def write_pump() -> None:
        try:
            async with logged_write_reader:
                async for item in logged_write_reader:
                    await logger.log(path=path, direction=outbound_direction, payload=item)
                    await write_stream.send(item)
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(read_pump)
        task_group.start_soon(write_pump)
        try:
            yield logged_read, logged_write
        finally:
            await logged_read.aclose()
            await logged_write.aclose()
            task_group.cancel_scope.cancel()


def _payload_to_json(payload: Any) -> Any:
    if isinstance(payload, Exception):
        return {
            "exception_type": type(payload).__name__,
            "message": str(payload),
        }

    message = getattr(payload, "message", None)
    if message is not None and hasattr(message, "model_dump"):
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json", by_alias=True, exclude_none=True)

    return payload
