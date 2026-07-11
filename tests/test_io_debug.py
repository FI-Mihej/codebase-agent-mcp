from __future__ import annotations

import json
from typing import Any

import anyio

from codebase_agent.io_debug import IODebugLogger, logged_io_streams


def test_logged_io_streams_writes_jsonl_and_forwards_messages(tmp_path) -> None:
    log_path = tmp_path / "io.jsonl"

    async def run() -> None:
        inbound_write, inbound_read = anyio.create_memory_object_stream[Any](0)
        outbound_write, outbound_read = anyio.create_memory_object_stream[Any](0)
        logger = IODebugLogger(log_path)

        async with logged_io_streams(
            inbound_read,
            outbound_write,
            logger=logger,
            path="client-server",
            inbound_direction="client_to_server",
            outbound_direction="server_to_client",
        ) as (logged_read, logged_write):
            await inbound_write.send({"jsonrpc": "2.0", "id": 1})
            assert await logged_read.receive() == {"jsonrpc": "2.0", "id": 1}

            await logged_write.send({"jsonrpc": "2.0", "id": 1, "result": {}})
            assert await outbound_read.receive() == {"jsonrpc": "2.0", "id": 1, "result": {}}

    anyio.run(run)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["direction"] for record in records] == [
        "client_to_server",
        "server_to_client",
    ]
    assert {record["path"] for record in records} == {"client-server"}
    assert records[0]["payload"] == {"jsonrpc": "2.0", "id": 1}
    assert records[1]["payload"] == {"jsonrpc": "2.0", "id": 1, "result": {}}
