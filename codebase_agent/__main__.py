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


import os
import argparse


def main():
    parser = argparse.ArgumentParser(prog="codebase-agent-mcp")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio", required=False)
    parser.add_argument("--host", default=None, help="default: 127.0.0.1", required=False)
    parser.add_argument("--port", type=int, default=None, help="default: 8000", required=False)
    args = parser.parse_args()

    os.environ.setdefault("CODEBASE_AGENT_MCP_TRANSPORT", args.transport or "stdio")
    os.environ.setdefault("CODEBASE_AGENT_MCP_HOST", args.host or "127.0.0.1")
    os.environ.setdefault("CODEBASE_AGENT_MCP_PORT", str(args.port or 8000))

    from .server import main as server_main
    return server_main()


if __name__ == "__main__":
    main()
