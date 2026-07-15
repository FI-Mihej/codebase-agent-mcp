![GitHub tag (with filter)](https://img.shields.io/github/v/tag/FI-Mihej/codebase-agent-mcp) ![Static Badge](https://img.shields.io/badge/OS-Linux_%7C_Windows_%7C_macOS-blue)

![Static Badge](https://img.shields.io/badge/wheels-Linux_%7C_Windows_%7C_macOS-blue) ![Static Badge](https://img.shields.io/badge/Architecture-x86__64_%7C_ARM__64-blue)

![GitHub License](https://img.shields.io/github/license/FI-Mihej/codebase-agent-mcp?color=darkgreen) ![Static Badge](https://img.shields.io/badge/API_status-Stable-darkgreen)

[![codebase-agent-mcp MCP server](https://glama.ai/mcp/servers/FI-Mihej/codebase-agent-mcp/badges/score.svg)](https://glama.ai/mcp/servers/FI-Mihej/codebase-agent-mcp)

# CodebaseAgent-MCP

CodebaseAgent-MCP is a token-efficient MCP server for AI coding agents that delegates large codebase analysis to a dedicated OpenAI-compatible LLM, reducing context size, latency, and token costs. Instead of forcing your primary coding assistant to repeatedly scan large codebases, it performs that work separately and returns only the information needed for the current task.

It can run against local models such as Gemma or Qwen, as well as inexpensive OpenAI-compatible cloud providers, reducing both latency and token consumption while keeping the primary assistant focused on reasoning and code generation.

Optional Qdrant integration can cache previous retrieval results today and is planned to evolve into semantic retrieval of code entities (files, classes and functions) from connected codebases.

## Why

Modern coding agents repeatedly spend thousands (sometimes millions) of tokens re-reading repositories, dependency sources, and documentation before they can start solving the actual task.

This becomes even more expensive when:

- the project is actively evolving;
- documentation is incomplete or outdated;
- source code must be inspected directly;
- each new agent session starts from an empty context.

CodebaseAgent-MCP delegates this exploration to a dedicated analysis model. The primary coding assistant receives only the relevant findings instead of repeatedly processing the entire codebase.

## How it works

1. Connect your local or low-cost cloud LLM to CodebaseAgent-MCP.
2. Connect CodebaseAgent-MCP as an MCP-server to your harness (ClaudeCode, Codex, OpenCode, etc.).
3. Start your development prompt with the phrase "Use skill `libraries-analysis-skill`." to reduce your costs.

## Benchmark

### Task

[task.md](example/prompts/cengal_based_system_of_apps/task.md) - write two small applications with a combined size of approximately 200 lines of code that use the [Cengal](https://github.com/FI-Mihej/Cengal) library (inter-process communication, async multiprocessing app with TUI, async wxPython GUI).

In real-world development, the number of output tokens is also relatively small compared to the amount of project data that must be reread every session and kept in the context window, consuming input tokens. The scale of both the work and the associated costs is simply much larger.

### Measurements

(VectorDB caching in CodebaseAgent-MCP was disabled to evaluate performance under the worst possible conditions.)

Tokens spent during the search and analysis stage of the Cengal codebase:

| Approach               | All Input tok. | Input tok. | Cached Input tok. | Output tok. |
| ---------------------- | -------------- | ---------- | ----------------- | ----------- |
| without                | 791930         | 684087     | 107843            | 5361        |
| With CodebaseAgent-MCP | 416160         | 229003     | 187157            | 13755       |

Estimated costs based on the current token pricing of various models:

| Approach               | Claude Haiku 4.5 | Claude Sonnet 5 (Sep 1, 2026) | Claude Opus 4.8 | Claude Fable 5 | GPT-5.5  | GPT-5.5-Pro * |
| ---------------------- | ---------------- | ----------------------------- | --------------- | -------------- | -------- | ------------- |
| without                | 0.89 USD         | 2.68 USD                      | 4.46 USD        | 8.93 USD       | 3.64 USD | 24.72 USD     |
| With CodebaseAgent-MCP | 0.37 USD         | 1.12 USD                      | 1.87 USD        | 3.74 USD       | 1.65 USD | 14.96 USD     |
| Cost Reduction (×)     | 2.4              | 2.39                          | 2.39            | 2.39           | 2.2 *    | 1.65          |

* GPT Pro models do not support cached tokens.
* I used a conservative (minimum) estimate for cached tokens. OpenAI dynamically accounts for cached tokens for GPT models (except Pro), meaning significantly more tokens are actually billed as cached. As a result, the real advantage of using CodebaseAgent-MCP is greater than "2.2×" because the proportion of cached tokens is higher when using CodebaseAgent-MCP than without it.

When VectorDB caching (`qdrant_*` plugins) is enabled in CodebaseAgent-MCP, the savings become even greater.

### Harness

The benchmark was conducted using the OpenCode harness because it provides detailed statistics for both the main agent session and all subagent sessions.

### LLM

The `google/gemma-4-12b-qat` model was used on both the OpenCode harness side and the CodebaseAgent-MCP harness side.

### Prompts

* [with__codebase_agent__agents.md](examples/prompts/cengal_based_system_of_apps/opencode/with__codebase_agent__agents.md)
  * The MCP server for CodebaseAgent-MCP was the only one connected to OpenCode, with agents and skills installed in the project dir.
* [without__codebase_agent__subagents.md](examples/prompts/cengal_based_system_of_apps/general/without__codebase_agent__subagents.md)
  * All MCP servers were subsequently disconnected (disabled) from OpenCode, and no agents or skills remain installed.
  * The phrase "Spawn subagents." is, of course, a significant advantage for plain OpenCode. I ended up using it out of necessity because, without it, clean OpenCode would consistently fall into an infinite loop: "Conduct research -> write one or two files until the context window is full -> delete part of the conversation history instead of summarizing it -> start over". At the same time, the CodebaseAgent-MCP server connected to OpenCode completes the task successfully even without any installed skills or agents, though it does so slightly less efficiently than with them.
  

### Cached Input

Every provider implements caching differently, and some do not support it at all. I chose a conservative accounting method that produces the minimum possible number of cached tokens to avoid overstating the results. In particular, the actual number of cached tokens with OpenAI would be approximately 1.2-1.5× higher than my calculations because of OpenAI's dynamic caching algorithm. As a result, the advantage of using CodebaseAgent-MCP with GPT-5.5 would likely be comparable to, or even greater than, the savings observed with Anthropic models.

## Architecture

```text
ClaudeCode + Opus -> CodebaseAgent-MCP -> OpenAI-compatible LLM (either local or cloud)
                         |
                         +-> configured local codebases
                         +-> built-in tools and plugins
                         +-> optional Qdrant cache
                         +-> external MCP plugins (any MCP-servers of your choice)
```

## Features

- Token-efficient code and dependency analysis for repositories whose files are larger than the connected model's context window.
- Automatic conversation-history compression
- Optional RAG cache through one of the `qdrant_*` built-in plugins.
- Sandboxed filesystem access scoped to configured library roots.
- Pluggable external stdio MCP tools.
- Works with local OpenAI-compatible servers such as LM Studio, llama.cpp servers, vLLM-compatible endpoints, or compatible hosted APIs.
- Works with cloud OpenAI-compatible servers.
- Async background jobs with SQLite persistence of results.

## How-To Start

1. Install
2. Configure
3. Connect to your coding agent
4. Use CodebaseAgent-MCP

## Installation

1. Install `uv`:
   https://docs.astral.sh/uv/getting-started/installation/


2. Initialize and create the CodebaseAgent-MCP configuration via `uvx`:

```bash
uvx --from codebase-agent-mcp cb-agent-init
```

It will return the path to your `codebase_agent.config.json` configuration file.

Feel free to use `uvx --from codebase-agent-mcp cb-agent-init` at any time to find the location of your configuration file.

### Update to latest version

```bash
uvx --from codebase-agent-mcp@latest cb-agent-init
```

This will not affect your config.

## Configuration

Technical details and configuration recommendations for local LLMs: [docs](docs/main.md)

Edit `codebase_agent.config.json` before starting the server.

### Minimal config

Define OpenAI-compatible LLM (either local or cloud)

| Field | Purpose |
| --- | --- |
| `base_url` | OpenAI-compatible endpoint. `/v1` is appended automatically when omitted. |
| `api_key` | API key sent to the endpoint. Use a placeholder such as `[EMPTY]` for local servers that do not require a key. |
| `model` | Model name exposed by the OpenAI-compatible server. |
| `reasoning_allowed` | When `false`, the client sends provider additional hints that disable thinking/reasoning where supported. |
| `reasoning_effort` | Reasoning effort value sent with the request. Use `none` for models/endpoints that support it. |
| `max_tokens` | Model context window limit in tokens. |

Local repositories, dependency sources, or documentation trees that the harness may analyze.

| Field | Purpose |
| --- | --- |
| `name` | Public name used as `library_name` in MCP tool calls. Names must be unique. |
| `allowed` | Enables or disables the library. Disabled libraries are not listed or analyzed. |
| `path` | Absolute path to the local directory. The path must exist when enabled. |
| `instructions` | Extra guidance for this codebase, such as preferred APIs, documentation folders, or project conventions. |

### Qdrant (Optional)

CodebaseAgent-MCP works as a client to [Qdrant](https://github.com/qdrant/qdrant): either local or cloud.

`qdrant_fastembed`, and `qdrant_cloud` enable the RAG cache. By default, `qdrant_fastembed` client is installed. Their `configuration` can contain:

| Field | Purpose |
| --- | --- |
| `model_name` | Embedding model name. Defaults to `sentence-transformers/all-MiniLM-L6-v2` when omitted. |
| `init` | Keyword arguments passed to `qdrant_client.QdrantClient`, such as `url`, `api_key`, or `cloud_inference`. |

> Before the first use, and after every change to the `"configuration"."model_name"` field in the `qdrant_*` plugin configuration, it is necessary to initialize (download) the model before the next use of the MCP server. The procedure is described below in the "Usage" -> "Qdrant (Optional)" section.

## Connection to ClaudeCode/Codex/etc.

### Register as MCP-server

2. Configure your MCP client (ClaudeCode/Codex/OpenCode/Hermes/PiAgent/etc.) to run CodebaseAgent-MCP via `uvx`:

```json
{
  "mcpServers": {
    "codebase-agent-mcp": {
      "command": "uvx",
      "args": [
        "codebase-agent-mcp"
      ]
    }
  }
}
```

### Install subagents and skills to your harness (ClaudeCode/Codex/etc.)

Go to the root directory of your project and run:

```bash
uvx --from codebase-agent-mcp cb-agent-install-skills-to-current-dir
```

Alternatively, you may clone the repository using `git clone https://github.com/FI-Mihej/codebase-agent-mcp.git` and proceed manually:

* ClaudeCode: copy `./codebase_agent/data/integration_to/claude/.claude` to root dir of your project.
* Codex: copy `./codebase_agent/data/integration_to/codex/.agents`, `./codebase_agent/data/integration_to/codex/.codex` and `./codebase_agent/data/integration_to/codex/.codex/config.toml` to root dir of your project.
* OpenCode: copy `./codebase_agent/data/integration_to/opencode/.opencode` to root dir of your project.
* Cursor: copy `./codebase_agent/data/integration_to/cursor/.cursor` to root dir of your project.
* Antigravity: copy `./codebase_agent/data/integration_to/antigravity/.agents` to root dir of your project. Antigravity lacks an agent concept, which means the work will be less token-efficient than when using other harnesses. Be sure to enable Implicit Caching to achieve significant savings.
* Hermes Agent: copy `./codebase_agent/data/integration_to/hermes/skills` to root dir of your project.
* Pi Coding Agent: 1. install `https://github.com/nicobailon/pi-subagents` or similar solution; 2. copy `./codebase_agent/data/integration_to/pi_agent/.pi` to root dir of your project.

## Usage

Start your development prompt with the phrase "Use skill `libraries-analysis-skill`."

Example prompt: [with__codebase_agent__skills.md](examples/prompts/cengal_based_system_of_apps/general/with__codebase_agent__skills.md)

### Qdrant (Optional)

#### Prepare models

Before the first run of the CodebaseAgent-MCP with the `qdrant_*` plugin(s) enabled, and after every change to the `"configuration"."model_name"` field in the `qdrant_*` plugin configuration, always run:

```bash
uvx --from codebase-agent-mcp cb-agent-ensure-qdrant-models
```

#### Index dependecy libraries (to be done)

Perform indexing of dependency library codebases to add the key features of individual entities (files, classes, functions) to the RAG storage.

```bash
uvx --from codebase-agent-mcp cb-agent-index-dependency-libraries
```

### Protecting Against Prompt Injections in Dependency Library Code

Embedding prompt injections into repository code is becoming increasingly widespread. This ranges from repositories maintained by Meta (`github.com/facebook/*`), where they mainly interfere with coding agents but are otherwise harmless, to genuinely dangerous cases that can lead to credential leaks, Social Security number exposure, financial losses, and other security incidents.

A few simple yet still effective examples:

* [They're Poisoning the Agents! (by The PrimeTime)](https://www.youtube.com/watch?v=bh6S4N8TnYQ)
* Prompt injection: [github.com/facebook/docusaurus](https://github.com/facebook/docusaurus/blob/main/AGENTS.md)

  * Result against Claude Code: [PR #12105](https://github.com/facebook/docusaurus/pull/12105)
  * [Post on X by a Meta core developer](https://x.com/mitchellh/status/2067970516951150721), where the author celebrates the remarkably high effectiveness of the prompt injections they embedded in `AGENTS.md`, code comments, and other locations throughout the repository.
* Prompt injection: [github.com/ghostty-org/ghostty](https://github.com/ghostty-org/ghostty/blob/main/AGENTS.md)

  * The same approach, with similar results.

Real-world prompt injections use millions of effective wording variations, with new ones being created constantly.

* Using regular expressions or other primitive techniques to defend against them is ineffective.
* Guardrail models. Even the best guardrail models achieve robustness of only around 85%. That means approximately one out of every six attacks succeeds. Is `that` an acceptable level of protection for a production system? See: [Evaluating the Robustness of Large Language Model Safety Guardrails Against Adversarial Attacks](https://arxiv.org/abs/2511.22047v1), [Bag of Tricks for Subverting Reasoning-based Safety Guardrails](https://arxiv.org/abs/2510.11570), etc.
* Naive LLM-based detection. There is now a substantial body of research arguing that using an LLM to detect prompt injections by prompting the same (or a similar) LLM is fundamentally unreliable due to vulnerabilities in the detector itself and unacceptably high false positive and/or false negative rates for production use. See: [How Not to Detect Prompt Injections with an LLM (2025)](https://arxiv.org/abs/2507.05630), [WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents](https://arxiv.org/abs/2510.01354), [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://arxiv.org/abs/2310.12815), [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710), etc.

#### Solution and Tool (to be done)

After adding a new dependency library (that is, adding a new entry to the `libraries` field in the configuration file), it is recommended to sanitize the dependency library codebases by removing prompt injections from them.

A dedicated tool for this purpose will be released very soon. Stay tuned for updates.

It will be launched similarly to the following:

```bash
uvx --from codebase-agent-mcp cb-agent-sanitize-library-codebases
```

# Github repository

Github repository is a curated public mirror of the project. Active development (including experimental code and private research notes) happens in a private repository; selected snapshots are published here periodically.

## Roadmap

* Support for the `qdrant_fastembed_gpu` plugin.
* Internal optimizations and an expanded set of tools.
* Integration of a content sanitization system for prompt injection protection.
* A configuration field for LLM instructions on how to use connected MCP servers.
* An internal sub-agent hierarchy for faster LLM operation.

# Glama.AI

[![codebase-agent-mcp MCP server](https://glama.ai/mcp/servers/FI-Mihej/codebase-agent-mcp/badges/card.svg)](https://glama.ai/mcp/servers/FI-Mihej/codebase-agent-mcp)

# Cengal

Based on [Cengal](https://github.com/FI-Mihej/Cengal)

## Projects using Cengal

* [text_file_read_and_refactor_mcp](https://github.com/FI-Mihej/text_file_read_and_refactor_mcp) - Token-efficient Python stdio MCP server exposing safe text-file search, reading, and refactoring tools. Tools automatically resolve the file BOM and codepage. 
* [InterProcessPyObjects](https://github.com/FI-Mihej/InterProcessPyObjects) - High-performance package delivers blazing-fast inter-process communication through shared memory, enabling Python objects to be shared across processes with exceptional efficiency. 
* [cengal_app_dir_path_finder](https://github.com/FI-Mihej/cengal_app_dir_path_finder) - A Python module offering a unified API for easy retrieval of OS-specific application directories, enhancing data management across Windows, Linux, and macOS 
* [cengal_cpu_info](https://github.com/FI-Mihej/cengal_cpu_info) - Extended, cached CPU info with consistent output format.
* [cengal_memory_barriers](https://github.com/FI-Mihej/cengal_memory_barriers) - Fast cross-platform memory barriers for Python.
* [Bensbach](https://github.com/FI-Mihej/Bensbach) - decompiler from Unreal Engine 3 bytecode to a Lisp-like script and compiler back to Unreal Engine 3 bytecode. Made for a game modding purposes
* [Realistic-Damage-Model-mod-for-Long-War](https://github.com/FI-Mihej/Realistic-Damage-Model-mod-for-Long-War) - Mod for both the original XCOM:EW and the mod Long War. Was made with a Bensbach, which was made with Cengal

# License

Copyright © 2026 ButenkoMS. All rights reserved.

Licensed under the Apache License, Version 2.0.
