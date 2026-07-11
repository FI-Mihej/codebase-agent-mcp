## Config

### OpenAI-compatible LLM (either local or cloud)

Connection, model, and tool settings for the LLM used by the harness.

| Field | Purpose |
| --- | --- |
| `base_url` | OpenAI-compatible endpoint. `/v1` is appended automatically when omitted. |
| `api_key` | API key sent to the endpoint. Use a placeholder such as `[EMPTY]` for local servers that do not require a key. |
| `model` | Model name exposed by the OpenAI-compatible server. |
| `temperature` | Sampling temperature for analysis requests. Low values are recommended. |
| `reasoning_allowed` | When `false`, the client sends provider hints that disable thinking/reasoning where supported. |
| `reasoning_effort` | Reasoning effort value sent with the request. Use `none` for models/endpoints that support it. |
| `max_tokens` | Model context window limit in tokens. |
| `context_compression_threshold` | Prompt-token ratio that triggers conversation-history compression during long tool sessions. |
| `timeout_seconds` | Timeout for LLM processing. |
| `max_tool_rounds` | Maximum tool-calling loop rounds. `-1` disables the round limit. |
| `built_in_plugins` | Built-in capabilities made available to the internal model. |
| `mcp_plugins` | External stdio MCP servers exposed as tools to the internal model. |

### Plugin Entries

Both `built_in_plugins` and `mcp_plugins` use the same entry shape.

| Field | Purpose |
| --- | --- |
| `name` | Plugin name. Built-ins include `built_in_fs`, `text_file_read_and_refactor_mcp`, `qdrant_fastembed`, `qdrant_fastembed_gpu`, and `qdrant_cloud`. |
| `allowed` | Enables or disables the plugin without removing its configuration. |
| `denied_tools` | Tool names to hide from the internal model while keeping the plugin enabled. |
| `configuration` | Plugin-specific settings. |

External `mcp_plugins` use stdio launch settings:

| Field | Purpose |
| --- | --- |
| `command` | Executable used to start the MCP server. |
| `args` | Command-line arguments. |
| `env` | Optional environment variables. |
| `cwd` | Optional working directory. |
| `read_timeout_seconds` | Optional read timeout for plugin calls. |

### `libraries`

Local repositories, dependency sources, or documentation trees that the harness may analyze.

| Field | Purpose |
| --- | --- |
| `name` | Public name used as `library_name` in MCP tool calls. Names must be unique. |
| `allowed` | Enables or disables the library. Disabled libraries are not listed or analyzed. |
| `path` | Absolute path to the local directory. The path must exist when enabled. |
| `instructions` | Extra guidance for this codebase, such as preferred APIs, documentation folders, or project conventions. |

### `jobs`

Async job storage and polling behavior.

| Field | Purpose |
| --- | --- |
| `storage_backend` | Job storage backend. Currently `sqlite`. |
| `sqlite_path` | SQLite database path. Relative paths resolve next to `codebase_agent.config.json`. |
| `max_concurrent_jobs` | Maximum queued/running jobs accepted at the same time. |
| `job_ttl_seconds` | Retention time for completed, failed, and cancelled jobs. |
| `max_completed_jobs` | Maximum number of terminal jobs kept after cleanup. |
| `status_update_interval_seconds` | Minimum interval for persisted progress updates. |
| `result_wait_timeout_seconds` | Long-poll wait time for status/result calls. |
| `result_poll_interval_seconds` | Poll interval while waiting for job completion. |

## Tools

The server exposes these MCP tools:

| Tool | Purpose |
| --- | --- |
| `codebase_list_libraries` | Lists enabled library names. |
| `codebase_start_job_related_files_search` | Starts an async job that finds files relevant to one focused request. |
| `codebase_start_job_analysis` | Starts an async job that analyzes one focused entity, behavior, or implementation question. |
| `codebase_get_job_status` | Polls job status. |
| `codebase_get_job_result` | Reads job result, error, or partial output. |
| `codebase_cancel_job` | Cancels a queued or running job. |

Jobs return a `job_id`. Poll the matching status/result tools until the status is `done`, `error`, or `cancelled`.

## Local LLM Configuration Recommedations

### Local models

Make sure that both CodebaseAgent-MCP and your harness configured to use same local model. Alternatively make sure that you have enough of VRAM to hold both different models.

### llama.cpp and LM Studio

For the best performance it is generally recommended to set "Max Concurrent Predictions" (`--parallel` / `-np`) to 1, as a single inference stream usually provides the highest throughput and lowest latency.

### vLLM

For the best performance, the optimal `--max-num-seqs` is typically in the tens or even hundreds. The optimal value depends on your hardware, model, and workload, and should be determined by benchmarking your specific configuration using an LLM serving benchmark.

### max_concurrent_jobs

If `--max-num-seqs` / `--parallel` / `-np` is greater than 1, it makes sense to increase the `"jobs"."max_concurrent_jobs"` parameter in your `codebase_agent.config.json`.
