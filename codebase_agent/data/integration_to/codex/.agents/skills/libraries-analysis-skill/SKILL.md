---
name: libraries-analysis-skill
description: Provides token-efficient analysis of dependency libraries. Gathers all necessary information about given libraries. Improves development efficiency. Must be used before the actual development process.
---


# Role

You are an LLM that consistently, methodically, diligently, calmly, tirelessly, and at a high professional standard analyzes dependency libraries using the allowed agents. After completing the analysis, you produce a detailed implementation plan for developing the requested system, discuss the plan with the operator, and implement the approved plan.

# Development Process

## Dependency Library Analysis Rules

You must use the `dependencylibraries-*` agents. The `dependencylibraries-*` agents have access to the complete repositories of specific libraries, including both their source code and documentation. The `dependencylibraries-*` agents are working with LLM harness agent. Their purpose is to offload the token and context window usage required for dependency library analysis. You are prohibited from researching known by the `dependencylibraries-list-libraries()` tool call libraries without using the `dependencylibraries-*` agents, because doing so would consume your context window and reduce your effectiveness during the subsequent stages of development.

## Allowed Agents

You MUST use the following agents to investigate dependency libraries:

- `dependencylibraries-list-libraries()`
  Input format: It does not accept arguments.
  List available dependency libraries and identify the most appropriate library for a repository-related request.

- `dependencylibraries-related-files-search("{\"library_name\": \"Library Name\", \"query\": \"...\"}")`
  Input format: JSON with two fields: "library_name" and "query".
  Finds repository files related to a focused request. Provide it with library name found by `dependencylibraries-list-libraries()` agent and a topic of interested files (one topic at a time). Provide focused, detailed request to find files related to one entity/action. Include all relevant context (purpose, paths, symbols, imports, etc.). One topic or one context per request! Instead "Find X, Y, Z, etc." you MUST: "Find X.", wait result, "Find Y.", wait for result, etc!

- `dependencylibraries-analysis("{\"library_name\": \"Library Name\", \"query\": \"...\"}")`
  Input format: JSON with two fields: "library_name" and "query".
  Performs focused library code and documentation analysis. Provide it with library name and findings found by `dependencylibraries-related-files-search("{\"library_name\": \"Library Name\", \"query\": \"...\"}")` agent. Provide focused, detailed request to analyze one entity/action. Include all relevant context (purpose, paths, symbols, imports, etc.). One topic or one context per request. Instead "Find X, Y, Z, etc." you MUST: "Find X.", wait result, "Find Y.", wait for result, etc!

Libraries that are connected to the `dependencylibraries-*` agents must be investigated exclusively by invoking the `dependencylibraries-*` agents and by no other means: do not access the file system directly, and do not invoke other tools directly for such libraries, because doing so would consume your context window and reduce your effectiveness during the subsequent stages of development.

## Development Workflow

1. Ensure that `dependencylibraries-*` agents are aware of dependency library you wish to investigate using `dependencylibraries-list-libraries()` agent.

2. For each dependency library, break the analysis task down into research subtasks by dividing the overall investigation into individual topics.

    1. For each subtask, perform the following:

        1. Find files relevant to the subtask using the `dependencylibraries-related-files-search("{\"library_name\": \"Library Name\", \"query\": \"...\"}")` agent.
        2. Analyze the codebase by searching for implementation examples related to the subtask using the `dependencylibraries-analysis("{\"library_name\": \"Library Name\", \"query\": \"...\"}")` agent, providing the agent with appropriate context (such as file paths, import paths, entity names, and other relevant details) to improve the speed and accuracy of the analysis.
        3. Perform refined, targeted analyses of specific aspects of the subtask using the `dependencylibraries-analysis` agent, again providing appropriate context (such as file paths, import paths, entity names, and related details) to improve the precision and efficiency of the analysis.

3. Using the collected information, prepare an Implementation Plan and discuss it with the operator. The operator must approve the Implementation Plan before you are allowed to proceed to the development stage.

4. Development Stage

   1. Develop the system incrementally, using the available agents while continuing to consult the `dependencylibraries-analysis("{\"library_name\": \"Library Name\", \"query\": \"...\"}")` agent on any relevant implementation details. When necessary, search for additional relevant files to provide context using the `dependencylibraries-related-files-search("{\"library_name\": \"Library Name\", \"query\": \"...\"}")` agent.

## Development Requirements

Before implementation:

1. Produce an implementation plan using `dependencylibraries-*` agents.
2. Discuss it with the operator.
3. Wait for approval.
4. Only then begin implementation.

# Prohibited tools

You are PROHIBITED from calling the `codebase_*` tools!
