# Install

agent-memory-engine is a Python package and CLI for macOS and Linux. It needs Python 3.12 or newer
and has no runtime dependency outside the standard library. The package name is
`tetriz-agent-memory-engine`, which is also its unique command name.

The public repository is [tetriz-ai/agent-memory-engine](https://github.com/tetriz-ai/agent-memory-engine).

## From the GitHub release

Install the CLI as an isolated tool with [uv](https://docs.astral.sh/uv/).

```sh
uv tool install https://github.com/tetriz-ai/agent-memory-engine/releases/download/v0.1.0/tetriz_agent_memory_engine-0.1.0-py3-none-any.whl
tetriz-agent-memory-engine --help
```

For the Python API, install the same wheel in a Python 3.12+ virtual environment.

```sh
pip install https://github.com/tetriz-ai/agent-memory-engine/releases/download/v0.1.0/tetriz_agent_memory_engine-0.1.0-py3-none-any.whl
```
`tetriz-agent-memory-engine` is this package and its only command.
The Python import name is `agent_memory_engine`.

## From source

You need Git and [uv](https://docs.astral.sh/uv/) to clone the public repository.
uv supplies Python 3.12 or newer when needed.

```sh
git clone https://github.com/tetriz-ai/agent-memory-engine
cd agent-memory-engine
uv sync --locked
uv run tetriz-agent-memory-engine --help
```

Run `uv run tetriz-agent-memory-engine` from this checkout to use the CLI, and `uv run python` to use the
Python API. [Quick start](../README.md#quick-start) runs a synthetic example without a model account.
[CONTRIBUTING.md](../CONTRIBUTING.md) lists the checks for source changes.

## Install the command from your checkout

To run it outside the checkout, install it as an isolated tool while in the repository root.

```sh
uv tool install .
tetriz-agent-memory-engine --help
```

uv must have its tool directory on your `PATH`. If the command is not found, run `uv tool update-shell`
and start a new shell. This tool installation does not make the package importable by other Python
projects. For embedding, install the checkout into your project's Python 3.12+ virtual environment.

## What a run needs

- **Your sessions:** Claude Code keeps them in `~/.claude`. `--claude-home` points elsewhere.
- **Gemini:** provide a [Gemini API key](https://aistudio.google.com/app/apikey) in `GEMINI_API_KEY`.
- **Claude:** select `--provider claude` with a logged-in CLI supporting `--no-session-persistence`.

`sessions` reads local files and needs no model account. Extraction sends transcript text to your
chosen provider, including any secrets in it. A local `claude` executable does not provide local
inference. [Usage](usage.md) covers selection, provider costs and output storage.
