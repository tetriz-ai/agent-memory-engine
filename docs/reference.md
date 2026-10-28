# Reference

The exact rules behind each command: providers, flags, session selection, the output layout, exit
codes and the supported API. Every command also documents its flags in `--help`.

## Providers

| Provider | How it authenticates | What was measured |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY`, or `--gemini-key-file PATH` | The default settings below, on two sets of 30 real sessions |
| `claude` | Your local `claude` login | Not measured |

The default is `gemini-3.7-flash` at thinking level `medium`, with explicit prompt caching.
[measurements.md](measurements.md) compares its cost and quality with level `high`.

The `claude` provider sends the same prompts through `claude -p` with no tools, with the model
pinned to `claude-opus-4-8`. It needs a CLI that supports `--no-session-persistence`.

## Model flags

`run`, `extract` and `merge` take these flags. Passing `--thinking-level` or `--gemini-key-file`
with `--provider claude` is a usage error, exit code 2.

| Flag | Rule |
|---|---|
| `--provider NAME` | `gemini` (the default) or `claude` |
| `--thinking-level LEVEL` | Gemini's reasoning level: `minimal`, `low`, `medium` (the default) or `high`. Gemini only. |
| `--gemini-key-file PATH` | Read the Gemini API key from `PATH` instead of `GEMINI_API_KEY`. Gemini only. |
| `--arm NAME` | `extract` only, and required: `facts` or `preferences`. Give both `extract` calls the same `--run-id`. |
| `--model ID` | Use another model instead of the pinned `gemini-3.7-flash` or `claude-opus-4-8`. The measurements cover the pinned Gemini model only. `run`, `extract` and `merge` take it. |
| `--workers N` | Sessions processed at once, at most 6. The default is 6 for `gemini` and 1 for `claude`, and the output does not depend on it. `run` and `extract` take it. |

## Session selection

`sessions`, `run` and `extract` take these flags.

| Flag | Rule |
|---|---|
| `--claude-home DIR` | Sessions are `DIR/projects/<folder>/<session-id>.jsonl`. The default is `~/.claude`. Sub-agent transcripts are not sessions. |
| `--since DATE` | Keep sessions whose first event falls on or after `DATE`, a local calendar day |
| `--until DATE` | Keep sessions whose first event falls before `DATE` |
| `--project PATH` | Keep sessions whose working directory is `PATH` or below it. A session is filed under the first working directory its events record. Repeatable. |
| `--limit N` | Keep only the `N` most recent of the selected sessions |

[Inside a session](#inside-a-session) lists the lines of a session file that are skipped.

## Output

Paths are under `--data-dir`, which is `~/.agent-memory-engine` by default. `--run-id` names the
run directory, and defaults to the time the run started.

| Path | Holds |
|---|---|
| `runs/<run-id>/memories.jsonl` | The merged memories, ranked by how many sessions back each memory |
| `runs/<run-id>/metrics.json` | After `run`, the run summary. After `extract` plus `merge`, the merge's own metrics. |
| `runs/<run-id>/sessions.jsonl` | The selected sessions |
| `runs/<run-id>/arms/<arm>/` | Each arm's own `memories.jsonl` and full `metrics.json` |
| `cache/` | Every model answer, keyed on its exact inputs |

The run summary in `metrics.json` has one entry per stage. Each holds sessions run and skipped,
windows, unparseable windows, memories, cost and the stop reason if any. Cost comes in three
fields, each rounded to the microdollar (6 places), and the run has a `_total` of each:

| Field | Holds |
|---|---|
| `spent_usd` | What the run paid |
| `cached_answers_usd` | Replayed answers, priced at the current price card. With `--provider claude`, at the cost the CLI recorded. |
| `cost_usd` | Both together |

- **Owner-only:** the data dir is created with mode 700.
- **Never in a repo:** a data dir inside a git repository is refused, since outputs quote sessions.
- **Explicit counts:** every failure is counted in a `metrics.json`, never dropped silently.
  [Failure counts](#failure-counts) names each one.

## Exit codes

| Code | Means |
|---|---|
| 0 | Done |
| 1 | The run stopped, for example because a model call failed for good. Partial outputs and `metrics.json` say where. |
| 2 | A usage or setup error before any model call, such as a bad flag, a missing key or `claude` CLI, or a run id that already holds other output |

## Versioning and compatibility

The project follows semantic versioning. The supported API is the command line and the names
exported from `agent_memory_engine`: `list_sessions`, `run`, `SessionRef`, `RunResult`, `ModelClient`
and `ExtractResult`. From `agent_memory_engine.providers`, it covers `ModelClient`, `ExtractResult`,
`GeminiClient` and `ClaudeCLIClient`. Other modules are internal and may change in any release.

Python 3.12 is the floor. The configured CI matrix covers Linux with Python 3.12, 3.13, 3.14 and 3.15, and
macOS with 3.15. Windows is not supported, because the tool uses POSIX file locks. Once 1.0 ships,
the memory format changes only with a new major version. Before 1.0, a minor release may change the
format, the command line or the Python API, and [CHANGELOG.md](../CHANGELOG.md) says how.

## Failure counts

| Count | Where | Means |
|---|---|---|
| `skipped_sessions` | each arm's `metrics.json`, and the run's after `run` | Sessions that could not be read, each with its reason |
| `windows_unparseable` | each arm's `metrics.json`, and the run's after `run` | Model answers with no readable memories list |
| `dropped_evidence` | each arm's `metrics.json` only | Memories that failed the evidence gate |
| `reconcile_unparseable`, `reconcile_overflow` | each arm's `metrics.json`, and the run's after `run` | Memories an arm left unmerged across one session's windows, because the model answer was unreadable or too long |

## Inside a session

Sub-agent sidechain events, summary lines and lines that are not JSON are skipped. A sub-agent's
final report, returned into the main session, is read like any other turn. `list_sessions` raises
`FileNotFoundError` when the home has no `projects` directory.

## A model backend of your own

A backend subclasses `ModelClient` from `agent_memory_engine`, implements
`extract(system_prompt, text)`, and returns an `ExtractResult` with the model's JSON text.
Everything it returns still goes through the evidence gate. Pass it to the Python function
`agent_memory_engine.run(sessions, client=...)`, which uses it for every stage. That function writes
the same run directory as the `run` command, under `data_dir` (default `~/.agent-memory-engine`).

The [bring-your-own-provider example](../examples/bring-your-own-provider/) is a complete backend.

## Prices a run used

Every run records the Gemini rates it used under `price_card`, in each `arms/<arm>/metrics.json`
and in the merge's metrics. Holding the cached prompts is recorded as `explicit_cache_storage_usd`.

In `run`'s summary, each cached prompt is priced once, up to its deletion, and shared between the
stages that use it. A reused client's caches from earlier runs are not counted again. A run that
stops early has already cached the prompts of stages it never reached, and their storage is in
`unused_prompt_cache_usd`, which the totals include.

Once `run` ends, each arm's `metrics.json` carries its share of that storage. The arm files, the
merge stage and `unused_prompt_cache_usd` together add up to what the run paid for storage.

[measurements.md](measurements.md#cost) says how the published costs were computed.
