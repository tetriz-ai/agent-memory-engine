# agent-memory-engine

## What this is

A CLI and Python package that reads local Claude Code sessions and extracts typed, tagged memories
with Gemini or the local `claude` CLI, each backed by a quote code checks against the session. It does not
write memories into any agent, read other agents' sessions, or run on Windows.

## Layout

- `src/agent_memory_engine/`: `cli.py` (four commands), `pipeline.py` (arms, then merge),
  `extract.py` (windows, extraction, reconcile), `merge.py` and `cluster.py` (the merge),
  `contract.py` (format, evidence gate, responseSchema), `transcript.py` (session to turns, with
  authorship), `sources/claude_code.py` (session selection), `providers/`, `prompts/`, `schema/`.
- `tests/`: pytest, a synthetic session home in `tests/fixtures/claude-home/`, a scripted model in
  `tests/scripted.py`. `examples/`: four runnable examples. `docs/format/`: the memory fields.

## Commands

Python 3.12 is the floor. The configured CI matrix covers 3.12, 3.13, 3.14 and 3.15 on Linux
and 3.15 on macOS, with uv 0.13.5. Actions is disabled; run checks locally.
ruff 0.16.9 and mypy 2.3.1 are pinned in the `dev` group, so `uv run` gives CI's versions.

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
scripts/run-examples.sh
scripts/check-package.sh
uv run python tests/fixtures/make_storefront_session.py --check
```

## Contract to keep

- A memory survives only if every quote is in the cited turn, said by the cited speaker, and the
  stored quote is the turn's own text (`tests/test_contract.py`, `EvidenceGate`).
- Tool, shell and harness output in the user slot is assistant context, and a machine-sent turn is
  never human prose (`tests/test_contract.py`, `ParsedAuthorship`, `MixedToolResultAndText`;
  `tests/test_sources.py`).
- `review` comes from the type alone, and an `intention` is always `hold-for-review`, also after
  reconcile and merge (`tests/test_contract.py`, `ScopeAndIntentionShape`; `tests/test_merge.py`,
  `ReviewIsStampedFromType`).
- Every merged memory validates against `schema/memory.schema.json`, and the schema's keys equal
  what `merge.build_final` emits (`tests/test_merge.py`, `OutputSchemaMatchesCode`;
  `tests/test_pipeline.py`).
- Nothing is a silent zero: skipped sessions, unparseable windows and gate drops are counted in
  `metrics.json` (`tests/test_extract.py`, `UnparseableWindowsAreCounted`; `tests/test_cli.py`).
- Spend is accounted before a cache write, on every stop and for a failed call
  (`tests/test_extract.py`, `APaidCallSurvivesLocalFailure`, `ConversationParallelFatalStop`;
  `tests/test_provider_claude.py`, `Failures`).
- Output order and content do not depend on `--workers` (`tests/test_extract.py`,
  `StreamingSchedulerDeterminism`).
- Clustering gives exactly the groups of plain average linkage (`tests/test_cluster.py`).
- The data dir is owner-only and refused inside a git repo (`tests/test_pipeline.py`).
- The cache identity of the default configuration is pinned (`tests/test_contract.py`,
  `test_schema_digest_is_pinned`; `tests/test_provider_gemini.py`, `CacheIdentity`).
- The supported API is the CLI, `agent_memory_engine`, and from `agent_memory_engine.providers`
  only `ModelClient`, `ExtractResult`, `GeminiClient` and `ClaudeCLIClient`. The rest is internal.
  Commands exit 0, 1 for a stopped run, 2 for a usage error (`tests/test_cli.py`).

## Making a change

- Write the test first: it fails before the change and passes after. Put a case next to the class
  that tests the same module. Use `tests/scripted.py` or a stub client, never a real provider.
- A user-visible change adds a line under `[Unreleased]` in `CHANGELOG.md`. Refactors need none.
- Do not silence ruff or mypy inline. A rule that must be off goes in `pyproject.toml`, commented.
- `scripts/run-examples.sh` builds the wheel, installs it alone, and runs each example's `sh` block
  from the repo root with an empty `HOME`. It compares stdout with the README's last `text` block
  byte for byte and checks the index. Keep example output free of paths, times and random ids.
- A new example is a folder `examples/<task-in-words>/` with a README, and a row in the index
  (`examples/README.md`). The row links the folder as `[name](name/)`, and the runner reads only
  names matching `[a-z0-9-]+`.
- Each example is self-contained: it reads only files in its own folder. Its README follows
  `examples/find-memories-by-tag/README.md`: what it does, how to run it from a clone, the command,
  then the exact output.
- `tests/test_readme.py` pins the README's commands and output to the examples, the `docs/usage.md`
  blocks to a real run, and the examples' data copies to `tests/fixtures/`. Change both sides together.

## Do not

- Do not run a test or an example against `~/.claude` or a real key. `tests/conftest.py` blocks the
  network and subprocesses. Keep it that way.
- Do not edit the text between `===BEGIN` and `===END` in `prompts/` casually: it changes model
  output, cost and every cache key.
- Do not change `extract.ARMS` prompt versions, `SEGMENTATION`, or the cache-id suffix order in
  `providers/gemini.py`: that orphans every cached answer.
- Do not add a runtime dependency: the package is standard library only.
