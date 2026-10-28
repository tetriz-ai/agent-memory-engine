# Examples

| You want | Example |
|---|---|
| To see which sessions a run would read, before spending anything | [list-sessions-in-a-date-range](list-sessions-in-a-date-range/) |
| To pick memories by tag for the task at hand | [find-memories-by-tag](find-memories-by-tag/) |
| To use your own model backend | [bring-your-own-provider](bring-your-own-provider/) |
| To see the whole pipeline and one memory in full, without an API key | [extract-from-a-recorded-session](extract-from-a-recorded-session/) |

To run one, clone the repo and, from its root, run `uv run python examples/<name>/main.py`. The
shell example runs as `uv run examples/list-sessions-in-a-date-range/run.sh`.

uv installs the package into the clone first. A `uv tool install` copy cannot be imported.

`scripts/run-examples.sh` builds the package, installs the wheel in a fresh virtual environment,
and runs every example's command from the repo root with an empty temporary `HOME`. No example
calls a model or the network.

The examples use synthetic sessions and hand-written or rule-based answers instead of a model. The runner compares each
example's standard output with the last block of its README, byte for byte.
