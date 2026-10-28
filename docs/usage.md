# Usage

Flags, the output layout and exit codes are in [reference.md](reference.md).
The commands below assume you [installed the CLI](install.md#from-the-github-release).
When working directly in the checkout, prefix `tetriz-agent-memory-engine` with `uv run`.

## Run it again

Run the same `run` command again whenever you want the newer sessions, or move `--since` forward.
Each run writes a new directory, `~/.agent-memory-engine/runs/<run-id>/`, named for the second it
started.

Every model answer is also kept in `~/.agent-memory-engine/cache/`. A later run with the same
provider and settings replays the answers it already has. It calls the model only for windows it
has not seen and for merge steps whose input changed.

The `metrics.json` files under `arms/` count fresh calls as `llm_calls` and replayed ones as
`cache_hits`.

## Run several at once

To start runs in parallel, give each its own `--run-id`. A second run started in the same second
without one exits with an error, because both would want the same directory.

## Cost

A run pays only for new calls and for holding the prompt cache (`explicit_cache_storage_usd`). When
it replays answers, each cost line shows what it spent beside what those answers cost at the
current price card, such as `$0.00 spent (cached answers worth $0.11)`. With `--provider claude`, a
replayed answer counts at the cost the CLI recorded for it, not at a price card.

At Gemini's price card as read on 2026-08-26, the default settings cost 1.8 to 2.9 cents per
window, and long sessions cost the most. The `claude` provider is not measured, and a claude run
prints the cost the `claude` CLI reports.

[measurements.md](measurements.md#cost) says why long sessions cost more, and how these were
measured. Each run records the rates it used, as
[reference.md](reference.md#prices-a-run-used) describes.

## Choose sessions

`sessions`, `run` and `extract` take the same selection flags, listed in
[reference.md](reference.md#session-selection). Run `sessions` first to see what a selection
matches. It calls no model and needs no API key.

```sh
tetriz-agent-memory-engine sessions --since 2026-09-10 --until 2026-10-01 --project /home/dev/src/shop
```

On the [sample session home](../examples/list-sessions-in-a-date-range/), it prints this.

```text
DATE        PROJECT                              SESSION                               TURNS  WINDOWS
2026-09-15  /home/dev/src/shop/services/billing  c41f8e22-6b9d-4f3a-8d71-90e5a2b7c013      4        1
1 session, 1 window.
```

A window is the slice of transcript sent to the model in one call, and cost is per window.
`DATE` is the session's local calendar day, the day `--since` and `--until` compare. The sample
output was made with `TZ=UTC`.

`--project` keeps the directory and everything below it. So `/home/dev/src/shop` keeps
`/home/dev/src/shop/services/billing` but not `/home/dev/src/shop-admin`. A session with no
recorded working directory shows its folder name, such as `-home-dev-notes`, and is matched on it.
So `/home/dev/notes` and `/home/dev-notes` can catch each other's sessions.

## Run one arm at a time

`extract` runs or redoes one arm, and `merge` combines the arms. Give each `extract` the same
`--run-id` and selection flags, then `merge` that run id. It writes `runs/<run-id>/memories.jsonl`,
as `run` does.

```sh
tetriz-agent-memory-engine extract --arm facts --since 2026-09-01 --run-id shop-sept
tetriz-agent-memory-engine extract --arm preferences --since 2026-09-01 --run-id shop-sept
tetriz-agent-memory-engine merge --run-id shop-sept
```

`merge` calls the model too, so it needs the same key or `--provider`.

## Use the Python API

The Python API selects sessions and runs the pipeline from code. In a clone, `uv run python`
imports the package. In your own project, install the source checkout into your project's virtual environment as [install.md](install.md)
describes, since a `uv tool install` copy cannot be imported.

The first snippet reads the sample sessions in a clone of this repository and needs no API key. Save
this as `api.py` in the clone's root and run `uv run python api.py`. To read your own sessions, set
`home` to `~/.claude`.

```python
import datetime

from agent_memory_engine import list_sessions

home = "examples/list-sessions-in-a-date-range/claude-home"
sessions = list_sessions(home, since=datetime.date(2026, 9, 1), projects=["/home/dev/src/shop"])
for session in sessions:
    print(session.date[:10], session.project, session.conversation_id)
```

```text
2026-09-03 /home/dev/src/shop cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11
2026-09-15 /home/dev/src/shop/services/billing cc:c41f8e22-6b9d-4f3a-8d71-90e5a2b7c013
```

`conversation_id` is the session id with a `cc:` prefix. To extract, append this to `api.py` and
run it with `GEMINI_API_KEY` set. It sends those transcripts to Gemini.

```python
import os
from agent_memory_engine import run

result = run(sessions, api_key=os.environ["GEMINI_API_KEY"])
print(result.ok, result.memories_path)
```

Pass `client=` with any `ModelClient` to use another backend, as the
[bring-your-own-provider example](../examples/bring-your-own-provider/) does.
[reference.md](reference.md#a-model-backend-of-your-own) has the details.

## Pick memories by tag

Each line of `memories.jsonl` is one memory, so [jq](https://jqlang.org/) can filter it. The
README's [Pick memories by tag](../README.md#pick-memories-by-tag) filter keeps the confident, active
memories about payments. It leaves out every `intention`, which waits for a human check, and
anything the model marked borderline. Like it, the filter below runs on the sample file in a clone.
Put the path `run` printed in its place for your own.

A quote in a user turn is not always the person's own typing. To keep only memories backed by what
they typed, as [format/provenance.md](format/provenance.md) defines it, add the `provenance` and
`authorship` conditions to that filter:

```sh
jq -r 'select(.review.status == "active" and .disposition.level == "confident" and .provenance.confidence != "low" and all(.evidence[]; .authorship == "human_prose") and .tags.domain == "payments") | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Always reuse the original idempotency key on billing retries.
```

The [find-memories-by-tag example](../examples/find-memories-by-tag/) does the same from Python,
and [format/README.md](format/README.md) lists every field you can filter on.
