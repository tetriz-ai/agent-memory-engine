# The memory format

Every line of a run's `memories.jsonl` is one merged memory. It is a single durable thing worth
remembering, learned in a conversation and backed by quotes that code checked against the session. The
packaged JSON Schema, `src/agent_memory_engine/schema/memory.schema.json`, validates every line.

The fields fall into four groups.

| Group | Fields |
|---|---|
| What it says | [`statement`, `description`, `memory_id`](statement.md)<br>[`type`, `secondary_type`, `decision_outcome`, `intention_meta`](type.md) |
| How to find it | [`tags`](tags.md)<br>[`scope`](scope.md)<br>[`generality`](generality.md) |
| How long it holds | [`temporal`](temporal.md) |
| How far to trust it | [`evidence`](evidence.md)<br>[`provenance`](provenance.md)<br>[`disposition`, `review`](review-fields.md) |

The model proposes the content and plain code owns the rest. Code stamps identity, provenance,
corroboration and the `review` status, and it drops any memory whose quote it cannot find in the
cited turn. Each page says which side writes each field.

To try the `jq` filter on each page without an API key, use `examples/find-memories-by-tag/memories.jsonl`,
the recorded-session example's output. Each page shows what its filter prints on that file.

## One memory, end to end

This memory comes from the [recorded-session example](../../examples/extract-from-a-recorded-session/),
whose sessions are synthetic and whose model answers are hand-written in `responses.json` or follow
fixed rules in `replay.py`, so it shows the format, not what a model would find. It is a `decision`,
so every field a decision can carry is set. `intention_meta` is null because it only applies to type
`intention`.

```json
{
  "memory_id": "5f2d3aa8f126b49dd0bf6f96",
  "statement": "In shop, run pnpm install --frozen-lockfile in each worktree instead of symlinking node_modules from the main checkout, because Turbopack refuses a node_modules outside the project root.",
  "description": "The worktree setup script now removes a stale node_modules symlink and installs with pnpm. The same holds for any repo in the dev org that builds with Turbopack.",
  "type": "decision",
  "secondary_type": "convention",
  "scope": {
    "identified": {"level": "repo", "ref": "dev/shop"},
    "applicable": {"level": "org", "ref": "dev"}
  },
  "tags": {
    "domain": "developer-tooling",
    "activity": "coding",
    "component": ["worktree-setup"],
    "tech": ["pnpm", "turbopack"],
    "concept": ["dependency-resolution", "symlink"],
    "keyword": ["frozen-lockfile", "node_modules", "debugging"]
  },
  "generality": "instance-specific",
  "review": {"status": "active"},
  "intention_meta": null,
  "decision_outcome": "Adopted pnpm install per worktree; rejected symlinking node_modules.",
  "temporal": {
    "kind": "dated",
    "volatility": "med",
    "valid_as_of": "2026-09-03"
  },
  "disposition": {"level": "borderline", "note": "Decided in one session; no other worktree has confirmed the new setup script yet."},
  "provenance": {
    "confidence": "high",
    "corroboration": {
      "distinct_arms": 2,
      "distinct_conversations": 1,
      "source_arms": ["facts", "preferences"],
      "source_conversation_ids": ["cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11"]
    }
  },
  "evidence": [
    {
      "quote": "run pnpm install --frozen-lockfile in each worktree instead",
      "turn": 4,
      "speaker": "user",
      "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
      "arm": "facts",
      "authorship": "human_prose"
    },
    {
      "quote": "Turbopack says node_modules resolves outside the project root",
      "turn": 1,
      "speaker": "user",
      "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
      "arm": "facts",
      "authorship": "human_prose"
    },
    {
      "quote": "don't symlink node_modules, turbopack refuses it.",
      "turn": 4,
      "speaker": "user",
      "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
      "arm": "preferences",
      "authorship": "human_prose"
    }
  ]
}
```

## When a field is null

| Field | Null when | Explained in |
|---|---|---|
| `description` | The model gave no longer explanation | [statement.md](statement.md) |
| `secondary_type` | The memory is of one kind only | [type.md](type.md) |
| `scope.identified.ref`, `scope.applicable.ref` | The level is `personal` | [scope.md](scope.md) |
| `scope.applicable` | The memory applies where it was seen | [scope.md](scope.md) |
| `tags.domain`, `tags.activity` | The model named none | [tags.md](tags.md) |
| `intention_meta` | The type is not `intention` | [type.md](type.md) |
| `decision_outcome` | The type is not `decision` | [type.md](type.md) |
| `temporal.kind`, `temporal.volatility` | The model gave none, or gave an unknown value | [temporal.md](temporal.md) |
| `temporal.valid_as_of` | The conversation gives no date anchor | [temporal.md](temporal.md) |
| `evidence[].authorship` | Code could not attribute the quote's turn | [provenance.md](provenance.md) |

`disposition.note` is never null: it is an empty string on a `confident` memory, as
[review-fields.md](review-fields.md) says.

## Per-arm files

Each arm also writes its own `arms/<arm>/memories.jsonl` before the merge. Those lines carry the
same content flat, with no corroboration, and the schema does not describe them.

Their extra fields are `scope_level`, `scope_ref`, `temporal_kind`, `volatility`, `valid_as_of` and
`origin`.
