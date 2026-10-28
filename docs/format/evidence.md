# evidence

`evidence` is a list of the quotes a memory rests on. It is never empty.

| Field | Written by | Means |
|---|---|---|
| `quote` | Model, checked by code | The stretch of the cited turn the quote matched, as the session holds it |
| `turn` | Model, checked by code | The 1-based turn index the quote came from |
| `speaker` | Model, checked by code | `user` or `assistant`, which must equal the role of that turn |
| `conversation_id` | Code | The session it came from, as `cc:<session-id>` |
| `arm` | Code | `facts` or `preferences` |
| `authorship` | Code | Who wrote the quoted bytes. See [provenance](provenance.md). |

The evidence gate runs on every candidate. The turn must exist, and its role must equal the
speaker. The quote must appear in the cited turn once both are folded the same way:

1. Each character with its combining marks goes through Unicode NFKC, so `x²` matches `x2`.
2. Letters are lowercased.
3. Curly quotes become straight ones.
4. Each run of whitespace becomes one space, and leading and trailing whitespace is dropped.

The quote may not include text the code inserts into a turn, such as `[...turn truncated...]`. The
memory then stores the matched stretch of the turn as the session holds it, not the model's retyping.

Tool output, shell output and harness text that came back through the user slot count as
`assistant`, so they cannot pass as the person's own words. A memory with any failing quote is
dropped, and each arm's `arms/<arm>/metrics.json` counts the drop in `dropped_evidence`.

A value looks like:

```json
{"evidence": [{"quote": "don't symlink node_modules, turbopack refuses it.", "turn": 4,
               "speaker": "user", "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
               "arm": "preferences", "authorship": "human_prose"}]}
```

To pick the memories backed by the person's own words, for example with `jq`:

```sh
jq -r 'select(any(.evidence[]; .authorship == "human_prose")) | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
In shop, run pnpm install --frozen-lockfile in each worktree instead of symlinking node_modules from the main checkout, because Turbopack refuses a node_modules outside the project root.
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Always reuse the original idempotency key on billing retries.
Phrase code review comments as questions, not directives.
```
