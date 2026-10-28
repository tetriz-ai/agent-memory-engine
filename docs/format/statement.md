# statement, description, memory_id

| Field | Written by | Meaning |
|---|---|---|
| `statement` | Model. The merge may rephrase it. | One self-contained sentence that names the subject and the thing to remember. It reads on its own, with no "this repo" or "it". |
| `description` | Model | One to four sentences the statement cannot carry: the symptom, the consequence, the rejected alternative, an anchor. The reason is given only when the conversation gave one. `null` when absent. |
| `memory_id` | Code | A stable id: the first 24 hex characters of a SHA-256 over the normalized statement. Two memories with the same normalized statement share an id. |

Normalization lowercases, applies Unicode NFKC, turns curly quotes into straight ones, and folds
runs of whitespace into one space.

A value looks like:

```json
{"statement": "Phrase code review comments as questions, not directives.",
 "description": "Stated directly as a preference for how review feedback is written."}
```

To list every statement, for example with `jq`:

```sh
jq -r '.statement' examples/find-memories-by-tag/memories.jsonl
```

```text
In shop, run pnpm install --frozen-lockfile in each worktree instead of symlinking node_modules from the main checkout, because Turbopack refuses a node_modules outside the project root.
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Always reuse the original idempotency key on billing retries.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
Phrase code review comments as questions, not directives.
```
