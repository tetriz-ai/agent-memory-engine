# type, secondary_type, decision_outcome, intention_meta

`type` is the kind of memory, written by the model and checked by code. A memory with any other
value is dropped.

| Type | Arm | Means |
|---|---|---|
| `fact` | facts | A specific, non-obvious fact about this setup: a flag, a path, a version pin, a quirk |
| `decision` | facts | A settled choice: what was chosen and what was rejected |
| `caution` | facts | A sharp edge, a silent failure, a failure and its fix |
| `failed-attempt` | facts | An approach that was tried and did not work |
| `intention` | facts | A stated commitment to do something later, with a way to tell it is done |
| `preference` | preferences | How this person likes work done |
| `convention` | preferences | A repeated "we always" or "we never" practice |

| Field | Holds | On merge |
|---|---|---|
| `secondary_type` | One more type when a memory is two kinds at once. Never equals `type`, else `null`. | The latest-dated member's secondary type that differs from the merged `type`, or `null` if no member names one |
| `decision_outcome` | For a `decision`, what was adopted and what was rejected. `null` for other types. | From the latest-dated decision member. If members recorded different outcomes, the text says so. |
| `intention_meta` | For an `intention`, `{what, done_check}`. `null` for other types. | From the latest-dated intention member that has one. A member without one never erases it. |

An intention folded together with other memories stays an intention with its `intention_meta`,
whatever type the model names. That holds within one session, in reconcile and in merge.

A value looks like:

```json
{"type": "decision", "secondary_type": null,
 "decision_outcome": "Adopted pnpm install per worktree; rejected symlinking node_modules."}
```

To pick the memories to see before risky work, for example with `jq`:

```sh
jq -r 'select(.type == "caution" or .type == "failed-attempt") | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
```
