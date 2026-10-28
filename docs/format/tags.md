# tags

`tags` is an object over six axes. Every axis is always present.

| Axis | Shape | Holds |
|---|---|---|
| `domain` | One string or `null` | The engineering work area the memory serves, such as `payments`, `frontend` or `developer-tooling`. The value is open: the model infers it. |
| `activity` | One string or `null` | The work mode in which a future agent should see this memory. |
| `component` | List of strings | Named services, modules or tools, lowercased, such as `billing-retry-job`. |
| `tech` | List of strings | Languages, frameworks and tool classes, such as `pnpm` or `go`. |
| `concept` | List of strings | Cross-cutting patterns, such as `idempotency` or `rate-limiting`. |
| `keyword` | List of strings | The catch-all for any other useful term. |

With Gemini, `activity` is one of these 14 values or `null`, which its response schema enforces.
With `claude`, the prompt asks for the same list, but nothing enforces it.

`coding`, `reviewing`, `debugging`, `deploying`, `planning`, `incident`, `onboarding`, `researching`,
`analyzing`, `experimenting`, `training`, `operating`, `writing`, `designing`.

The merge does not check the list again. A second work mode goes in `concept`.

The extraction model proposes each memory's tags. Reconcile and merge combine them in code, and
both union the open axes in first-seen order.

For `domain` and `activity`, one value wins:

| Step | Winning value |
|---|---|
| Reconcile, within one session | The first member's |
| Merge, across sessions | The one most members carry, with a tie going to the lexicographically smallest |

Under both rules, every other `domain` or `activity` value moves to `keyword`, so nothing is
dropped. A second value for one of them in a single answer moves there too.

A model that answers with a list of strings such as `tech:go` gets the same object. Code reads the
text before the first colon as an axis only when it names one of the six. Any other string, such as
`std::vector` or a bare word, is kept whole in `keyword`.

A value looks like:

```json
{
  "tags": {
    "domain": "payments",
    "activity": "incident",
    "component": ["billing-retry-job"],
    "tech": [],
    "concept": ["idempotency"],
    "keyword": ["double-charge"]
  }
}
```

Filtering on an axis is a plain JSON lookup, for example with `jq`:

```sh
jq -r 'select(.tags.domain == "payments") | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Always reuse the original idempotency key on billing retries.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
```
