# temporal

`temporal` is `{kind, volatility, valid_as_of}`. All three are written by the model and may be
`null`, and an unknown value becomes `null` rather than dropping the memory.

On merge, all three come together from the member with the most recent session date.

| Field | Values | Means |
|---|---|---|
| `kind` | `timeless`, `dated`, `episodic`, `until-event` | Whether it is a standing invariant, a state that holds now but will change, a one-off past event, or true until something happens |
| `volatility` | `low`, `med`, `high` | A rough guess at how fast it goes stale |
| `valid_as_of` | A free-form date anchor, such as `2026-09-03` or `2026-05` | When the conversation anchored it, resolved against the session date. `null` when the conversation gives no anchor. |

A value looks like:

```json
{"temporal": {"kind": "dated", "volatility": "med", "valid_as_of": "2026-09-03"}}
```

To pick memories the model labelled `timeless`, for example with `jq`:

```sh
jq -r 'select(.temporal.kind == "timeless") | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Always reuse the original idempotency key on billing retries.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
Phrase code review comments as questions, not directives.
```
