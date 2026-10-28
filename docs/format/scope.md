# scope

`scope` says where a memory holds. It is `{identified, applicable}`, and each side is `{level, ref}`.

| Side | Means |
|---|---|
| `identified` | Where the memory was observed. The model picks the narrowest correct level. |
| `applicable` | Where it applies, when that differs, such as a convention seen in one directory that holds repo-wide. Usually `null`. |

A non-null `applicable` always means the memory holds somewhere other than where it was seen.

| Level | `ref` holds |
|---|---|
| `personal` | Always `null` |
| `dir` | The directory path |
| `service` | The bare service name, lowercased, such as `billing` |
| `repo` | `<owner>/<repo>` |
| `team` | The team name |
| `org` | The organization or provider name, which need not be the user's own |

The level is written by the model and checked by code. A memory without a valid level is dropped.
On merge, `applicable` comes from the latest-dated member that names one.

A value looks like:

```json
{"scope": {"identified": {"level": "service", "ref": "billing"}, "applicable": null}}
```

To pick what holds for one service, for example with `jq`:

```sh
jq -r 'select(.scope.identified == {"level": "service", "ref": "billing"}) | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Always reuse the original idempotency key on billing retries.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
```
