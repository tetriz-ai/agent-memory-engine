# disposition, review

Both fields help a person or a downstream tool decide which memories to trust.

`disposition` is `{level, note}`, and the level is `confident` or `borderline`. The model marks a
memory borderline when its statement claims more than its evidence supports, and the note says why.
On a `confident` memory the note is an empty string.

On merge, a memory stays borderline if any member was, unless two or more distinct sessions back
it. Two arms reading one session do not count as two.

`review` is `{status}`, written by code from the type alone. It marks intentions for human review. An active
status does not establish that a memory is correct, current or safe to load. Every `intention` is `hold-for-review`, so a person checks it first,
and every other type is `active`, whatever the model says. An intention is a "we should do this"
note, and a group of memories that holds one stays an intention through reconcile and merge.

A value looks like:

```json
{"disposition": {"level": "borderline", "note": "seen in one session only"},
 "review": {"status": "hold-for-review"}}
```

To select active memories that the model marked confident for your review, for example with `jq`:

```sh
jq -r 'select(.review.status == "active" and .disposition.level == "confident") | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Always reuse the original idempotency key on billing retries.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
Phrase code review comments as questions, not directives.
```
