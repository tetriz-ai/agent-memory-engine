# generality

`generality` is how broad the memory's subject is, written by the model.

| Value | Means |
|---|---|
| `instance-specific` | About one concrete subject: this repo's setup, this file, this service |
| `general-principle` | A portable rule that holds across subjects and projects |

The test: could you carry this memory to a different codebase unchanged? If not, it is
`instance-specific`. A rule stated once is not thereby instance-specific.

Generality differs from `temporal`, which is how long the memory stays true, and from
`scope.applicable`, which is where it applies.

The field is never null, and code fills it in two cases. A memory the merge model left out gets
`general-principle` if it is a `preference`, `convention` or `caution`, and `instance-specific`
otherwise. A value that is neither allowed value becomes `general-principle` if it contains
"general" or "principle", and `instance-specific` otherwise. So `instance-specific` also covers
memories the model did not label.

A value looks like:

```json
{"statement": "Phrase code review comments as questions, not directives.", "generality": "general-principle"}
```

To pick the rules that carry to another codebase, for example with `jq`:

```sh
jq -r 'select(.generality == "general-principle") | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
Always reuse the original idempotency key on billing retries.
The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
Phrase code review comments as questions, not directives.
```
