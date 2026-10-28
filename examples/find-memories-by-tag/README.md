# Find memories by tag

This picks memories out of a run's `memories.jsonl` by their typed tags and type, keeping only those
whose `review.status` is `active` and whose disposition is `confident`. Each lookup is a plain read of
`tags`.

It finds the payments memories about idempotency, the cautions and failed attempts to see while
debugging, and what to keep in mind before a review. The file is the recorded-session example's
output, copied as is.

To run it, clone the repo and, from its root, run
`uv run python examples/find-memories-by-tag/main.py`. uv installs the package first.

Install into the Python environment you run with `pip install .` from a clone. Then run:

```sh
python examples/find-memories-by-tag/main.py
```

```text
Before touching billing retries (domain=payments, concept=idempotency):
  [failed-attempt] Generating a fresh idempotency key for each billing retry attempt double-charged customers.
  [convention] Always reuse the original idempotency key on billing retries.
  [caution] The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
While debugging or in an incident (cautions and failed attempts only):
  [debugging] The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
  [incident] Generating a fresh idempotency key for each billing retry attempt double-charged customers.
Before a review:
  Phrase code review comments as questions, not directives.
```
