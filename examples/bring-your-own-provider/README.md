# Bring your own provider

This plugs a backend of your own into the pipeline. Subclass `ModelClient` and implement
`extract(system_prompt, text)`, then pass an instance to `run(..., client=...)`.

The backend here is a few rules instead of a model, so it runs anywhere. It also proposes a memory
whose quote is not in the session, and the evidence gate drops it.

To run it, clone the repo and, from its root, run
`uv run python examples/bring-your-own-provider/main.py`. uv installs the package first.

Install into the Python environment you run with `pip install .` from a clone. Then run:

```sh
python examples/bring-your-own-provider/main.py
```

```text
kept 1, dropped by the evidence gate 1
convention: Always reuse the original idempotency key on billing retries.
  quote: 'always reuse the original idempotency key on billing retries.' (turn 3)
```
