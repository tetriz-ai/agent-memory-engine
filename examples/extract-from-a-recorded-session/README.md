# Extract from a recorded session

This runs both extraction arms and the merge over two synthetic sessions. The extraction answers
are hand-written in `responses.json`, and `replay.py` answers the merge steps with fixed rules. The
output shows the pipeline and the memory format, not what a model would find.

The three node_modules memories, two from the facts arm and one from the preferences arm, merge into
one decision backed by both arms. It is the first memory, printed in full at the end: its
`source_arms` lists `facts` and `preferences`.

To run it, clone the repo and, from its root, run
`uv run python examples/extract-from-a-recorded-session/main.py`. uv installs the package first.

Install into the Python environment you run with `pip install .` from a clone. Then run:

```sh
python examples/extract-from-a-recorded-session/main.py
```

```text
2 sessions, 5 memories, ok=True

[decision] In shop, run pnpm install --frozen-lockfile in each worktree instead of symlinking node_modules from the main checkout, because Turbopack refuses a node_modules outside the project root.
[failed-attempt] Generating a fresh idempotency key for each billing retry attempt double-charged customers.
[convention] Always reuse the original idempotency key on billing retries.
[caution] The billing retry path built a new request without the original Idempotency-Key header, so the payment provider treated each retry as a new charge.
[preference] Phrase code review comments as questions, not directives.

The first memory in full:
{
  "memory_id": "5f2d3aa8f126b49dd0bf6f96",
  "statement": "In shop, run pnpm install --frozen-lockfile in each worktree instead of symlinking node_modules from the main checkout, because Turbopack refuses a node_modules outside the project root.",
  "description": "The worktree setup script now removes a stale node_modules symlink and installs with pnpm. The same holds for any repo in the dev org that builds with Turbopack.",
  "type": "decision",
  "secondary_type": "convention",
  "scope": {
    "identified": {
      "level": "repo",
      "ref": "dev/shop"
    },
    "applicable": {
      "level": "org",
      "ref": "dev"
    }
  },
  "tags": {
    "domain": "developer-tooling",
    "activity": "coding",
    "component": [
      "worktree-setup"
    ],
    "tech": [
      "pnpm",
      "turbopack"
    ],
    "concept": [
      "dependency-resolution",
      "symlink"
    ],
    "keyword": [
      "frozen-lockfile",
      "node_modules",
      "debugging"
    ]
  },
  "generality": "instance-specific",
  "review": {
    "status": "active"
  },
  "intention_meta": null,
  "decision_outcome": "Adopted pnpm install per worktree; rejected symlinking node_modules.",
  "temporal": {
    "kind": "dated",
    "volatility": "med",
    "valid_as_of": "2026-09-03"
  },
  "disposition": {
    "level": "borderline",
    "note": "Decided in one session; no other worktree has confirmed the new setup script yet."
  },
  "provenance": {
    "confidence": "high",
    "corroboration": {
      "distinct_arms": 2,
      "distinct_conversations": 1,
      "source_arms": [
        "facts",
        "preferences"
      ],
      "source_conversation_ids": [
        "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11"
      ]
    }
  },
  "evidence": [
    {
      "quote": "run pnpm install --frozen-lockfile in each worktree instead",
      "turn": 4,
      "speaker": "user",
      "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
      "arm": "facts",
      "authorship": "human_prose"
    },
    {
      "quote": "Turbopack says node_modules resolves outside the project root",
      "turn": 1,
      "speaker": "user",
      "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
      "arm": "facts",
      "authorship": "human_prose"
    },
    {
      "quote": "don't symlink node_modules, turbopack refuses it.",
      "turn": 4,
      "speaker": "user",
      "conversation_id": "cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11",
      "arm": "preferences",
      "authorship": "human_prose"
    }
  ]
}
```
