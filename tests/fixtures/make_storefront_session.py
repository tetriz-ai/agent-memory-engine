# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Write a synthetic Claude Code session long enough to make three windows.

Deterministic. Its output is committed, byte-identical, as
projects/-home-dev-src-storefront/<SID>.jsonl under both tests/fixtures/claude-home/ and
examples/list-sessions-in-a-date-range/claude-home/. Regenerate each with
  python3 tests/fixtures/make_storefront_session.py > <that path>
and check both committed copies against a fresh generation with
  python3 tests/fixtures/make_storefront_session.py --check
which exits 1 and names each copy that differs or is missing.
"""

import argparse
import json
import os
import sys
import tempfile

SID = "3f6a2c81-9d47-4b0e-a5c3-1e8b7d2f4a60"
CWD = "/home/dev/src/storefront"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COPIES = [
    os.path.join(home, "projects", "-home-dev-src-storefront", f"{SID}.jsonl")
    for home in ("tests/fixtures/claude-home", "examples/list-sessions-in-a-date-range/claude-home")
]


def log_lines() -> str:
    specs = [
        "cart totals",
        "apply coupon",
        "guest checkout",
        "saved card",
        "3-D Secure challenge",
        "address validation",
        "shipping quote",
        "order confirmation email",
    ]
    out = []
    for run in range(1, 9):
        for n, spec in enumerate(specs, 1):
            ms = 400 + (run * 37 + n * 91) % 900
            out.append(
                f"[shard {run}/8] checkout > {spec} > completes within the step timeout ... ok ({ms} ms)"
            )
        if run == 6:
            out.append("[shard 6/8] checkout > saved card > charges the stored card once ... FAILED")
            out.append(
                "  Error: expected status 'paid', received 'rate_limited' from POST /sandbox/v1/charges"
            )
            out.append("  at waitForCharge (tests/e2e/checkout/helpers.ts:88:11)")
    return "\n".join(out)


def para(topic: str, n: int) -> str:
    s = (
        f"On {topic}: the failing step is always the sandbox charge call, never the browser step before it. "
        "The CI runner starts eight shards at once, and every shard creates its own test customer "
        "and charges it. "
        "Locally the suite runs with two workers, so the sandbox never sees more than a couple of "
        "charges a second. "
    )
    return (s * n).strip()


TURNS = [
    (
        "user",
        "The checkout e2e suite fails about one run in five on CI but never locally. "
        "Here is the last failing run:\n\n" + log_lines() * 3,
    ),
    ("assistant", para("the failing run", 10)),
    ("user", "That matches what I see. What would you change?"),
    (
        "assistant",
        para("the fix", 14)
        + "\n\nProposed change: cap the checkout project at two workers in playwright.config.ts and keep the "
        "other projects at eight.",
    ),
    (
        "user",
        "That fixed the flake on 20 CI runs. Note for later: the payment sandbox rate-limits at "
        "10 requests per second per key, so run checkout specs with --workers 2 at most.",
    ),
    ("assistant", para("the follow-up", 7)),
    ("user", "Thanks, merge it."),
    ("assistant", "Merged. The checkout project now runs with two workers on CI."),
]


def render() -> str:
    lines = []
    prev = None
    for i, (role, text) in enumerate(TURNS, 1):
        uid = f"s{i}"
        ev = {
            "parentUuid": prev,
            "isSidechain": False,
            "userType": "external",
            "cwd": CWD,
            "sessionId": SID,
            "version": "2.1.4",
            "gitBranch": "main",
            "type": role,
        }
        if role == "user":
            ev["promptSource"] = "typed"
            ev["message"] = {"role": "user", "content": text}
        else:
            ev["message"] = {"role": "assistant", "content": [{"type": "text", "text": text}]}
        ev["uuid"] = uid
        ev["timestamp"] = f"2026-09-08T09:{i * 3:02d}:00.000Z"
        lines.append(json.dumps(ev) + "\n")
        prev = uid
    return "".join(lines)


def check(root: str) -> int:
    """Regenerate into a temp file and compare it byte for byte with each committed copy."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = os.path.join(tmp, f"{SID}.jsonl")
        with open(fresh, "w", encoding="utf-8", newline="") as f:
            f.write(render())
        with open(fresh, "rb") as f:
            want = f.read()
    bad = 0
    for rel in COPIES:
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as f:
                same = f.read() == want
        except FileNotFoundError:
            print(f"missing: {rel}", file=sys.stderr)
            bad += 1
            continue
        if not same:
            print(f"differs from a fresh generation: {rel}", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the storefront session, or check its committed copies.")
    ap.add_argument(
        "--check", action="store_true", help="compare the committed copies with a fresh generation"
    )
    ap.add_argument("--root", default=ROOT, help="the repository root to check (default: this checkout)")
    args = ap.parse_args(argv)
    if args.check:
        return check(args.root)
    sys.stdout.write(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
