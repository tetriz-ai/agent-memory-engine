#!/bin/sh
# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
#
# Runs every example the way a user would: build the wheel, install it alone in a fresh virtualenv,
# then for each examples/<name>/README.md run its `sh` block from the repo root, with an empty
# temporary HOME and the virtualenv first on PATH, and compare standard output with the README's
# last block. Fails when the index and the folders disagree. Shell examples are shellchecked.
set -eu
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
uv build --quiet --out-dir "$work/dist"
uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$work"/dist/*.whl
if command -v shellcheck >/dev/null 2>&1; then
  find examples -name '*.sh' -exec shellcheck {} +
else
  echo "shellcheck not found; skipping the shell lint" >&2
fi
PATH="$work/venv/bin:$PATH" WORK="$work" "$work/venv/bin/python" - <<'PY'
import os, re, subprocess, sys, tempfile

fail = False
folders = sorted(d for d in os.listdir("examples") if os.path.isdir(os.path.join("examples", d)))
index = open("examples/README.md", encoding="utf-8").read()
listed = sorted(set(re.findall(r"\]\(([a-z0-9-]+)/\)", index)))
if listed != folders:
    print(f"FAIL index: listed {listed}, folders {folders}")
    fail = True
for name in folders:
    text = open(f"examples/{name}/README.md", encoding="utf-8").read()
    blocks = re.findall(r"```(\w+)\n(.*?)```", text, re.S)
    commands = [body for lang, body in blocks if lang == "sh"]
    if not blocks or blocks[-1][0] != "text" or len(commands) != 1:
        print(f"FAIL {name}: needs one sh block and a final text block")
        fail = True
        continue
    home = tempfile.mkdtemp()
    env = dict(os.environ, HOME=home)
    env.pop("GEMINI_API_KEY", None)
    r = subprocess.run(["sh", "-c", commands[0]], env=env, capture_output=True, text=True)
    if r.returncode != 0 or r.stdout != blocks[-1][1]:
        print(f"FAIL {name} (exit {r.returncode})")
        sys.stdout.writelines(f"  {line}\n" for line in (r.stdout + r.stderr).splitlines())
        fail = True
    else:
        print(f"ok   {name}")
sys.exit(1 if fail else 0)
PY
