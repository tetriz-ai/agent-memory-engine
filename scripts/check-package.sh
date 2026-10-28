#!/bin/sh
# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
# Build the wheel, install it alone in a fresh virtualenv, and check what a user gets: the command
# runs, it reads a session home, the wheel has no runtime dependency, and every license is MIT.
set -eu
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
uv build --quiet --out-dir "$work/dist"
uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$work"/dist/*.whl
"$work/venv/bin/python" - <<'PYTHON'
from importlib.metadata import distribution

commands = {
    ep.name for ep in distribution("tetriz-agent-memory-engine").entry_points
    if ep.group == "console_scripts"
}
assert commands == {"tetriz-agent-memory-engine"}, commands
PYTHON
test ! -e "$work/venv/bin/agent-memory-engine"
"$work/venv/bin/tetriz-agent-memory-engine" --version
HOME="$work/home" "$work/venv/bin/tetriz-agent-memory-engine" sessions --claude-home tests/fixtures/claude-home > "$work/sessions.out"
tail -1 "$work/sessions.out" | grep -Fx "5 sessions, 7 windows."
if unzip -p "$work"/dist/*.whl '*/METADATA' | grep -q '^Requires-Dist:'; then
  echo "the wheel declares a runtime dependency" >&2
  exit 1
fi
uvx pip-licenses==5.5.5 --python "$work/venv/bin/python" --allow-only "MIT" --format plain
