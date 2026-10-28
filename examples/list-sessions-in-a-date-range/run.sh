#!/bin/sh
# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
# The README's "See what a run would read" task: the sessions on a sample Claude Code home, placed
# in a temporary HOME.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
HOME="$(mktemp -d)"
# The DATE column is the local calendar day; UTC keeps this output the same in every time zone.
TZ=UTC
export HOME TZ
trap 'rm -rf "$HOME"' EXIT
mkdir -p "$HOME/.claude"
cp -R "$here/claude-home/projects" "$HOME/.claude/"

tetriz-agent-memory-engine sessions --since 2026-09-01
