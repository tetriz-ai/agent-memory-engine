# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Run both arms and the merge over two sample sessions, with recorded model answers."""

import json
import os
import tempfile

from replay import HERE, ReplayClient

from agent_memory_engine import list_sessions, run

sessions = list_sessions(os.path.join(HERE, "claude-home"))
data_dir = os.path.join(tempfile.mkdtemp(), "data")
result = run(sessions, client=ReplayClient(), data_dir=data_dir, run_id="example", log=lambda msg: None)

with open(result.memories_path, encoding="utf-8") as f:
    memories = [json.loads(line) for line in f]

print(f"{len(sessions)} sessions, {len(memories)} memories, ok={result.ok}")
print()
for memory in memories:
    print(f"[{memory['type']}] {memory['statement']}")
print()
print("The first memory in full:")
print(json.dumps(memories[0], indent=2, ensure_ascii=False))
