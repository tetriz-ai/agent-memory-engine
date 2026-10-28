# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Plug in your own model backend: subclass ModelClient and pass it to run().

This backend is a few lines of rules instead of a model, so the example runs anywhere. A real one
would send `system_prompt` and `text` to its model and return the model's JSON answer. Whatever it
returns still goes through the evidence gate: a quote that is not in the cited turn is dropped.
"""

import json
import os
import re
import tempfile

from agent_memory_engine import ExtractResult, ModelClient, list_sessions, run

HERE = os.path.dirname(os.path.abspath(__file__))
TURN = re.compile(r"^\[(\d+)\] (user|assistant): (.*)$", re.M)


class RuleBackend(ModelClient):
    """Turns every user sentence that starts with "always" into a convention."""

    model_id = "rules-v1"
    cache_model_id = "rules-v1"

    def extract(self, system_prompt, text):
        if not text.startswith("--- SESSION METADATA ---") or "WHAT THIS ARM CAPTURES" in system_prompt:
            return self._json({"memories": []})  # only the preferences arm, and only extraction
        memories = []
        for turn, speaker, content in TURN.findall(text):
            for sentence in re.split(r"(?<=[.!?])\s+", content):
                if speaker == "user" and sentence.lower().startswith("always "):
                    memories.append(
                        {
                            "statement": sentence[0].upper() + sentence[1:],
                            "type": "convention",
                            "scope_level": "personal",
                            "tags": {"domain": "general", "keyword": ["always"]},
                            "evidence": [{"quote": sentence, "turn": int(turn), "speaker": "user"}],
                            "disposition": "confident",
                        }
                    )
        # One more, citing words that are not in the session: the gate will drop it.
        memories.append(
            {
                "statement": "Deploy on Fridays.",
                "type": "convention",
                "scope_level": "personal",
                "tags": {"domain": "general"},
                "disposition": "confident",
                "evidence": [{"quote": "we deploy on fridays", "turn": 1, "speaker": "user"}],
            }
        )
        return self._json({"memories": memories})

    def _json(self, obj):
        return ExtractResult(json.dumps(obj), 0.0, 0, self.model_id, ok=True)


sessions = list_sessions(os.path.join(HERE, "claude-home"))
result = run(
    sessions, client=RuleBackend(), data_dir=os.path.join(tempfile.mkdtemp(), "data"), log=lambda m: None
)
with open(os.path.join(result.run_dir, "arms", "preferences", "metrics.json"), encoding="utf-8") as f:
    arm = json.load(f)
print(f"kept {arm['memories_total']}, dropped by the evidence gate {arm['dropped_evidence']}")
with open(result.memories_path, encoding="utf-8") as f:
    for line in f:
        memory = json.loads(line)
        print(f"{memory['type']}: {memory['statement']}")
        print(f"  quote: {memory['evidence'][0]['quote']!r} (turn {memory['evidence'][0]['turn']})")
