# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""A scripted ModelClient for tests: fixed, hand-written answers for the synthetic session home.

Extraction answers come from fixtures/responses.json, keyed by arm and the window's session date.
Reconcile keeps every candidate; merge synthesis keeps every member as it is; the dedup pass
groups the statements that mention a node_modules symlink. No model is called.
"""

from __future__ import annotations

import json
import os
import re

from agent_memory_engine.providers import ExtractResult, ModelClient

RESPONSES = os.path.join(os.path.dirname(__file__), "fixtures", "responses.json")


class ScriptedClient(ModelClient):
    model_id = "scripted"
    cache_model_id = "scripted"

    def __init__(self, fail_on: str | None = None, responses_path: str = RESPONSES):
        with open(responses_path, encoding="utf-8") as f:
            self.responses = json.load(f)
        self.fail_on = fail_on
        self.calls = 0

    def _ok(self, obj) -> ExtractResult:
        return ExtractResult(json.dumps(obj), 0.0, 1, self.model_id, ok=True)

    def extract(self, system_prompt: str, text: str) -> ExtractResult:
        self.calls += 1
        if text.startswith("--- SESSION METADATA ---"):
            arm = "facts" if "WHAT THIS ARM CAPTURES" in system_prompt else "preferences"
            if arm == self.fail_on:
                return ExtractResult("", 0.0, 1, self.model_id, ok=False, error="scripted failure")
            date = json.loads(text.splitlines()[1])["session_date"][:10]
            return self._ok(self.responses.get(f"{arm}:{date}", {"memories": []}))
        if "CANDIDATE MEMORIES" in text:
            labels = re.findall(r"^\[(C\d+)\]", text, re.M)
            return self._ok({"memories": [{"members": [lab]} for lab in labels], "deleted": []})
        if text.startswith("Identify duplicate"):
            ids = re.findall(r"^\[(S\d+)\][^\n]*symlink", text, re.M | re.I)
            return self._ok({"duplicate_groups": [ids] if len(ids) >= 2 else []})
        return self._ok({"memories": []})  # synthesis: every member passes through as it is
