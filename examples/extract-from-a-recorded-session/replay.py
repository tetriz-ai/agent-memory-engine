# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""A ModelClient that answers from recorded responses instead of calling a model.

Extraction answers come from responses.json, keyed by arm and the window's session date. They were
written by hand for the two sample sessions. Reconcile keeps every candidate, merge synthesis keeps
every member as it is, and the dedup pass groups the statements that mention a node_modules symlink.
"""

import json
import os
import re

from agent_memory_engine import ExtractResult, ModelClient

HERE = os.path.dirname(os.path.abspath(__file__))
RESPONSES = os.path.join(HERE, "responses.json")


class ReplayClient(ModelClient):
    model_id = "replay"
    cache_model_id = "replay"

    def __init__(self, path=RESPONSES):
        with open(path, encoding="utf-8") as f:
            self.responses = json.load(f)

    def _answer(self, obj):
        return ExtractResult(json.dumps(obj), 0.0, 1, self.model_id, ok=True)

    def extract(self, system_prompt, text):
        if text.startswith("--- SESSION METADATA ---"):
            arm = "facts" if "WHAT THIS ARM CAPTURES" in system_prompt else "preferences"
            date = json.loads(text.splitlines()[1])["session_date"][:10]
            return self._answer(self.responses.get(f"{arm}:{date}", {"memories": []}))
        if "CANDIDATE MEMORIES" in text:
            labels = re.findall(r"^\[(C\d+)\]", text, re.M)
            return self._answer({"memories": [{"members": [label]} for label in labels], "deleted": []})
        if text.startswith("Identify duplicate"):
            ids = re.findall(r"^\[(S\d+)\][^\n]*symlink", text, re.M | re.I)
            return self._answer({"duplicate_groups": [ids] if len(ids) >= 2 else []})
        return self._answer({"memories": []})
