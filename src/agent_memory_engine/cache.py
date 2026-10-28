# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Never pay for the same model call twice.

Every model result is saved under a fingerprint of its exact inputs: the call id, a prompt version
label, the model id, and the actual bytes sent (system prompt and rendered transcript). Hashing the
real prompt and transcript, not just a version label, means editing a prompt or changing how a
transcript is rendered forces a fresh call even if nobody bumped the label. A hit costs no model
call and replays the saved response, cost and duration. Writes are atomic (temp file + rename), so
a killed run never leaves a half-written entry behind.

The cache lives in the data dir: it holds raw model output over conversation content and must
never be committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile


def fingerprint(
    conversation_id: str, prompt_version: str, model_id: str, prompt_text: str = "", transcript_text: str = ""
) -> str:
    h = hashlib.sha256()
    for part in (conversation_id, prompt_version, model_id, prompt_text, transcript_text):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


class Cache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, fp: str) -> str:
        return os.path.join(self.root, f"{fp}.json")

    def get(self, fp: str):
        p = self._path(fp)
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None  # corrupt entry -> treat as miss, re-extract

    def put(self, fp: str, record: dict) -> None:
        p = self._path(fp)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
            os.replace(tmp, p)  # atomic
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
