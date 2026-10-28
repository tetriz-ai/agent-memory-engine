# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Truncation limits of a turn and of a whole session, and which events of a session file count."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent_memory_engine import transcript


class TurnTruncation(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(transcript._truncate_turn("x" * 8000), "x" * 8000)
        text = "h" * 6000 + "m" * 2001 + "t" * 1500  # 9501 chars
        out = transcript._truncate_turn("a" + "x" * 8000)
        self.assertNotEqual(out, "a" + "x" * 8000)
        out = transcript._truncate_turn(text)
        head, tail = out.split("\n[...turn truncated...]\n")
        self.assertEqual(head, "h" * 6000)
        self.assertEqual(tail, "t" * 1500)


class TranscriptBudget(unittest.TestCase):
    def _conv(self, sizes):
        turns = [transcript.Turn(i=k + 1, role="user", content="x" * n) for k, n in enumerate(sizes)]
        return transcript.Conversation("c", "claude-code", "local", None, turns, False)

    def test_each_turn_costs_its_length_plus_32(self):
        # 30 turns of 9,968 chars cost exactly 300,000: all kept. One more char cuts the last turn.
        full = transcript.render_transcript(self._conv([9968] * 30))
        self.assertEqual((len(full.turns), full.truncated), (30, False))
        over = transcript.render_transcript(self._conv([9968] * 29 + [9969]))
        self.assertEqual((len(over.turns), over.truncated), (29, True))
        self.assertEqual(transcript.DEFAULT_MAX_CHARS, 300_000)


class SessionEvents(unittest.TestCase):
    def test_a_line_of_another_session_is_not_read(self):
        sid = "11111111-2222-3333-4444-555555555555"
        other = "99999999-2222-3333-4444-555555555555"
        lines = [
            {
                "type": "user",
                "sessionId": sid,
                "promptSource": "typed",
                "message": {"role": "user", "content": "mine"},
            },
            {
                "type": "user",
                "sessionId": other,
                "promptSource": "typed",
                "message": {"role": "user", "content": "theirs"},
            },
        ]
        path = os.path.join(tempfile.mkdtemp(), f"{sid}.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in lines)
        turns, _ = transcript.parse_claude_code_session(path, sid)
        self.assertEqual([t.content for t in turns], ["mine"])
