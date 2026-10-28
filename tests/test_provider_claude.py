# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""ClaudeCLIClient without running `claude`: subprocess.run is patched in every test."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from agent_memory_engine import extract
from agent_memory_engine.providers.claude_cli import ClaudeCLIClient
from agent_memory_engine.sources import list_sessions


def _proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


_OK = json.dumps(
    {
        "result": '{"memories": []}',
        "total_cost_usd": 0.12,
        "duration_ms": 900,
        "modelUsage": {"claude-opus-4-8": {}},
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "session_id": "s-1",
    }
)


class Command(unittest.TestCase):
    def test_the_call_is_isolated_and_not_saved_as_a_session(self):
        cmd = ClaudeCLIClient().build_command("/tmp/prompt.txt")
        self.assertEqual(cmd[:2], ["claude", "-p"])
        for flag in ("--no-session-persistence", "--exclude-dynamic-system-prompt-sections"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "claude-opus-4-8")
        self.assertEqual(cmd[cmd.index("--disallowedTools") + 1], "*")
        self.assertEqual(cmd[cmd.index("--system-prompt-file") + 1], "/tmp/prompt.txt")

    def test_the_prompt_goes_by_file_and_the_text_by_stdin(self):
        seen = {}

        def fake_run(cmd, **kw):
            with open(cmd[cmd.index("--system-prompt-file") + 1], encoding="utf-8") as f:
                seen["prompt"] = f.read()
            seen["input"] = kw["input"]
            seen["cwd"] = kw["cwd"]
            return _proc(stdout=_OK)

        with mock.patch("subprocess.run", fake_run):
            r = ClaudeCLIClient(neutral_cwd="/tmp").extract("SYSTEM → prompt", "the transcript")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(
            (seen["prompt"], seen["input"], seen["cwd"]), ("SYSTEM → prompt", "the transcript", "/tmp")
        )
        self.assertEqual((r.raw_text, r.cost_usd, r.duration_ms), ('{"memories": []}', 0.12, 900))
        self.assertEqual(r.meta["canonical_model"], "claude-opus-4-8")


class Failures(unittest.TestCase):
    def _run(self, *results):
        calls = iter(results)

        def fake_run(cmd, **kw):
            r = next(calls)
            if isinstance(r, BaseException):
                raise r
            return r

        with mock.patch("subprocess.run", fake_run), mock.patch("time.sleep"):
            return ClaudeCLIClient(max_attempts=3).extract("p", "t")

    def test_table(self):
        cases = [
            (
                "non-zero exit is retried then surfaces",
                [_proc(1, stderr="auth failed")] * 3,
                False,
                "exit 1: auth failed",
            ),
            ("a retry can succeed", [_proc(1, stderr="blip"), _proc(stdout=_OK)], True, None),
            ("timeout is terminal", [subprocess.TimeoutExpired("claude", 1)], False, "timeout"),
            ("missing binary", [FileNotFoundError("claude")] * 3, False, "spawn: claude"),
            ("unparseable CLI JSON", [_proc(stdout="not json")] * 3, False, "cli-json-parse"),
            (
                "CLI reports an error",
                [_proc(stdout=json.dumps({"is_error": True, "result": "limit"}))] * 3,
                False,
                "cli-is_error: limit",
            ),
        ]
        for name, results, ok, err in cases:
            with self.subTest(name):
                r = self._run(*results)
                self.assertEqual(r.ok, ok)
                if err:
                    self.assertIn(err, r.error)

    def test_spend_is_summed_across_attempts_and_kept_on_failure(self):
        failed = _proc(stdout=json.dumps({"is_error": True, "result": "limit", "total_cost_usd": 0.40}))
        r = self._run(failed, failed, failed)
        self.assertFalse(r.ok)
        self.assertAlmostEqual(r.cost_usd, 1.20)
        self.assertEqual(r.meta["http_retries"], 2)
        r = self._run(failed, _proc(stdout=_OK))
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.cost_usd, 0.40 + json.loads(_OK)["total_cost_usd"])
        self.assertEqual(r.meta["http_retries"], 1)

    def test_an_answer_with_no_cost_field_costs_nothing(self):
        r = self._run(_proc(stdout=json.dumps({"result": '{"memories": []}'})))
        self.assertTrue(r.ok)
        self.assertEqual(r.cost_usd, 0.0)

    def test_a_failed_call_is_accounted_by_the_arm(self):
        home = os.path.join(os.path.dirname(__file__), "fixtures", "claude-home")
        rows = [s.as_row() for s in list_sessions(home, projects=["/home/dev/src/shop"])][:1]
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out)
        failed = _proc(stdout=json.dumps({"is_error": True, "result": "limit", "total_cost_usd": 0.40}))
        with mock.patch("subprocess.run", lambda cmd, **kw: failed), mock.patch("time.sleep"):
            metrics, ok = extract.run(
                rows,
                out,
                ClaudeCLIClient(),
                arm="facts",
                cache_dir=os.path.join(out, "cache"),
                log=lambda m: None,
            )
        self.assertFalse(ok)
        self.assertAlmostEqual(metrics["cost_usd_total"], 1.20)
        self.assertEqual(metrics["http_retries"], 2)


class NoRealCall(unittest.TestCase):
    def test_a_call_in_the_suite_never_reaches_a_process(self):
        # Under the test guard the client's spawn fails, and the client reports it as a labelled
        # error instead of running the real CLI.
        r = ClaudeCLIClient(max_attempts=1).extract("p", "t")
        self.assertFalse(r.ok)
        self.assertTrue(r.error.startswith("spawn: a test tried to start a process"), r.error)


class DefaultsAndShapes(unittest.TestCase):
    def test_a_failing_call_is_tried_three_times_by_default(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _proc(1, stderr="busy")

        with mock.patch("subprocess.run", fake_run), mock.patch("time.sleep"):
            r = ClaudeCLIClient().extract("p", "t")
        self.assertFalse(r.ok)
        self.assertEqual(len(calls), 3)

    def test_json_that_is_not_an_object_is_a_labelled_error(self):
        with mock.patch("subprocess.run", lambda cmd, **kw: _proc(stdout="[]")), mock.patch("time.sleep"):
            r = ClaudeCLIClient().extract("p", "t")
        self.assertFalse(r.ok)
        self.assertTrue(r.error.startswith("cli-json-shape"), r.error)
