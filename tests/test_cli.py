# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The command line: selection flags, provider flags, exit codes, and what each command prints.
`run`, `extract` and `merge` use the scripted model in place of the provider's client."""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest import mock

from agent_memory_engine import __version__, cli, pipeline
from scripted import ScriptedClient

HOME = os.path.join(os.path.dirname(__file__), "fixtures", "claude-home")


def _main(*argv):
    out = io.StringIO()
    err = io.StringIO()
    with mock.patch("sys.stderr", err):
        try:
            code = cli.main(list(argv), out=out)
        except SystemExit as e:  # argparse
            code = e.code
    return code, out.getvalue(), err.getvalue()


def _scripted_clients(provider, **kw):
    c = ScriptedClient(fail_on=os.environ.get("SCRIPTED_FAIL_ON"))
    return pipeline.Clients(c, c, c)


class Sessions(unittest.TestCase):
    def test_table_and_totals(self):
        code, out, _ = _main("sessions", "--claude-home", HOME)
        self.assertEqual(code, 0)
        with open(os.path.join(os.path.dirname(__file__), "fixtures", "sessions-table.txt")) as f:
            self.assertEqual(out, f.read())

    def test_selection_flags(self):
        cases = [
            (["--since", "2026-09-10", "--until", "2026-10-01"], "2 sessions, 2 windows."),
            (["--project", "/home/dev/src/shop-admin"], "1 session, 1 window."),
            (["--limit", "1"], "1 session, 1 window."),
            (["--since", "2027-01-01"], "0 sessions, 0 windows."),
        ]
        for flags, last in cases:
            with self.subTest(flags=flags):
                code, out, _ = _main("sessions", "--claude-home", HOME, *flags)
                self.assertEqual(code, 0)
                self.assertEqual(out.splitlines()[-1], last)

    def test_bad_flags_are_usage_errors(self):
        cases = [
            (["sessions", "--since", "yesterday"], "not a date"),
            (["sessions", "--limit", "0"], "must be at least 1"),
            (["sessions", "--since", "2026-09-05", "--until", "2026-09-05"], "must be before"),
            (["run", "--provider", "claude", "--thinking-level", "high"], "Gemini setting"),
            (["run", "--provider", "claude", "--gemini-key-file", "k"], "only applies to --provider gemini"),
            (["run", "--provider", "openai"], "invalid choice"),
            (["merge"], "--run-id"),
        ]
        for argv, msg in cases:
            with self.subTest(argv=argv):
                code, _, err = _main(*argv, "--claude-home", HOME) if argv[0] != "merge" else _main(*argv)
                self.assertEqual(code, 2)
                self.assertIn(msg, err)

    def test_version(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), f"tetriz-agent-memory-engine {__version__}")

    def test_help_uses_the_package_command_name(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), self.assertRaises(SystemExit) as ctx:
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertTrue(out.getvalue().startswith("usage: tetriz-agent-memory-engine "))


@mock.patch.object(pipeline, "make_clients", _scripted_clients)
class Run(unittest.TestCase):
    def setUp(self):
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        self.data = os.path.join(os.environ["HOME"], "out")

    def test_run_prints_one_line_per_stage(self):
        code, out, _ = _main(
            "run",
            "--claude-home",
            HOME,
            "--project",
            "/home/dev/src/shop",
            "--data-dir",
            self.data,
            "--run-id",
            "r1",
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            out.splitlines(),
            [
                "facts        2 sessions, 0 skipped, 0 unparseable windows, 4 memories, $0.00",
                "preferences  2 sessions, 0 skipped, 0 unparseable windows, 3 memories, $0.00",
                "merge        7 in, 5 merged, $0.00",
                "Wrote ~/out/runs/r1/memories.jsonl (5 memories), total $0.00",
            ],
        )

    def test_no_matching_session_is_a_usage_error(self):
        code, _, err = _main("run", "--claude-home", HOME, "--since", "2027-01-01", "--data-dir", self.data)
        self.assertEqual(code, 2)
        self.assertIn("no sessions match", err)

    def test_missing_key_is_a_usage_error_before_any_work(self):
        del os.environ["GEMINI_API_KEY"]
        code, _, err = _main("run", "--claude-home", HOME, "--data-dir", self.data)
        self.assertEqual(code, 2)
        self.assertIn("GEMINI_API_KEY", err)
        self.assertFalse(os.path.exists(self.data))

    def test_data_dir_in_a_git_repo_is_a_usage_error(self):
        os.makedirs(os.path.join(self.data, ".git"))
        code, _, err = _main("run", "--claude-home", HOME, "--data-dir", os.path.join(self.data, "x"))
        self.assertEqual(code, 2)
        self.assertIn("inside a git repo", err)

    def test_a_stopped_run_exits_1_and_says_where(self):
        with mock.patch.dict(os.environ, {"SCRIPTED_FAIL_ON": "preferences"}):
            code, out, _ = _main("run", "--claude-home", HOME, "--data-dir", self.data, "--run-id", "s")
        self.assertEqual(code, 1)
        self.assertIn("STOPPED: scripted failure", out)
        self.assertTrue(out.splitlines()[-1].startswith("Stopped. Partial outputs and metrics.json are in"))

    def test_extract_then_merge_equals_run(self):
        args = ["--claude-home", HOME, "--project", "/home/dev/src/shop", "--data-dir", self.data]
        self.assertEqual(_main("extract", "--arm", "facts", *args, "--run-id", "x")[0], 0)
        self.assertEqual(_main("extract", "--arm", "preferences", *args, "--run-id", "x")[0], 0)
        code, out, _ = _main("merge", "--data-dir", self.data, "--run-id", "x")
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[0], "merge        7 in, 5 merged, $0.00")
        _main("run", *args, "--run-id", "y")

        def read(rid):
            with open(os.path.join(self.data, "runs", rid, "memories.jsonl"), encoding="utf-8") as f:
                return f.read()

        self.assertEqual(read("x"), read("y"))

    def test_merge_without_arm_output_is_a_usage_error(self):
        code, _, err = _main("merge", "--data-dir", self.data, "--run-id", "nothing")
        self.assertEqual(code, 2)
        self.assertIn("has no arm output", err)

    def test_unparseable_windows_are_printed(self):
        class _Bad(ScriptedClient):
            def extract(self, system_prompt, text):
                if text.startswith("--- SESSION METADATA ---") and "WHAT THIS ARM CAPTURES" in system_prompt:
                    self.calls += 1
                    from agent_memory_engine.providers import ExtractResult

                    return ExtractResult('{"memories": [', 0.0, 1, self.model_id, ok=True)
                return super().extract(system_prompt, text)

        def clients(provider, **kw):
            c = _Bad()
            return pipeline.Clients(c, c, c)

        with mock.patch.object(pipeline, "make_clients", clients):
            code, out, _ = _main(
                "run",
                "--claude-home",
                HOME,
                "--project",
                "/home/dev/src/shop",
                "--data-dir",
                self.data,
                "--run-id",
                "u",
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            out.splitlines()[0],
            "facts        2 sessions, 0 skipped, 2 unparseable windows, 0 memories, $0.00",
        )
        with open(os.path.join(self.data, "runs", "u", "metrics.json")) as f:
            m = json.load(f)
        self.assertEqual(m["stages"]["facts"]["windows_unparseable"], 2)


@mock.patch.object(pipeline, "make_clients", _scripted_clients)
class MoreRunBehaviour(unittest.TestCase):
    def setUp(self):
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        self.data = os.path.join(os.environ["HOME"], "out")

    def test_a_failed_dedup_is_reported(self):
        class _NoDedup(ScriptedClient):
            def extract(self, system_prompt, text):
                if text.startswith("Identify duplicate"):
                    from agent_memory_engine.providers import ExtractResult

                    return ExtractResult("", 0.0, 1, self.model_id, ok=False, error="http 500: busy")
                return super().extract(system_prompt, text)

        def clients(provider, **kw):
            c = _NoDedup()
            return pipeline.Clients(c, c, c)

        with mock.patch.object(pipeline, "make_clients", clients):
            code, out, _ = _main(
                "run",
                "--claude-home",
                HOME,
                "--project",
                "/home/dev/src/shop",
                "--data-dir",
                self.data,
                "--run-id",
                "d",
            )
        self.assertEqual(code, 0)
        self.assertIn("dedup failed: http 500: busy", out)
        with open(os.path.join(self.data, "runs", "d", "metrics.json")) as f:
            merge_stage = json.load(f)["stages"]["merge"]
        self.assertEqual(merge_stage["dedup_error"], "http 500: busy")
        self.assertIn("members_uncovered", merge_stage)

    def test_a_held_run_id_is_a_usage_error(self):
        from agent_memory_engine.paths import run_lock

        with run_lock(os.path.join(self.data, "runs", "held"), ".run-lock"):
            for argv in (["run"], ["extract", "--arm", "facts"]):
                with self.subTest(argv=argv):
                    code, _, err = _main(
                        *argv, "--claude-home", HOME, "--data-dir", self.data, "--run-id", "held"
                    )
                    self.assertEqual(code, 2)
                    self.assertIn("another live run", err)

    def test_a_run_id_with_other_output_is_a_usage_error(self):
        base = ("--claude-home", HOME, "--data-dir", self.data, "--run-id", "mixed")
        code, _, _ = _main("extract", "--arm", "facts", "--project", "/home/dev/src/shop", *base)
        self.assertEqual(code, 0)
        for argv in (["extract", "--arm", "preferences", "--project", "/home/dev/src/shop-admin"], ["run"]):
            with self.subTest(argv=argv):
                code, _, err = _main(*argv, *base)
                self.assertEqual(code, 2)
                self.assertIn("--run-id", err)

    def test_a_corrupt_sessions_file_is_a_usage_error(self):
        run_dir = os.path.join(self.data, "runs", "bad")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "sessions.jsonl"), "w") as f:
            f.write("not json\n")
        for argv in (["extract", "--arm", "facts"], ["run"]):
            with self.subTest(argv=argv):
                code, _, err = _main(*argv, "--claude-home", HOME, "--data-dir", self.data, "--run-id", "bad")
                self.assertEqual(code, 2)
                self.assertIn("unreadable", err)

    def test_extract_prints_the_stop_reason(self):
        with mock.patch.dict(os.environ, {"SCRIPTED_FAIL_ON": "facts"}):
            code, out, _ = _main(
                "extract", "--arm", "facts", "--claude-home", HOME, "--data-dir", self.data, "--run-id", "x"
            )
        self.assertEqual(code, 1)
        self.assertIn("STOPPED: scripted failure", out)
        self.assertNotIn("Wrote", out)


class MissingClaudeCLI(unittest.TestCase):
    def test_no_claude_on_path_is_a_usage_error_before_any_run_dir(self):
        data = os.path.join(os.environ["HOME"], "out")
        for argv in (["run"], ["extract", "--arm", "facts"]):
            with self.subTest(argv=argv), mock.patch("shutil.which", return_value=None):
                code, _, err = _main(
                    *argv, "--provider", "claude", "--claude-home", HOME, "--data-dir", data, "--run-id", "c"
                )
                self.assertEqual(code, 2)
                self.assertIn("no claude CLI on PATH", err)
                self.assertFalse(os.path.exists(os.path.join(data, "runs", "c")))


class RunFlagsReachThePipeline(unittest.TestCase):
    def test_workers_and_model(self):
        seen = {}

        class _Stop(Exception):
            pass

        def fake_run(refs, **kw):
            seen.update(kw)
            raise _Stop

        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}),
            mock.patch.object(pipeline, "run", fake_run),
            self.assertRaises(_Stop),
        ):
            cli.main(
                ["run", "--claude-home", HOME, "--workers", "3", "--model", "example-model"],
                out=io.StringIO(),
            )
        self.assertEqual((seen["workers"], seen["model"]), (3, "example-model"))


class NonExecutableClaude(unittest.TestCase):
    def test_a_claude_file_that_cannot_run_is_not_a_claude_cli(self):
        bindir = os.path.join(os.environ["HOME"], "bin")
        os.makedirs(bindir)
        open(os.path.join(bindir, "claude"), "w").close()  # present, not executable
        data = os.path.join(os.environ["HOME"], "out")
        with mock.patch.dict(os.environ, {"PATH": bindir}):
            code, _, err = _main("run", "--provider", "claude", "--claude-home", HOME, "--data-dir", data)
        self.assertEqual(code, 2)
        self.assertIn("no claude CLI on PATH", err)


class PrintedPaths(unittest.TestCase):
    def test_the_home_directory_shows_as_a_tilde_only_at_a_path_boundary(self):
        base = os.path.realpath(os.environ["HOME"])
        home = os.path.join(base, "dev")
        os.makedirs(home)
        link = os.path.join(base, "link-to-dev")
        os.symlink(home, link)
        cases = [
            (home, os.path.join(home, "runs", "r"), "~/runs/r"),
            (home + "/", os.path.join(home, "runs", "r"), "~/runs/r"),  # a trailing slash on HOME
            (base + "//dev", os.path.join(home, "runs", "r"), "~/runs/r"),  # a doubled separator in HOME
            (home, home, "~"),
            (home, home + "2/runs/r", home + "2/runs/r"),  # a sibling that shares the prefix
            (link, os.path.join(home, "runs", "r"), "~/runs/r"),  # HOME is a symlink to the real dir
        ]
        for env_home, path, shown in cases:
            with self.subTest(home=env_home, path=path), mock.patch.dict(os.environ, {"HOME": env_home}):
                self.assertEqual(cli._tilde(path), shown)


class WorkersAreClamped(unittest.TestCase):
    def test_more_workers_than_the_maximum_run_the_maximum(self):
        from agent_memory_engine import extract
        from agent_memory_engine.sources import list_sessions

        seen = []
        real = extract.ThreadPoolExecutor

        def pool(max_workers):
            seen.append(max_workers)
            return real(max_workers=max_workers)

        rows = [r.as_row() for r in list_sessions(HOME)]
        out = os.path.join(os.environ["HOME"], "w")
        with mock.patch.object(extract, "ThreadPoolExecutor", pool):
            extract.run(
                rows,
                out,
                ScriptedClient(),
                arm="facts",
                cache_dir=os.path.join(out, "c"),
                workers=50,
                log=lambda m: None,
            )
        self.assertEqual(seen, [extract.CONV_MAX_WORKERS])


class GeminiNeedsNoClaude(unittest.TestCase):
    def test_choosing_gemini_never_looks_up_claude(self):
        with mock.patch("shutil.which", side_effect=AssertionError("looked up claude")):
            clients = pipeline.make_clients("gemini", api_key="fake-key-for-test")
        clients.close()


class SelectionOutput(unittest.TestCase):
    def test_limit_keeps_the_most_recent(self):
        code, out, _ = _main("sessions", "--claude-home", HOME, "--limit", "1")
        self.assertEqual(code, 0)
        self.assertIn("5b9e0c47-a1d2-4f83-b6c5-d4e3f2a1b0c9", out)
        self.assertNotIn("7d3c9a10", out)

    def test_the_date_column_is_the_local_day(self):
        import time

        os.environ["TZ"] = "Asia/Tokyo"
        time.tzset()
        try:
            code, out, _ = _main("sessions", "--claude-home", HOME, "--since", "2026-10-02")
        finally:
            os.environ["TZ"] = "UTC"
            time.tzset()
        # 2026-10-01T19:00Z is 2026-10-02 04:00 in Tokyo: kept by --since 2026-10-02, shown as that day.
        self.assertEqual(code, 0)
        self.assertTrue(out.splitlines()[1].startswith("2026-10-02"), out)

    def test_a_missing_projects_dir_is_a_usage_error(self):
        missing = os.path.join(os.environ["HOME"], "nowhere")
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        for argv in (["sessions"], ["run"], ["extract", "--arm", "facts"]):
            with self.subTest(argv=argv):
                code, _, err = _main(*argv, "--claude-home", missing)
                self.assertEqual(code, 2)
                self.assertIn(os.path.join(missing, "projects"), err)
