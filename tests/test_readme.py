# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The commands the README and docs/usage.md show print what they show, the README's offline commands
are the examples' commands, and the example copies of test data have not drifted from the fixtures."""

from __future__ import annotations

import contextlib
import datetime
import filecmp
import importlib.util
import io
import json
import os
import re
import unittest
from unittest import mock

from agent_memory_engine import cli, list_sessions, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The setup commands need a shell and network. The replay command is checked with fixed answers below.
QUICK_START = {
    "git clone https://github.com/tetriz-ai/agent-memory-engine": "downloads the public source repository",
    "cd agent-memory-engine": "changes the shell working directory",
    "uv sync --locked": "installs the locked development environment",
    "uv run python examples/extract-from-a-recorded-session/main.py | head -n 7": "checked below",
}
RECORDED = ["uv run python examples/extract-from-a-recorded-session/main.py | head -n 7"]
SAMPLE_HOME = os.path.join(ROOT, "examples", "list-sessions-in-a-date-range", "claude-home")


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _section(rel: str, heading: str) -> str:
    """The text under `heading` in file `rel`, up to the next heading of the same or a higher level."""
    text = _read(rel)
    start = text.index("\n" + heading + "\n") + 1
    level = len(heading) - len(heading.lstrip("#"))
    stop = re.compile(rf"^#{{1,{level}}} ", re.M).search(text, start + len(heading))
    return text[start : stop.start() if stop else len(text)]


def _blocks(section: str, lang: str) -> list[str]:
    """The bodies of the `lang` code blocks in a section, less the fence's indent under a list item."""
    blocks = []
    for indent, body in re.findall(rf"^( *)```{lang}\n(.*?)^\1```", section, re.S | re.M):
        lines = body.splitlines(keepends=True)
        assert all(ln.startswith(indent) or ln == "\n" for ln in lines), body
        blocks.append("".join(ln[len(indent) :] for ln in lines))
    return blocks


def _commands(section: str) -> list[str]:
    return [
        ln for body in _blocks(section, "sh") for ln in body.splitlines() if ln and not ln.startswith("#")
    ]


def _load_replay():
    example = os.path.join(ROOT, "examples", "extract-from-a-recorded-session")
    spec = importlib.util.spec_from_file_location("replay", os.path.join(example, "replay.py"))
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    return example, replay


def _format_memory():
    """The full memory shown under "One memory, end to end" in docs/format/README.md."""
    return json.loads(_blocks(_section("docs/format/README.md", "## One memory, end to end"), "json")[0])


class QuickStart(unittest.TestCase):
    def test_the_quick_start_is_the_listed_source_and_replay_steps(self):
        self.assertEqual(_commands(_section("README.md", "## Quick start")), list(QUICK_START))

    def test_the_real_run_commands_are_kept_separate_from_the_offline_sample(self):
        self.assertEqual(
            _commands(_section("README.md", "### Extract memories")),
            [
                'export GEMINI_API_KEY="your-key"',
                "uv run tetriz-agent-memory-engine run --since 2026-09-01 --limit 5",
            ],
        )

    def test_the_run_summary(self):
        # The summary the quick start shows is what the CLI prints for the recorded-session example.
        (want,) = _blocks(_section("README.md", "### Extract memories"), "text")
        run_id = re.search(r"runs/([^/]+)/memories.jsonl", want).group(1)
        example, replay = _load_replay()
        home = os.path.expanduser("~")
        result = run(
            list_sessions(os.path.join(example, "claude-home")),
            client=replay.ReplayClient(),
            data_dir=os.path.join(home, ".agent-memory-engine"),
            run_id=run_id,
            log=lambda m: None,
        )
        out = io.StringIO()
        cli._print_summary(result.metrics, result.run_dir, out)
        self.assertEqual(out.getvalue(), want)  # the CLI itself shortens the home directory to ~


class CommonTasks(unittest.TestCase):
    def test_the_preview_is_the_sessions_example(self):
        section = _section("README.md", "### See what a run would read")
        script = _read("examples/list-sessions-in-a-date-range/run.sh")
        self.assertEqual(
            _commands(section),
            [
                "uv run " + ln + " --limit 5"
                for ln in script.splitlines()
                if ln.startswith("tetriz-agent-memory-engine ")
            ],
        )
        example = _blocks(_read("examples/list-sessions-in-a-date-range/README.md"), "text")[-1]
        self.assertEqual(_blocks(section, "text"), [example])

    def test_the_recorded_memories_shown_are_the_example_output(self):
        # The block is the example's first 7 lines of stdout. The rest of that stdout is the memory
        # docs/format/README.md shows in full, and the README shows it folded under the block.
        section = _section("README.md", "### Try a recorded run, no API key needed")
        self.assertEqual(_commands(section), RECORDED)
        (shown,) = _blocks(section, "text")
        example = _blocks(_read("examples/extract-from-a-recorded-session/README.md"), "text")[-1]
        self.assertEqual(shown, "".join(example.splitlines(keepends=True)[:7]))
        _, sep, rest = example.partition("The first memory in full:\n")
        self.assertTrue(sep)
        self.assertEqual(_format_memory(), json.loads(rest))
        # The collapsed JSON under the block is the rest of that stdout, exactly as printed.
        (folded,) = _blocks(section, "json")
        self.assertEqual(folded, rest)

    def test_the_format_memory_is_the_first_line_of_the_sample_memories(self):
        with open(os.path.join(ROOT, "examples/find-memories-by-tag/memories.jsonl"), encoding="utf-8") as f:
            self.assertEqual(_format_memory(), json.loads(f.readline()))

    def test_the_format_memory_matches_the_script_output(self):
        # The docs/format memory equals the first memory of a fresh run of the recorded-session example.
        example, replay = _load_replay()
        result = run(
            list_sessions(os.path.join(example, "claude-home")),
            client=replay.ReplayClient(),
            data_dir=os.path.join(os.path.expanduser("~"), ".agent-memory-engine"),
            run_id="hero",
            log=lambda m: None,
        )
        with open(result.memories_path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        self.assertEqual(_format_memory(), first)


class UsageBlocks(unittest.TestCase):
    """The docs/usage.md commands print what it shows, on the sample session home."""

    def test_sessions(self):
        section = _section("docs/usage.md", "## Choose sessions")
        (code,) = _blocks(section, "sh")
        want = _blocks(section, "text")[0]
        argv = code.strip().split()[1:]
        out = io.StringIO()
        self.assertEqual(cli.main([*argv, "--claude-home", SAMPLE_HOME], out=out), 0)
        self.assertEqual(out.getvalue(), want)

    def test_the_python_api(self):
        section = _section("docs/usage.md", "## Use the Python API")
        code = _blocks(section, "python")[0]
        want = _blocks(section, "text")[0]
        out = io.StringIO()
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(code, "docs/usage.md", "exec"), {"datetime": datetime})
        finally:
            os.chdir(cwd)
        self.assertEqual(out.getvalue(), want)

    def test_the_python_api_snippets_run_as_one_program(self):
        # The two snippets, run together from the clone root: the first lists the sessions it prints,
        # and the second hands that same list to `run` (stubbed here, so no model is called).
        section = _section("docs/usage.md", "## Use the Python API")
        program = "\n".join(_blocks(section, "python"))
        want = _blocks(section, "text")[0]
        seen = {}

        def fake_run(sessions, **kw):
            seen["ids"] = [s.conversation_id for s in sessions]
            seen["kw"] = kw

            class _Result:
                ok, memories_path = True, "memories.jsonl"

            return _Result()

        import agent_memory_engine

        out = io.StringIO()
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            with (
                mock.patch.object(agent_memory_engine, "run", fake_run),
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}),
                contextlib.redirect_stdout(out),
            ):
                exec(compile(program, "docs/usage.md", "exec"), {})
        finally:
            os.chdir(cwd)
        self.assertEqual(out.getvalue(), want + "True memories.jsonl\n")
        self.assertEqual(seen["ids"], [ln.split()[-1] for ln in want.splitlines()])
        self.assertEqual(seen["kw"], {"api_key": "fake-key-for-test"})


class ExampleDataMatchesTheFixtures(unittest.TestCase):
    def test_copies(self):
        pairs = [
            ("examples/extract-from-a-recorded-session/responses.json", "tests/fixtures/responses.json"),
        ]
        homes = [
            "examples/list-sessions-in-a-date-range/claude-home",
            "examples/extract-from-a-recorded-session/claude-home",
            "examples/bring-your-own-provider/claude-home",
        ]
        for home in homes:
            for dirpath, _, files in os.walk(os.path.join(ROOT, home)):
                for name in files:
                    rel = os.path.relpath(os.path.join(dirpath, name), os.path.join(ROOT, home))
                    pairs.append((os.path.join(home, rel), os.path.join("tests/fixtures/claude-home", rel)))
        for copy, original in pairs:
            with self.subTest(copy=copy):
                self.assertTrue(
                    filecmp.cmp(os.path.join(ROOT, copy), os.path.join(ROOT, original), shallow=False)
                )
