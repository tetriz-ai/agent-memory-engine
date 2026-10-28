# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Session discovery and selection over the synthetic Claude Code home in tests/fixtures."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from agent_memory_engine import cli, extract, transcript
from agent_memory_engine.sources import claude_code as cc
from scripted import ScriptedClient

FIXTURE_HOME = os.path.join(os.path.dirname(__file__), "fixtures", "claude-home")
SHOP = "7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11"
BILLING = "c41f8e22-6b9d-4f3a-8d71-90e5a2b7c013"
ADMIN = "e2a7d5b0-3c18-4e6f-9b42-7f0c1d8a6e35"
NOTES = "5b9e0c47-a1d2-4f83-b6c5-d4e3f2a1b0c9"
STOREFRONT = "3f6a2c81-9d47-4b0e-a5c3-1e8b7d2f4a60"  # written by fixtures/make_storefront_session.py


def _ids(refs):
    return [r.native_id for r in refs]


class ListSessions(unittest.TestCase):
    def test_every_top_level_session_oldest_first(self):
        refs = cc.list_sessions(FIXTURE_HOME)
        # the sub-agent transcript, the non-uuid file and the README are not sessions
        self.assertEqual(_ids(refs), [SHOP, STOREFRONT, BILLING, ADMIN, NOTES])
        self.assertTrue(all(r.conversation_id == f"cc:{r.native_id}" for r in refs))

    def test_date_is_the_first_event_timestamp(self):
        shop = cc.list_sessions(FIXTURE_HOME)[0]
        # the summary line has no timestamp; the first user event's is used
        self.assertEqual(shop.date, "2026-09-03T09:14:02+00:00")

    def test_project_comes_from_cwd_else_the_folder_name(self):
        by_id = {r.native_id: r for r in cc.list_sessions(FIXTURE_HOME)}
        self.assertEqual((by_id[SHOP].project, by_id[SHOP].project_from_cwd), ("/home/dev/src/shop", True))
        self.assertEqual(by_id[BILLING].project, "/home/dev/src/shop/services/billing")
        self.assertEqual((by_id[NOTES].project, by_id[NOTES].project_from_cwd), ("-home-dev-notes", False))

    def test_date_range_since_inclusive_until_exclusive(self):
        cases = [
            (dt.date(2026, 9, 3), None, [SHOP, STOREFRONT, BILLING, ADMIN, NOTES]),
            (dt.date(2026, 9, 4), None, [STOREFRONT, BILLING, ADMIN, NOTES]),
            (None, dt.date(2026, 9, 15), [SHOP, STOREFRONT]),
            (dt.date(2026, 9, 15), dt.date(2026, 9, 21), [BILLING, ADMIN]),
            (dt.date(2026, 11, 1), None, []),
        ]
        for since, until, want in cases:
            with self.subTest(since=since, until=until):
                self.assertEqual(_ids(cc.list_sessions(FIXTURE_HOME, since=since, until=until)), want)

    def test_since_not_before_until_is_refused(self):
        with self.assertRaises(ValueError):
            cc.list_sessions(FIXTURE_HOME, since=dt.date(2026, 9, 5), until=dt.date(2026, 9, 5))

    def test_project_matches_cwd_as_a_path_prefix(self):
        cases = [
            (["/home/dev/src/shop"], [SHOP, BILLING]),  # the repo and a directory below it
            (["/home/dev/src/shop/"], [SHOP, BILLING]),  # a trailing slash changes nothing
            (["/home/dev/src/shop/services"], [BILLING]),
            (["/home/dev/src/shop-admin"], [ADMIN]),  # a sibling sharing a prefix is not below
            (["/home/dev/src"], [SHOP, STOREFRONT, BILLING, ADMIN]),
            (["/home/dev/src/shop", "/home/dev/notes"], [SHOP, BILLING, NOTES]),
            (["/somewhere/else"], []),
        ]
        for projects, want in cases:
            with self.subTest(projects=projects):
                self.assertEqual(_ids(cc.list_sessions(FIXTURE_HOME, projects=projects)), want)

    def test_encoded_fallback_collides_for_distinct_paths(self):
        # Without a cwd, only the encoded folder name is known, and "/home/dev/notes" and
        # "/home/dev-notes" encode the same: this pins that documented limitation.
        self.assertEqual(cc.encode_project_dir("/home/dev/notes"), cc.encode_project_dir("/home/dev-notes"))
        self.assertEqual(_ids(cc.list_sessions(FIXTURE_HOME, projects=["/home/dev-notes"])), [NOTES])

    def test_a_missing_sessions_directory_is_an_error(self):
        # An empty list would read as "no sessions" for a mistyped home; the missing directory is named.
        absent = os.path.join(tempfile.mkdtemp(), "absent")
        with self.assertRaises(FileNotFoundError) as ctx:
            cc.list_sessions(absent)
        self.assertIn(os.path.join(absent, "projects"), str(ctx.exception))

    def test_a_session_without_timestamps_is_dated_by_mtime(self):
        home = tempfile.mkdtemp()
        proj = os.path.join(home, "projects", "-tmp-x")
        os.makedirs(proj)
        path = os.path.join(proj, f"{NOTES}.jsonl")
        with open(path, "w") as f:
            f.write('{"type":"user","message":{"role":"user","content":"hi"}}\n')
        os.utime(path, (1_790_000_000, 1_790_000_000))
        (ref,) = cc.list_sessions(home)
        self.assertEqual(ref.date, dt.datetime.fromtimestamp(1_790_000_000, dt.UTC).isoformat())

    def test_default_home_is_under_the_users_home(self):
        # The test-wide guard points HOME at an empty temp dir, so the default has no sessions dir.
        with self.assertRaises(FileNotFoundError):
            cc.list_sessions()
        shutil.copytree(FIXTURE_HOME, os.path.join(os.environ["HOME"], ".claude"))
        self.assertEqual(len(cc.list_sessions()), 5)


class ParseSession(unittest.TestCase):
    def _conv(self, sid):
        (ref,) = [r for r in cc.list_sessions(FIXTURE_HOME) if r.native_id == sid]
        return transcript.resolve(ref.as_row())

    def test_turns_skip_sidechain_summary_and_malformed_lines(self):
        conv = self._conv(SHOP)
        texts = [t.content for t in conv.turns]
        self.assertEqual(len(conv.turns), 7)
        self.assertFalse(any("sidechain prompt" in t for t in texts))
        self.assertEqual(conv.date, "2026-09-03T09:14:02+00:00")

    def test_tool_output_is_assistant_context_and_typed_prose_is_human(self):
        conv = self._conv(SHOP)
        t = {x.i: x for x in conv.turns}
        self.assertEqual((t[1].role, t[1].authorship), ("user", "human_prose"))
        self.assertEqual((t[3].role, t[3].authorship), ("assistant", "tool_output"))
        self.assertEqual(t[2].authorship, "model_generated")

    def test_a_long_session_spans_windows_and_a_long_turn_is_trimmed(self):
        # The storefront session: a pasted CI log over 8,000 characters keeps its head and tail and
        # fills the first window alone; long answers fill two more, and no turn is split.
        conv = self._conv(STOREFRONT)
        windows = extract.plan_windows(conv)
        self.assertEqual([[t.i for t in w] for w in windows], [[1], [2, 3, 4], [5, 6, 7, 8]])
        self.assertIn("[...turn truncated...]", conv.turns[0].content)
        self.assertEqual(len(conv.turns[0].content), 7_524)

    def test_windows_are_planned_without_a_model(self):
        self.assertEqual(len(extract.plan_windows(self._conv(SHOP))), 1)

    def test_an_unreadable_row_resolves_to_none(self):
        cases = [
            {},
            {"conversation_id": "cc:x", "native_id": "x", "tool": "claude-code"},
            {
                "conversation_id": "cc:x",
                "native_id": "x",
                "tool": "claude-code",
                "path": "/nonexistent.jsonl",
            },
            {"conversation_id": "cc:x", "native_id": "x", "tool": "other-agent", "path": "/dev/null"},
        ]
        for row in cases:
            with self.subTest(row=row):
                self.assertIsNone(transcript.resolve(row))

    def test_a_block_whose_content_is_not_text_renders_as_its_type(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        block = {"type": "web_search_tool_result", "content": [{"type": "web_search_result", "url": "u"}]}
        _write(
            home,
            "-a",
            SHOP,
            [
                {
                    "type": "user",
                    "sessionId": SHOP,
                    "timestamp": "2026-09-03T09:00:00Z",
                    "cwd": "/a",
                    "promptSource": "typed",
                    "message": {"role": "user", "content": "search for it"},
                },
                {
                    "type": "assistant",
                    "sessionId": SHOP,
                    "timestamp": "2026-09-03T09:00:01Z",
                    "message": {
                        "role": "assistant",
                        "content": [block, {"type": "map_block", "content": {"k": 1}}],
                    },
                },
            ],
        )
        (ref,) = cc.list_sessions(home)
        conv = transcript.resolve(ref.as_row())
        self.assertEqual(conv.turns[1].content, "[web_search_tool_result]\n[map_block]")

    def test_a_parse_error_skips_the_session_with_its_reason(self):
        (ref,) = [r for r in cc.list_sessions(FIXTURE_HOME) if r.native_id == SHOP]
        with mock.patch.object(transcript, "parse_claude_code_session", side_effect=ValueError("bad")):
            self.assertEqual(transcript.resolve_with_reason(ref.as_row()), (None, "cannot parse: ValueError"))


def _write(home, folder, sid, events):
    d = os.path.join(home, "projects", folder)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{sid}.jsonl"), "wb") as f:
        for e in events:
            f.write(e if isinstance(e, bytes) else (json.dumps(e) + "\n").encode())


class EdgeShapes(unittest.TestCase):
    """Session and block shapes at the edges, each pinned to what the reader does with it."""

    def _home(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        return home

    def _user(self, content, ts="2026-09-03T09:00:00Z", **extra):
        return {
            "type": "user",
            "sessionId": SHOP,
            "timestamp": ts,
            "cwd": "/a",
            "promptSource": "typed",
            "message": {"role": "user", "content": content},
            **extra,
        }

    def test_a_block_with_empty_or_no_content_renders_as_its_type(self):
        home = self._home()
        blocks = [{"type": "empty_block", "content": ""}, {"type": "none_block", "content": None}]
        _write(
            home,
            "-a",
            SHOP,
            [
                self._user("hello"),
                {
                    "type": "assistant",
                    "sessionId": SHOP,
                    "timestamp": "2026-09-03T09:00:01Z",
                    "message": {"role": "assistant", "content": blocks},
                },
            ],
        )
        (ref,) = cc.list_sessions(home)
        self.assertEqual(transcript.resolve(ref.as_row()).turns[1].content, "[empty_block]\n[none_block]")

    def test_a_lone_surrogate_only_in_a_cut_report_still_skips_the_session(self):
        # The report is cut in the middle, so the surrogate is gone from the turn text but not from
        # the report span the authorship check reads.
        home = self._home()
        body = "a" * 6_500 + "\ud83d" + "b" * 3_000
        rec = {
            "type": "user",
            "sessionId": SHOP,
            "timestamp": "2026-09-03T09:00:00Z",
            "cwd": "/a",
            "origin": {"kind": "task-notification"},
            "message": {
                "role": "user",
                "content": f"<task-notification><result>{body}</result></task-notification>",
            },
        }
        _write(home, "-a", SHOP, [rec])
        (ref,) = cc.list_sessions(home)
        self.assertEqual(transcript.resolve_with_reason(ref.as_row()), (None, "not valid UTF-8"))

    def test_skip_reasons(self):
        home = self._home()
        _write(home, "-a", SHOP, [])
        _write(
            home, "-a", BILLING, [self._user("only a sidechain", isSidechain=True) | {"sessionId": BILLING}]
        )
        refs = {r.native_id: r for r in cc.list_sessions(home)}
        cases = [
            (refs[SHOP].as_row(), "no readable turns"),
            (refs[BILLING].as_row(), "no readable turns"),
            (
                {**refs[SHOP].as_row(), "path": os.path.join(home, "gone.jsonl")},
                "cannot read: No such file or directory",
            ),
        ]
        for row, reason in cases:
            with self.subTest(reason=reason, path=row["path"]):
                self.assertEqual(transcript.resolve_with_reason(row), (None, reason))

    def test_projects_as_a_file_is_an_error_and_an_empty_projects_dir_lists_nothing(self):
        home = self._home()
        open(os.path.join(home, "projects"), "w").close()
        with self.assertRaises(FileNotFoundError):
            cc.list_sessions(home)
        code = cli.main(["sessions", "--claude-home", home], out=io.StringIO())
        self.assertEqual(code, 2)
        empty = self._home()
        os.makedirs(os.path.join(empty, "projects"))
        self.assertEqual(cc.list_sessions(empty), [])

    def test_a_marker_split_across_two_blocks_is_not_a_marker(self):
        # Blocks are joined with a newline, so a marker broken across two blocks is not recognised:
        # a typed turn stays the person's, an untyped one is unknown. Pinned as current behaviour.
        blocks = [
            {"type": "text", "text": "<local-command-"},
            {"type": "text", "text": "stdout>x</local-command-stdout>"},
        ]
        for source, authorship in (("typed", "human_prose"), (None, "unknown")):
            with self.subTest(promptSource=source):
                rec = {
                    "role": "user",
                    "content": blocks,
                    "native": True,
                    "promptSource": source,
                    "originKind": None,
                    "isMeta": False,
                }
                (t,) = transcript._turns_from_messages([rec], source="local", agent="claude-code")
                self.assertEqual((t.role, t.authorship), ("user", authorship))

    def test_harness_text_before_typed_prose_takes_the_turn(self):
        # Current behaviour, pinned on purpose: typed prose that follows an injected reminder, or that
        # mentions shell-output markup, is classified as harness or tool text and loses the user role.
        for content, authorship in (
            ("<system-reminder>x</system-reminder>\nuse tabs", "template_boilerplate"),
            ("why does <bash-stdout> show up here? use tabs", "tool_output"),
        ):
            with self.subTest(content=content):
                rec = {
                    "role": "user",
                    "content": content,
                    "native": True,
                    "promptSource": "typed",
                    "originKind": None,
                    "isMeta": False,
                }
                (t,) = transcript._turns_from_messages([rec], source="local", agent="claude-code")
                self.assertEqual((t.role, t.authorship), ("assistant", authorship))


class MoreSelection(unittest.TestCase):
    def test_text_that_cannot_encode_skips_the_session_not_the_arm(self):
        # A lone surrogate is valid JSON (an escape) but not encodable text: the session is skipped
        # with its reason, and the arm goes on to the next session.
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        bad = (
            b'{"type": "user", "sessionId": "%s", "timestamp": "2026-09-03T09:00:00Z", "cwd": "/a", '
            b'"promptSource": "typed", "message": {"role": "user", "content": "hi \\ud83d there"}}\n'
            % SHOP.encode()
        )
        _write(home, "-a", SHOP, [bad])
        _write(
            home,
            "-a",
            BILLING,
            [
                {
                    "type": "user",
                    "sessionId": BILLING,
                    "timestamp": "2026-09-04T09:00:00Z",
                    "cwd": "/a",
                    "promptSource": "typed",
                    "message": {"role": "user", "content": "always reuse the key"},
                },
            ],
        )
        refs = cc.list_sessions(home)
        self.assertEqual(transcript.resolve_with_reason(refs[0].as_row()), (None, "not valid UTF-8"))
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out)
        metrics, ok = extract.run(
            [r.as_row() for r in refs],
            out,
            ScriptedClient(),
            arm="facts",
            cache_dir=os.path.join(out, "cache"),
            log=lambda m: None,
        )
        self.assertTrue(ok)
        self.assertIsNone(metrics.get("fatal_error"))
        self.assertEqual(
            metrics["skipped_sessions"], [{"conversation_id": f"cc:{SHOP}", "reason": "not valid UTF-8"}]
        )
        self.assertEqual(metrics["conversations_run"], 1)

    def test_an_undecodable_session_is_marked_in_the_sessions_table(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        _write(
            home,
            "-a",
            SHOP,
            [
                {
                    "type": "user",
                    "sessionId": SHOP,
                    "timestamp": "2026-09-03T09:00:00Z",
                    "cwd": "/a",
                    "promptSource": "typed",
                    "message": {"role": "user", "content": "hello"},
                },
                b'{"type": "user", "message": {"role": "user", "content": "caf\xe9"}}\n',
            ],
        )
        out = io.StringIO()
        self.assertEqual(cli.main(["sessions", "--claude-home", home], out=out), 0)
        (row,) = [ln for ln in out.getvalue().splitlines() if SHOP in ln]
        self.assertTrue(row.endswith("skipped: not valid UTF-8"), row)

    def test_an_undecodable_session_is_skipped(self):
        home = tempfile.mkdtemp()
        _write(
            home,
            "-tmp-x",
            SHOP,
            [
                {
                    "type": "user",
                    "sessionId": SHOP,
                    "timestamp": "2026-09-03T09:00:00Z",
                    "cwd": "/tmp/x",
                    "promptSource": "typed",
                    "message": {"role": "user", "content": "hello"},
                },
                b'{"type": "user", "message": {"role": "user", "content": "caf\xe9"}}\n',
            ],
        )
        (ref,) = cc.list_sessions(home)
        self.assertIsNone(transcript.resolve(ref.as_row()))

    def test_a_no_cwd_session_matches_only_its_exact_folder(self):
        home = tempfile.mkdtemp()
        _write(home, "-home-dev-src-shop", SHOP, [{"type": "user", "timestamp": "2026-09-03T09:00:00Z"}])
        _write(
            home, "-home-dev-src-shop-admin2", ADMIN, [{"type": "user", "timestamp": "2026-09-04T09:00:00Z"}]
        )
        self.assertEqual(_ids(cc.list_sessions(home, projects=["/home/dev/src/shop"])), [SHOP])
        self.assertEqual(_ids(cc.list_sessions(home, projects=["/home/dev/src/shop-admin2"])), [ADMIN])

    def test_project_is_the_first_cwd(self):
        home = tempfile.mkdtemp()
        _write(
            home,
            "-a",
            SHOP,
            [
                # The first cwd arrives before any timestamp, so the scan reads past it.
                {"type": "user", "cwd": "/first"},
                {"type": "user", "timestamp": "2026-09-03T09:01:00Z", "cwd": "/second"},
            ],
        )
        self.assertEqual(cc.list_sessions(home)[0].project, "/first")

    def test_since_is_a_local_calendar_day(self):
        home = tempfile.mkdtemp()
        _write(home, "-a", SHOP, [{"type": "user", "timestamp": "2026-09-14T20:00:00Z", "cwd": "/a"}])
        old = os.environ.get("TZ")
        os.environ["TZ"] = "Asia/Kolkata"
        time.tzset()
        try:
            self.assertEqual(_ids(cc.list_sessions(home, since=dt.date(2026, 9, 15))), [SHOP])
        finally:
            os.environ["TZ"] = old or "UTC"
            time.tzset()
