# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The whole pipeline over the synthetic session home with a scripted model: both arms, the merge,
the run directory layout, the schema of every merged memory, and the privacy guards."""

from __future__ import annotations

import io
import json
import os
import stat
import tempfile
import unittest

import jsonschema

from agent_memory_engine import cli, pipeline, run
from agent_memory_engine.paths import UnsafePathError
from agent_memory_engine.sources import list_sessions
from scripted import ScriptedClient

FIXTURE_HOME = os.path.join(os.path.dirname(__file__), "fixtures", "claude-home")
with open(os.path.join(os.path.dirname(pipeline.__file__), "schema", "memory.schema.json")) as _f:
    SCHEMA = json.load(_f)


class WholePipeline(unittest.TestCase):
    def test_a_disposition_note_reaches_the_merged_memory(self):
        # The model marks one memory borderline with a reason; the reason survives the arm, the
        # reconcile step and the merge into memories.jsonl.
        note = "seen once, unclear schedule"

        class _Borderline(ScriptedClient):
            def extract(self, system_prompt, text):
                r = super().extract(system_prompt, text)
                if not text.startswith("--- SESSION METADATA ---"):
                    return r
                obj = json.loads(r.raw_text)
                for m in obj.get("memories", []):
                    if m["statement"].startswith("Phrase code review comments"):
                        m["disposition"] = {"level": "borderline", "note": note}
                return type(r)(json.dumps(obj), r.cost_usd, r.duration_ms, r.model_id, ok=r.ok)

        result = run(
            self.sessions, client=_Borderline(), data_dir=self.data_dir, run_id="n", log=lambda m: None
        )
        self.assertTrue(result.ok)
        with open(result.memories_path, encoding="utf-8") as f:
            (m,) = [json.loads(line) for line in f if "Phrase code review comments" in line]
        self.assertEqual(m["disposition"], {"level": "borderline", "note": note})
        jsonschema.validate(m, SCHEMA)

    def setUp(self):
        self.data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        self.sessions = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])

    def test_run_writes_the_layout_and_schema_valid_memories(self):
        result = run(
            self.sessions, client=ScriptedClient(), data_dir=self.data_dir, run_id="r1", log=lambda m: None
        )
        self.assertTrue(result.ok, result.metrics)
        rd = result.run_dir
        for rel in (
            "sessions.jsonl",
            "memories.jsonl",
            "metrics.json",
            "arms/facts/memories.jsonl",
            "arms/facts/metrics.json",
            "arms/preferences/memories.jsonl",
            "arms/preferences/metrics.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(rd, rel)), rel)
        with open(result.memories_path, encoding="utf-8") as f:
            mems = [json.loads(line) for line in f]
        self.assertGreaterEqual(len(mems), 3)
        for m in mems:
            jsonschema.validate(m, SCHEMA)
            self.assertTrue(any(m["tags"].values()))
        # every quote is verbatim from its session, and arms are named in the output
        self.assertEqual({e["arm"] for m in mems for e in m["evidence"]}, {"facts", "preferences"})
        with open(os.path.join(rd, "metrics.json")) as f:
            summary = json.load(f)
        self.assertEqual(summary["sessions_selected"], 2)
        self.assertEqual(summary["stages"]["merge"]["memories_merged"], len(mems))
        self.assertEqual(summary["skipped_sessions"], [])

    def test_every_output_line_carries_one_typed_tags_field(self):
        # Merged memories and each arm's lines name tags once, as `tags`: the six-axis object the
        # schema describes. One memory answers its tags in the list form, which lands typed too.
        class _OneListAnswer(ScriptedClient):
            def extract(self, system_prompt, text):
                r = super().extract(system_prompt, text)
                if not text.startswith("--- SESSION METADATA ---"):
                    return r
                obj = json.loads(r.raw_text)
                for m in obj.get("memories", []):
                    if m["statement"].startswith("Turbopack refuses"):
                        m["tags"] = ["domain:developer-tooling", "tech:turbopack", "symlink"]
                return type(r)(json.dumps(obj), r.cost_usd, r.duration_ms, r.model_id, ok=r.ok)

        result = run(
            self.sessions, client=_OneListAnswer(), data_dir=self.data_dir, run_id="t", log=lambda m: None
        )
        self.assertTrue(result.ok, result.metrics)
        files = [
            result.memories_path,
            os.path.join(result.run_dir, "arms", "facts", "memories.jsonl"),
            os.path.join(result.run_dir, "arms", "preferences", "memories.jsonl"),
        ]
        axes = ["domain", "activity", "component", "tech", "concept", "keyword"]
        tags_schema = SCHEMA["properties"]["tags"]
        seen = 0
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    m = json.loads(line)
                    seen += 1
                    self.assertEqual([k for k in m if "tag" in k], ["tags"], path)
                    self.assertEqual(sorted(m["tags"]), sorted(axes), path)
                    jsonschema.validate(m["tags"], tags_schema)
        self.assertGreaterEqual(seen, 6)
        self.assertEqual(sorted(tags_schema["required"]), sorted(axes))

    def test_a_rerun_replays_the_cache_and_calls_nothing(self):
        client = ScriptedClient()
        run(self.sessions, client=client, data_dir=self.data_dir, run_id="a", log=lambda m: None)
        calls = client.calls
        again = run(self.sessions, client=client, data_dir=self.data_dir, run_id="b", log=lambda m: None)
        self.assertTrue(again.ok)
        self.assertEqual(client.calls, calls)

    def test_a_cached_rerun_prints_what_it_spent_beside_what_it_replayed(self):
        # A fully cached re-run makes no model call, so it spends nothing. Its summary says so, and
        # shows the replayed answers' original price beside it rather than as money spent.
        class _Priced(ScriptedClient):
            def _ok(self, obj):
                r = super()._ok(obj)
                r.cost_usd = 0.01
                return r

        client = _Priced()
        first = run(self.sessions, client=client, data_dir=self.data_dir, run_id="p1", log=lambda m: None)
        calls = client.calls
        again = run(self.sessions, client=client, data_dir=self.data_dir, run_id="p2", log=lambda m: None)
        self.assertEqual(client.calls, calls)
        worth = f"${calls * 0.01:.2f}"

        def summary(result):
            out = io.StringIO()
            cli._print_summary(result.metrics, result.run_dir, out)
            return out.getvalue().splitlines()

        first_lines, lines = summary(first), summary(again)
        self.assertTrue(first_lines[-1].endswith(f"total {worth}"), first_lines[-1])
        self.assertTrue(lines[-1].endswith(f"total $0.00 spent (cached answers worth {worth})"), lines[-1])
        for line in lines[:-1]:
            self.assertRegex(line, r"\$0\.00 spent \(cached answers worth \$\d+\.\d\d\)$")
        self.assertEqual(again.metrics["spent_usd_total"], 0.0)
        self.assertEqual(again.metrics["cost_usd_total"], first.metrics["cost_usd_total"])
        self.assertEqual(first.metrics["spent_usd_total"], first.metrics["cost_usd_total"])

    def test_the_data_dir_is_owner_only(self):
        os.umask(0o022)  # a library caller with a permissive umask
        try:
            run(
                self.sessions, client=ScriptedClient(), data_dir=self.data_dir, run_id="m", log=lambda m: None
            )
        finally:
            os.umask(0o022)
        self.assertEqual(stat.S_IMODE(os.stat(self.data_dir).st_mode), 0o700)

    def test_an_existing_data_dir_is_tightened_to_owner_only(self):
        os.makedirs(self.data_dir, mode=0o755)
        os.chmod(self.data_dir, 0o755)
        run(self.sessions, client=ScriptedClient(), data_dir=self.data_dir, run_id="t", log=lambda m: None)
        self.assertEqual(stat.S_IMODE(os.stat(self.data_dir).st_mode), 0o700)

    def test_a_data_dir_inside_a_git_worktree_is_refused(self):
        # A worktree's .git is a file, not a directory.
        repo = tempfile.mkdtemp()
        with open(os.path.join(repo, ".git"), "w") as f:
            f.write("gitdir: /elsewhere/.git/worktrees/x\n")
        with self.assertRaises(UnsafePathError):
            run(
                self.sessions, client=ScriptedClient(), data_dir=os.path.join(repo, "out"), log=lambda m: None
            )

    def test_a_data_dir_inside_a_git_repo_is_refused(self):
        repo = tempfile.mkdtemp()
        os.makedirs(os.path.join(repo, ".git"))
        with self.assertRaises(UnsafePathError):
            run(
                self.sessions, client=ScriptedClient(), data_dir=os.path.join(repo, "out"), log=lambda m: None
            )

    def test_an_existing_run_id_is_refused(self):
        run(self.sessions, client=ScriptedClient(), data_dir=self.data_dir, run_id="dup", log=lambda m: None)
        with self.assertRaises(ValueError):
            run(
                self.sessions,
                client=ScriptedClient(),
                data_dir=self.data_dir,
                run_id="dup",
                log=lambda m: None,
            )

    def test_no_sessions_is_refused(self):
        with self.assertRaises(ValueError):
            run([], client=ScriptedClient(), data_dir=self.data_dir, log=lambda m: None)

    def test_a_model_failure_stops_before_the_merge(self):
        result = run(
            self.sessions,
            client=ScriptedClient(fail_on="preferences"),
            data_dir=self.data_dir,
            run_id="stop",
            log=lambda m: None,
        )
        self.assertFalse(result.ok)
        self.assertNotIn("merge", result.metrics["stages"])
        self.assertTrue(result.metrics["stages"]["preferences"]["fatal_error"])
        self.assertFalse(os.path.exists(result.memories_path))

    def test_a_default_gemini_run_without_a_key_is_refused_before_any_work(self):
        from agent_memory_engine.providers import GeminiKeyError

        with self.assertRaises(GeminiKeyError):
            run(self.sessions, data_dir=self.data_dir, log=lambda m: None)
        self.assertFalse(os.path.exists(os.path.join(self.data_dir, "runs")))


class SameRunIdTwice(unittest.TestCase):
    def test_a_concurrent_same_run_id_leaves_the_live_run_untouched(self):
        import threading

        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        first = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        second = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop-admin"])
        entered, release = threading.Event(), threading.Event()

        class _Blocking(ScriptedClient):
            def extract(self, system_prompt, text):
                entered.set()
                release.wait(10)
                return super().extract(system_prompt, text)

        result = {}
        t = threading.Thread(
            target=lambda: result.update(
                r=run(first, client=_Blocking(), data_dir=data_dir, run_id="same", log=lambda m: None)
            )
        )
        t.start()
        self.assertTrue(entered.wait(10))
        with self.assertRaises(RuntimeError):
            run(second, client=ScriptedClient(), data_dir=data_dir, run_id="same", log=lambda m: None)
        release.set()
        t.join(30)
        self.assertTrue(result["r"].ok)
        with open(os.path.join(result["r"].run_dir, "sessions.jsonl")) as f:
            ids = [json.loads(line)["native_id"] for line in f]
        self.assertEqual(ids, [s.native_id for s in first])

    def _clients(self):
        c = ScriptedClient()
        return pipeline.Clients(c, c, c)

    def test_a_second_selection_is_refused_for_the_same_run_id(self):
        # (a) one arm over shop, then the other arm over shop-admin under the same id: refused, so a
        # merge can never fold two different selections together.
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        admin = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop-admin"])
        kw = {"data_dir": data_dir, "run_id": "x", "log": lambda m: None}
        self.assertTrue(pipeline.run_arms(shop, self._clients(), arms=("facts",), **kw)["_ok"])
        with self.assertRaises(pipeline.RunConflictError):
            pipeline.run_arms(admin, self._clients(), arms=("preferences",), **kw)
        self.assertFalse(os.path.exists(os.path.join(data_dir, "runs", "x", "arms", "preferences")))
        # the same selection may redo an arm
        self.assertTrue(pipeline.run_arms(shop, self._clients(), arms=("preferences",), **kw)["_ok"])

    def test_the_selection_compare_ignores_order_and_refuses_a_superset(self):
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        self.assertGreater(len(shop), 1)
        kw = {"data_dir": data_dir, "run_id": "o", "log": lambda m: None}
        self.assertTrue(pipeline.run_arms(shop, self._clients(), arms=("facts",), **kw)["_ok"])
        # the same sessions in another order are the same selection
        self.assertTrue(
            pipeline.run_arms(list(reversed(shop)), self._clients(), arms=("preferences",), **kw)["_ok"]
        )
        # one more session is a different selection
        more = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop", "/home/dev/src/shop-admin"])
        with self.assertRaises(pipeline.RunConflictError):
            pipeline.run_arms(more, self._clients(), arms=("facts",), **kw)

    def test_a_corrupt_sessions_file_is_a_conflict_naming_it(self):
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        run_dir = os.path.join(data_dir, "runs", "k")
        os.makedirs(run_dir)
        path = os.path.join(run_dir, "sessions.jsonl")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        for body in ('{"x": 1}\n', "not json\n"):
            with self.subTest(body=body):
                with open(path, "w") as f:
                    f.write(body)
                with self.assertRaises(pipeline.RunConflictError) as ctx:
                    pipeline.run_arms(
                        shop,
                        self._clients(),
                        arms=("facts",),
                        data_dir=data_dir,
                        run_id="k",
                        log=lambda m: None,
                    )
                self.assertIn(path, str(ctx.exception))

    def test_same_ids_from_another_home_are_refused(self):
        # The same session ids read from other files are a different selection.
        import dataclasses

        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        kw = {"data_dir": data_dir, "run_id": "h", "log": lambda m: None}
        pipeline.run_arms(shop, self._clients(), arms=("facts",), **kw)
        elsewhere = [dataclasses.replace(r, path=f"/elsewhere/{r.native_id}.jsonl") for r in shop]
        with self.assertRaises(pipeline.RunConflictError):
            pipeline.run_arms(elsewhere, self._clients(), arms=("preferences",), **kw)

    def test_a_subset_is_refused(self):
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        kw = {"data_dir": data_dir, "run_id": "u", "log": lambda m: None}
        pipeline.run_arms(shop, self._clients(), arms=("facts",), **kw)
        with self.assertRaises(pipeline.RunConflictError):
            pipeline.run_arms(shop[:-1], self._clients(), arms=("preferences",), **kw)

    def test_unreadable_sessions_files(self):
        # Each of these names the file and says it is unreadable; a trailing blank line is fine.
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        good = "".join(json.dumps(r.as_row()) + "\n" for r in shop)
        cases = [
            ("", True),
            ('{"conversation_id": null}\n', True),
            ("[1]\n", True),
            ("5\n", True),
            (None, True),
            (good + "\n", False),
        ]
        for n, (body, unreadable) in enumerate(cases):
            with self.subTest(body=body):
                run_dir = os.path.join(data_dir, "runs", f"b{n}")
                os.makedirs(run_dir)
                path = os.path.join(run_dir, "sessions.jsonl")
                if body is None:
                    os.makedirs(path)  # a directory where the file should be
                else:
                    with open(path, "w") as f:
                        f.write(body)
                kw = {"data_dir": data_dir, "run_id": f"b{n}", "log": lambda m: None}
                if unreadable:
                    with self.assertRaises(pipeline.RunConflictError) as ctx:
                        pipeline.run_arms(shop, self._clients(), arms=("facts",), **kw)
                    self.assertIn(f"unreadable {path}", str(ctx.exception))
                else:
                    self.assertTrue(pipeline.run_arms(shop, self._clients(), arms=("facts",), **kw)["_ok"])

    def test_run_refuses_a_run_id_that_already_has_metrics(self):
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        run_dir = os.path.join(data_dir, "runs", "m")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            f.write("{}")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        with self.assertRaises(pipeline.RunConflictError):
            run(shop, client=ScriptedClient(), data_dir=data_dir, run_id="m", log=lambda m: None)

    def test_a_stale_sessions_file_with_no_arms_takes_the_same_selection(self):
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        run_dir = os.path.join(data_dir, "runs", "s")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "sessions.jsonl"), "w") as f:
            f.writelines(json.dumps(r.as_row()) + "\n" for r in shop)
        self.assertTrue(
            run(shop, client=ScriptedClient(), data_dir=data_dir, run_id="s", log=lambda m: None).ok
        )

    def test_run_refuses_a_run_id_that_already_has_arm_output(self):
        # (b) an extract under an id, then `run` under the same id: refused, the arm output kept.
        data_dir = os.path.join(os.environ["HOME"], ".agent-memory-engine")
        shop = list_sessions(FIXTURE_HOME, projects=["/home/dev/src/shop"])
        pipeline.run_arms(
            shop, self._clients(), arms=("facts",), data_dir=data_dir, run_id="y", log=lambda m: None
        )
        facts = os.path.join(data_dir, "runs", "y", "arms", "facts", "memories.jsonl")
        with open(facts, "rb") as f:
            before = f.read()
        self.assertTrue(before)
        with self.assertRaises(pipeline.RunConflictError):
            run(shop, client=ScriptedClient(), data_dir=data_dir, run_id="y", log=lambda m: None)
        with open(facts, "rb") as f:
            self.assertEqual(f.read(), before)
