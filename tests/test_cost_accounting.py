# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""What a run spent, what it replayed from the answer cache, and the prompt-cache storage it held.
Each is counted once, and in every stage spent plus replayed equals the stage's cost: on a fresh,
partly cached, fully cached or stopped run, and on `extract` then `merge`."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import re
import types
import unittest
from unittest import mock

from agent_memory_engine import cli, merge, pipeline, run
from agent_memory_engine.cache import Cache
from agent_memory_engine.providers import build_client, gemini
from agent_memory_engine.providers.base import ExtractResult
from agent_memory_engine.sources import list_sessions
from scripted import ScriptedClient

FIXTURE_HOME = os.path.join(os.path.dirname(__file__), "fixtures", "claude-home")
PRICE = 0.01
SHOP, STOREFRONT, BILLING, ADMIN, NOTES = "2026-09-03", "2026-09-08", "2026-09-15", "2026-09-20", "2026-10-01"
SPENT = re.compile(r"\$(\d+\.\d\d) spent \(cached answers worth \$(\d+\.\d\d)\)$")


class Priced(ScriptedClient):
    """The scripted model at PRICE per answer. It fails the extraction of any session dated in
    `fail_dates`, and every merge synthesis call when `fail_synthesis` is set."""

    def __init__(self, fail_dates=(), fail_synthesis=False):
        super().__init__()
        self.fail_dates = set(fail_dates)
        self.fail_synthesis = fail_synthesis

    def _ok(self, obj):
        r = super()._ok(obj)
        r.cost_usd = PRICE
        return r

    def extract(self, system_prompt, text):
        failing = (
            text.startswith("--- SESSION METADATA ---")
            and json.loads(text.splitlines()[1])["session_date"][:10] in self.fail_dates
        ) or (self.fail_synthesis and text.startswith("Unify the following"))
        if failing:
            self.calls += 1
            return ExtractResult("", 0.0, 1, self.model_id, ok=False, error="scripted failure")
        return super().extract(system_prompt, text)


def _sessions(*dates):
    return [s for s in list_sessions(FIXTURE_HOME) if s.date[:10] in dates]


def _data_dir():
    return os.path.join(os.environ["HOME"], ".agent-memory-engine")


def _run(sessions, client, run_id, workers=None, log=None):
    return run(
        sessions,
        client=client,
        data_dir=_data_dir(),
        run_id=run_id,
        workers=workers,
        log=log or (lambda m: None),
    )


def _arm_metrics(run_id, arm):
    with open(os.path.join(_data_dir(), "runs", run_id, "arms", arm, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


class _SplitAssertions(unittest.TestCase):
    def assertSplit(self, stage, hits):
        self.assertGreater(hits, 0)
        self.assertEqual(round(stage["spent_usd"] + stage["cached_answers_usd"], 6), stage["cost_usd"])
        self.assertEqual(stage["cached_answers_usd"], round(hits * PRICE, 6))


class StorageIsCountedOnce(unittest.TestCase):
    """Real Gemini clients from make_clients, with the HTTP calls faked and a clock that moves only
    when a call is made. Every prompt-cache resource is priced once, on its own lifetime."""

    def test_the_stages_storage_sums_to_each_resource_priced_once(self):
        clock = [1000.0]
        fake_time = types.SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: None, time=lambda: 0.0)
        names = itertools.count(1)
        scripted = ScriptedClient()
        made = {}
        real_make = pipeline.make_clients

        def capture(*args, **kwargs):
            made["clients"] = real_make(*args, **kwargs)
            return made["clients"]

        def api_call(self, path, method="GET", payload=None, timeout_s=None):
            clock[0] += 5.0
            if method == "POST":
                return {
                    "name": f"cachedContents/c{next(names)}",
                    "usageMetadata": {"totalTokenCount": 100_000},
                }
            return {}

        def extract_once(self, system_prompt, conversation_text, cache_name=None):
            clock[0] += 60.0
            return scripted.extract(system_prompt, conversation_text), False, None

        with (
            mock.patch.object(pipeline, "make_clients", capture),
            mock.patch.object(gemini, "time", fake_time),
            mock.patch.object(gemini.GeminiClient, "_api_call", api_call),
            mock.patch.object(gemini.GeminiClient, "_extract_once", extract_once),
        ):
            result = run(
                _sessions(SHOP, BILLING),
                api_key="fake-key-for-test",
                data_dir=_data_dir(),
                run_id="storage",
                log=lambda m: None,
            )
        self.assertTrue(result.ok, result.metrics)
        clients = made["clients"]
        resources = {e["name"]: e for c in (clients.extract, clients.merge) for e in c.explicit_caches}
        self.assertEqual(len(resources), 5)  # two extraction prompts, reconcile, synthesis, dedup
        bill = sum(gemini.gemini_cache_storage_cost(e["tokens"], e["lifetime_s"]) for e in resources.values())
        stages = result.metrics["stages"]
        # The scripted answers cost nothing, so every stage's cost is its share of the storage.
        self.assertAlmostEqual(sum(s["cost_usd"] for s in stages.values()), bill, delta=3e-6)
        self.assertAlmostEqual(result.metrics["cost_usd_total"], bill, delta=3e-6)
        self.assertEqual(result.metrics["spent_usd_total"], result.metrics["cost_usd_total"])


class ProgressLog(unittest.TestCase):
    def test_a_cached_rerun_logs_what_each_session_spent(self):
        sessions = _sessions(SHOP, BILLING)
        _run(sessions, Priced(), "first")
        lines: list[str] = []
        _run(sessions, Priced(), "again", log=lines.append)
        progress = [ln for ln in lines if re.match(r"\s+(facts|preferences): \[", ln)]
        self.assertEqual(len(progress), 4, lines)
        for ln in progress:
            self.assertIn(" calls=0 ", ln)
            self.assertRegex(ln, r"\| \$0\.00 spent \(cached answers worth \$\d+\.\d\d\)$")


def _arm(cost, cached, storage):
    """An arm's metrics, reduced to the fields its run-summary entry reads."""
    return {
        "conversations_run": 1,
        "conversations_skipped": 0,
        "windows_total": 1,
        "windows_unparseable": 0,
        "reconcile_unparseable": 0,
        "reconcile_overflow": 0,
        "memories_total": 1,
        "cost_usd_total": cost,
        "cached_answers_usd": cached,
        "explicit_cache_storage_usd": storage,
    }


class Rounding(unittest.TestCase):
    def test_spent_plus_cached_equals_cost_to_the_microdollar(self):
        m = _arm(4.981533787372589, 4.6657707467955065, 8.158e-05)
        stage = pipeline.arm_stage(m)
        self.assertEqual(round(stage["spent_usd"] + stage["cached_answers_usd"], 6), stage["cost_usd"])

    def test_the_run_totals_add_up_from_the_rounded_stages(self):
        arms = {
            "facts": _arm(4.981533787372589, 4.6657707467955065, 8.158e-05),
            "preferences": _arm(1.2345674999, 0.3333335001, 1.5e-06),
        }
        s = pipeline._summary([], arms, None, True)
        stages = s["stages"].values()
        self.assertEqual(s["cost_usd_total"], round(sum(st["cost_usd"] for st in stages), 6))
        self.assertEqual(s["spent_usd_total"], round(sum(st["spent_usd"] for st in stages), 6))
        self.assertEqual(
            s["cached_answers_usd_total"], round(sum(st["cached_answers_usd"] for st in stages), 6)
        )
        self.assertEqual(round(s["spent_usd_total"] + s["cached_answers_usd_total"], 6), s["cost_usd_total"])


class StoppedRuns(_SplitAssertions):
    def test_an_extract_fatal_after_cache_hits(self):
        _run(_sessions(SHOP), Priced(), "warm")
        result = _run(_sessions(SHOP, BILLING), Priced(fail_dates={BILLING}), "stopped")
        self.assertFalse(result.ok)
        self.assertSplit(result.metrics["stages"]["facts"], _arm_metrics("stopped", "facts")["cache_hits"])

    def test_a_parallel_extract_abandoned_after_the_fatal(self):
        _run(_sessions(ADMIN, NOTES), Priced(), "warm")
        result = _run(_sessions(BILLING, ADMIN, NOTES), Priced(fail_dates={BILLING}), "stopped", workers=6)
        self.assertFalse(result.ok)
        arm = _arm_metrics("stopped", "facts")
        self.assertEqual(arm["conversations_run"], 0)  # every cache hit came from an abandoned session
        self.assertSplit(result.metrics["stages"]["facts"], arm["cache_hits"])

    def _warm_merge_cache(self):
        puts = []
        real_put = Cache.put

        def put(cache, fp, record):
            if os.path.basename(cache.root) == "merge":
                puts.append(cache._path(fp))
            real_put(cache, fp, record)

        with mock.patch.object(Cache, "put", put):
            _run(_sessions(SHOP, BILLING), Priced(), "warm")
        return puts  # the synthesis groups in order, then the dedup

    def test_a_merge_failing_after_cached_groups(self):
        puts = self._warm_merge_cache()
        self.assertGreaterEqual(len(puts), 3)
        os.remove(puts[1])  # group 1 now misses the cache and its call fails; group 0 replays
        result = _run(_sessions(SHOP, BILLING), Priced(fail_synthesis=True), "stopped")
        self.assertFalse(result.ok)
        stage = result.metrics["stages"]["merge"]
        self.assertIsNotNone(stage["fatal_error"])
        self.assertSplit(stage, stage["cache_hits"])

    def test_a_parallel_merge_abandons_cached_groups_after_the_fatal(self):
        puts = self._warm_merge_cache()
        os.remove(puts[0])  # group 0 misses and fails; the groups after it replay and are abandoned
        run_dir = os.path.join(_data_dir(), "runs", "warm")
        arm_paths = {a: os.path.join(run_dir, "arms", a, "memories.jsonl") for a in ("facts", "preferences")}
        out_dir = os.path.join(_data_dir(), "runs", "parallel")
        os.makedirs(out_dir)
        metrics, ok = merge.run(
            arm_paths,
            out_dir,
            Priced(fail_synthesis=True),
            cache_dir=os.path.join(_data_dir(), "cache", "merge"),
            workers=3,
            log=lambda m: None,
        )
        self.assertFalse(ok)
        self.assertTrue(any(g["status"] == "abandoned" for g in metrics["partial_groups"]))
        self.assertSplit(pipeline.merge_stage(metrics), metrics["cache_hits"])


class StorageBucket(unittest.TestCase):
    """Holding the prompt cache is money spent: it is in cost_usd and spent_usd, never in
    cached_answers_usd."""

    def test_storage_recorded_by_the_stage(self):
        for storage in (0.003, 0.0):
            with self.subTest(storage=storage):
                stage = pipeline.arm_stage(_arm(0.10, 0.04, storage))
                self.assertEqual(stage["cost_usd"], round(0.10 + storage, 6))
                self.assertEqual(stage["spent_usd"], round(0.06 + storage, 6))
                self.assertEqual(stage["cached_answers_usd"], 0.04)

    def test_storage_priced_once_for_the_run(self):
        held = {"name": "cachedContents/x", "tokens": 100_000, "lifetime_s": 1800.0, "live": False}
        storage = gemini.gemini_cache_storage_cost(100_000, 1800.0)
        for caching in (True, False):
            with self.subTest(explicit_cache=caching):
                client = types.SimpleNamespace(explicit_cache=caching, explicit_caches=[held])
                clients = pipeline.Clients(client, client, client)
                arms = {"facts": _arm(0.10, 0.04, 0.0), "preferences": _arm(0.10, 0.04, 0.0)}
                s = pipeline._summary([], arms, None, True, clients=clients)
                held_usd = storage if caching else 0.0
                self.assertEqual(s["cost_usd_total"], round(0.20 + held_usd, 6))
                self.assertEqual(s["spent_usd_total"], round(0.12 + held_usd, 6))
                self.assertEqual(s["cached_answers_usd_total"], 0.08)


class CostFormat(unittest.TestCase):
    def test_the_cost_table(self):
        for cost, spent, want in [
            (0.10, 0.10, "$0.10"),
            (0.10, 0.098, "$0.10"),
            (0.12, 0.07, "$0.07 spent (cached answers worth $0.05)"),
            (0.10, None, "$0.10"),
            (0.006, 0.002, "$0.00 spent (cached answers worth $0.00)"),
        ]:
            with self.subTest(cost=cost, spent=spent):
                self.assertEqual(cli._cost(cost, spent), want)

    def test_a_partly_cached_run_prints_the_spent_form(self):
        _run(_sessions(SHOP, BILLING), Priced(), "two")
        result = _run(_sessions(SHOP, STOREFRONT, BILLING, ADMIN, NOTES), Priced(), "all")
        out = io.StringIO()
        cli._print_summary(result.metrics, result.run_dir, out)
        facts = out.getvalue().splitlines()[0]
        self.assertTrue(facts.startswith("facts "), facts)
        spent, worth = SPENT.search(facts).groups()
        self.assertGreater(float(spent), 0)
        self.assertGreater(float(worth), 0)


class ExtractThenMerge(unittest.TestCase):
    def _main(self, client_type, *argv):
        out = io.StringIO()
        clients = lambda provider, **kw: pipeline.Clients(*(client_type(),) * 3)  # noqa: E731
        with (
            mock.patch.object(pipeline, "make_clients", clients),
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(cli.main(list(argv), out=out), 0)
        return out.getvalue().splitlines()

    def _extract_and_merge(self, client_type, run_id):
        sel = ["--claude-home", FIXTURE_HOME, "--project", "/home/dev/src/shop", "--data-dir", _data_dir()]
        lines = []
        for arm in ("facts", "preferences"):
            lines += self._main(client_type, "extract", "--arm", arm, *sel, "--run-id", run_id)[:1]
        lines += self._main(client_type, "merge", "--data-dir", _data_dir(), "--run-id", run_id)[:1]
        return lines

    def test_replayed_into_a_new_run_id_prints_the_spent_form(self):
        self._extract_and_merge(Priced, "x")
        for line in self._extract_and_merge(Priced, "y"):
            m = SPENT.search(line)
            self.assertIsNotNone(m, line)
            self.assertEqual(m.group(1), "0.00")

    def test_a_cached_answer_that_cost_nothing_prints_the_plain_form(self):
        self._extract_and_merge(ScriptedClient, "x")
        for line in self._extract_and_merge(ScriptedClient, "y"):
            self.assertTrue(line.endswith(", $0.00"), line)


class Boom(BaseException):
    """Raised from inside a model call to stop a run mid-arm. Not an Exception, so nothing in the
    pipeline turns it into a counted failure."""


class FakeGemini:
    """Real GeminiClient objects with their HTTP faked. Each prompt cache holds 1M tokens, each model
    call takes `call_s` (600 s by default) on a clock that moves only when a call is made, the
    answers are scripted, and every cache create and delete is logged."""

    TOKENS = 1_000_000

    def __init__(self, fail_facts_on=(), boom_on_call=None, call_s=600.0):
        self.call_s = call_s
        self.clock = [1000.0]
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.fail_facts_on = set(fail_facts_on)
        self.boom_on_call = boom_on_call
        self.calls = 0
        self.scripted = ScriptedClient()
        self.clients = []

    @contextlib.contextmanager
    def patched(self):
        fake = self
        real_make = pipeline.make_clients
        names = itertools.count(1)

        def make_clients(*args, **kwargs):
            clients = real_make(*args, **kwargs)
            fake.clients.append(clients)
            return clients

        def api_call(client, path, method="GET", payload=None, timeout_s=None):
            fake.clock[0] += 5.0
            if method == "POST":
                name = f"cachedContents/c{next(names)}"
                fake.created.append(name)
                return {"name": name, "usageMetadata": {"totalTokenCount": fake.TOKENS}}
            if method == "DELETE":
                fake.deleted.append(path)
            return {}

        def extract_once(client, system_prompt, text, cache_name=None):
            fake.clock[0] += fake.call_s
            fake.calls += 1
            if fake.calls == fake.boom_on_call:
                raise Boom()
            facts_window = (
                text.startswith("--- SESSION METADATA ---") and "WHAT THIS ARM CAPTURES" in system_prompt
            )
            if facts_window and json.loads(text.splitlines()[1])["session_date"][:10] in fake.fail_facts_on:
                return ExtractResult("", 0.0, 1, client.model_id, ok=False, error="scripted"), False, None
            return fake.scripted.extract(system_prompt, text), False, None

        clock = types.SimpleNamespace(monotonic=lambda: fake.clock[0], sleep=lambda s: None, time=lambda: 0.0)
        with (
            mock.patch.object(pipeline, "make_clients", make_clients),
            mock.patch.object(gemini, "time", clock),
            mock.patch.object(gemini.GeminiClient, "_api_call", api_call),
            mock.patch.object(gemini.GeminiClient, "_extract_once", extract_once),
        ):
            yield self


def _bill(*clients, only=None):
    """Each prompt-cache resource of `clients` priced once on its lifetime, with its role."""
    out = {}
    for c in clients:
        roles = c.explicit_cache_roles
        for e in c.explicit_caches:
            if only is None or e["name"] in only:
                usd = gemini.gemini_cache_storage_cost(e["tokens"], e["lifetime_s"])
                out[e["name"]] = (roles.get(e["name"]), usd)
    return out


def _by_role(bill, *roles):
    return sum(usd for role, usd in bill.values() if role in roles)


def _gemini_run(sessions, run_id, workers=None):
    return pipeline.run(
        sessions,
        api_key="fake-key-for-test",
        data_dir=_data_dir(),
        run_id=run_id,
        workers=workers,
        log=lambda m: None,
    )


class StorageByStage(unittest.TestCase):
    """Prompt-cache storage in `run`: each resource priced once and charged to the stages that use
    it, and to no other."""

    def test_each_stage_carries_its_own_prompts(self):
        with FakeGemini().patched() as fake:
            result = _gemini_run(_sessions(SHOP, BILLING), "full")
        self.assertTrue(result.ok)
        c = fake.clients[0]
        bill = _bill(c.extract, c.merge)
        self.assertEqual(len(bill), 5)
        stages = result.metrics["stages"]
        reconcile = _by_role(bill, "reconcile")
        want = {
            "facts": _by_role(bill, "extract-facts") + reconcile / 2,
            "preferences": _by_role(bill, "extract-preferences") + reconcile / 2,
            "merge": _by_role(bill, "merge-synthesis", "merge-dedup"),
        }
        for stage, usd in want.items():
            self.assertAlmostEqual(stages[stage]["cost_usd"], usd, delta=1e-6, msg=stage)

    def test_a_stopped_run_keeps_unused_storage_off_the_stages_that_ran(self):
        with FakeGemini(fail_facts_on={BILLING}).patched() as fake:
            result = _gemini_run(_sessions(SHOP, BILLING), "stopped")
        self.assertFalse(result.ok)
        self.assertEqual(list(result.metrics["stages"]), ["facts"])
        c = fake.clients[0]
        bill = _bill(c.extract, c.merge)
        own = _by_role(bill, "extract-facts", "reconcile")
        unused = _by_role(bill, "extract-preferences", "merge-synthesis", "merge-dedup")
        self.assertGreater(unused, 0)
        self.assertAlmostEqual(result.metrics["stages"]["facts"]["cost_usd"], own, delta=1e-6)
        self.assertAlmostEqual(result.metrics["unused_prompt_cache_usd"], unused, delta=1e-6)
        self.assertAlmostEqual(result.metrics["cost_usd_total"], own + unused, delta=2e-6)
        self.assertAlmostEqual(result.metrics["spent_usd_total"], own + unused, delta=2e-6)

    def test_a_reused_client_prices_only_the_caches_of_its_own_run(self):
        with FakeGemini().patched() as fake:
            client = build_client("gemini", None, api_key="fake-key-for-test", explicit_cache=True)
            _run(_sessions(SHOP), client, "a")
            first = set(fake.created)
            second = _run(_sessions(BILLING), client, "b")
        own = set(fake.created) - first
        self.assertTrue(own)
        bill = _bill(client, only=own)
        self.assertAlmostEqual(second.metrics["cost_usd_total"], sum(u for _, u in bill.values()), delta=3e-6)

    def test_the_stage_files_sum_to_the_bill(self):
        # 60 s per call keeps every cache under the one-hour TTL, so lifetimes differ by stage.
        with FakeGemini(call_s=60.0).patched() as fake:
            result = _gemini_run(_sessions(SHOP, BILLING), "files")
        c = fake.clients[0]
        bill = _bill(c.extract, c.merge)
        arms = [_arm_metrics("files", a)["explicit_cache_storage_usd"] for a in ("facts", "preferences")]
        merged = result.metrics["stages"]["merge"]
        total = sum(arms) + merged["explicit_cache_storage_usd"]
        self.assertAlmostEqual(total, sum(u for _, u in bill.values()), delta=3e-6)
        self.assertEqual(
            sorted(e["name"] for e in merged["explicit_caches"]),
            sorted(n for n, (role, _) in bill.items() if role in ("merge-synthesis", "merge-dedup")),
        )
        self.assertEqual(merged["explicit_cache_tokens"], 2 * FakeGemini.TOKENS)
        listed = sum(
            gemini.gemini_cache_storage_cost(e["tokens"], e["lifetime_s"]) for e in merged["explicit_caches"]
        )
        self.assertAlmostEqual(listed, merged["explicit_cache_storage_usd"], delta=1e-6)


class _CliCase(unittest.TestCase):
    def _cli(self, *argv):
        out = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(cli.main(list(argv), out=out), 0)
        return out.getvalue().splitlines()

    SEL = ("--claude-home", FIXTURE_HOME, "--project", "/home/dev/src/shop")


class EveryCacheIsDeleted(_CliCase):
    def test_after_a_run_that_raises_mid_arm(self):
        with FakeGemini(boom_on_call=2).patched() as fake, self.assertRaises(Boom):
            _gemini_run(_sessions(SHOP, BILLING), "boom", workers=1)
        self.assertTrue(fake.created)
        self.assertEqual(sorted(fake.deleted), sorted(fake.created))

    def test_an_arm_file_written_before_a_raise_lists_only_its_own_caches(self):
        with FakeGemini(boom_on_call=2).patched() as fake, self.assertRaises(Boom):
            _gemini_run(_sessions(SHOP, BILLING), "boom", workers=1)
        roles = fake.clients[0].extract.explicit_cache_roles | fake.clients[0].merge.explicit_cache_roles
        listed = {roles[e["name"]] for e in _arm_metrics("boom", "facts")["explicit_caches"]}
        self.assertEqual(listed, {"extract-facts", "reconcile"})

    def test_after_extract(self):
        with FakeGemini().patched() as fake:
            self._cli("extract", "--arm", "facts", *self.SEL, "--data-dir", _data_dir(), "--run-id", "x")
        self.assertTrue(fake.created)
        self.assertEqual(sorted(fake.deleted), sorted(fake.created))

    def test_after_merge(self):
        with FakeGemini().patched() as fake:
            for arm in ("facts", "preferences"):
                self._cli("extract", "--arm", arm, *self.SEL, "--data-dir", _data_dir(), "--run-id", "x")
            del fake.created[:], fake.deleted[:]
            self._cli("merge", "--data-dir", _data_dir(), "--run-id", "x")
        self.assertTrue(fake.created)
        self.assertEqual(sorted(fake.deleted), sorted(fake.created))


class SubcommandStorage(_CliCase):
    """`extract` and `merge` print their stage's cost with its own prompt-cache storage."""

    def test_extract_and_merge_lines_include_storage(self):
        with FakeGemini().patched():
            lines = {}
            for arm in ("facts", "preferences"):
                sel = (*self.SEL, "--data-dir", _data_dir(), "--run-id", "x")
                lines[arm] = self._cli("extract", "--arm", arm, *sel)[0]
            lines["merge"] = self._cli("merge", "--data-dir", _data_dir(), "--run-id", "x")[0]
        for arm in ("facts", "preferences"):
            m = _arm_metrics("x", arm)
            self.assertGreater(m["explicit_cache_storage_usd"], 0)
            self.assertTrue(
                lines[arm].endswith(f", ${m['cost_usd_total'] + m['explicit_cache_storage_usd']:.2f}")
            )
        with open(os.path.join(_data_dir(), "runs", "x", "metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertGreater(m["explicit_cache_storage_usd"], 0)
        self.assertTrue(
            lines["merge"].endswith(f", ${m['cost_usd_total'] + m['explicit_cache_storage_usd']:.2f}")
        )


class MergeStage(unittest.TestCase):
    M = {
        "candidates_in": 7,
        "final_merged": 5,
        "cost_usd_total": 0.1234567,
        "cached_answers_usd": 0.0,
        "explicit_cache_storage_usd": 0.0456789,
    }

    def test_its_own_storage(self):
        st = pipeline.merge_stage(dict(self.M))
        self.assertEqual(st["cost_usd"], round(0.1234567 + 0.0456789, 6))

    def test_the_run_share_replaces_it(self):
        st = pipeline.merge_stage(dict(self.M), 0.5)
        self.assertEqual(st["cost_usd"], round(0.1234567 + 0.5, 6))
        self.assertEqual(st["explicit_cache_storage_usd"], 0.5)
