# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""One extraction arm: windowing, per-window extraction, reconcile, ordering, fatal stops and the
spend a run reports. No test calls a model: the clients are in-process stubs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unittest
from unittest import mock

from agent_memory_engine import cache as cache_mod
from agent_memory_engine import contract as mc
from agent_memory_engine import extract as rc
from agent_memory_engine import merge, transcript
from agent_memory_engine.providers import base as _b
from agent_memory_engine.providers import gemini as mcl
from agent_memory_engine.providers.base import ExtractResult
from agent_memory_engine.providers.gemini import GeminiClient


def _with_reason(fake_resolve):
    """A `resolve_with_reason` stand-in built from a fake that returns a Conversation or None."""

    def _resolve(row):
        conv = fake_resolve(row)
        return conv, (None if conv is not None else "no readable turns")

    return _resolve


def _turns(specs):
    """specs: list of (role, length) -> Turns with 1-based real indices and content of that length."""
    return [
        transcript.Turn(i=idx, role=role, content=(role[0].upper() * length))
        for idx, (role, length) in enumerate(specs, 1)
    ]


def _flatten(windows):
    return [t for w in windows for t in w]


class Windowing(unittest.TestCase):
    def test_no_turn_is_split_and_real_indices_preserved(self):
        turns = _turns([("user", 30), ("assistant", 30), ("assistant", 30), ("user", 30), ("assistant", 30)])
        windows = rc.build_windows(turns, budget=100)
        flat = _flatten(windows)
        self.assertEqual([t.i for t in flat], [1, 2, 3, 4, 5])
        self.assertEqual([t.content for t in flat], [t.content for t in turns])
        for w in windows:
            if len(w) > 1:
                self.assertLessEqual(sum(len(t.content) for t in w), 100)

    def test_packs_greedily_under_budget(self):
        windows = rc.build_windows(_turns([("assistant", 30)] * 7), budget=100)
        self.assertEqual([len(w) for w in windows], [3, 3, 1])

    def test_boundary_cases(self):
        # (turn specs, expected window turn-index lists). budget 100 -> "near" is 60.
        cases = [
            # two 40-char assistant turns fill to 80 >= near: the user turn opens a new window
            ([("assistant", 40), ("assistant", 40), ("user", 10), ("assistant", 40)], [[1, 2], [3, 4]]),
            # a user turn while the window is lightly filled does not force a break
            ([("assistant", 10), ("user", 10)], [[1, 2]]),
            # an over-budget turn becomes its own window; neighbours survive
            ([("assistant", 50), ("user", 250), ("assistant", 50)], [[1], [2], [3]]),
            # a window filled to exactly the near-budget mark: the user turn still opens a new one
            ([("assistant", 60), ("user", 10)], [[1], [2]]),
            # just under the mark: it stays in the same window
            ([("assistant", 59), ("user", 10)], [[1, 2]]),
            # no turns, no windows
            ([], []),
        ]
        for specs, want in cases:
            with self.subTest(specs=specs):
                got = [[t.i for t in w] for w in rc.build_windows(_turns(specs), budget=100)]
                self.assertEqual(got, want)

    def test_invalid_budget_is_refused(self):
        for bad in (0, -1, 1.5, True, None):
            with self.subTest(budget=bad), self.assertRaises(ValueError):
                rc.build_windows(_turns([("user", 5)]), budget=bad)

    def test_a_parsed_turn_always_fits_the_default_budget(self):
        # The per-turn cap is below the window budget, so the over-budget branch never fires on parsed input.
        longest = len(transcript._truncate_turn("x" * 50_000))
        self.assertLess(longest, rc.WINDOW_BUDGET)


class Prompts(unittest.TestCase):
    def test_every_prompt_loads_and_carries_its_contract(self):
        for arm in rc.ARMS:
            with self.subTest(arm=arm.name):
                ext = rc.load_prompt(arm.prompt_file)
                self.assertNotIn("BEGIN SYSTEM PROMPT", ext)
                for key in (
                    '"statement"',
                    '"evidence"',
                    '"disposition"',
                    '"generality"',
                    '"scope_level"',
                    '"scope_ref"',
                    "typed object",
                ):
                    self.assertIn(key, ext.lower() if key == "typed object" else ext)
        rec = rc.load_reconcile_prompt()
        for key in ('"members"', '"deleted"', "generality", '"scope_level"'):
            self.assertIn(key, rec)

    def test_arm_versions_are_pinned_and_distinct(self):
        # Each label is inside every cache key, so a change orphans the cached answers: pin it.
        versions = [a.prompt_version for a in rc.ARMS] + [rc.RECONCILE_PROMPT_VERSION]
        self.assertEqual(
            versions, ["facts-v1+fmt-v2", "preferences-v1+fmt-v2", "chunked-reconcile-v2+fmt-v2"]
        )
        self.assertEqual(merge.PROMPT_VERSION, "merge-v1+fmt-v3")
        self.assertEqual(len(set(versions)), len(versions))
        self.assertEqual([a.name for a in rc.ARMS], ["facts", "preferences"])

    def test_a_missing_marker_is_refused(self):
        with mock.patch("importlib.resources.files") as files:
            files.return_value.joinpath.return_value.read_text.return_value = "no markers here"
            with self.assertRaises(ValueError):
                rc.load_prompt("extract_facts.md")

    def test_unknown_arm_is_refused(self):
        with self.assertRaises(ValueError):
            rc.run([], tempfile.mkdtemp(), _StubClient(), arm="nope", cache_dir=tempfile.mkdtemp())


def _reconcile_keep_all(text: str) -> ExtractResult:
    """A reconcile answer that keeps every candidate as its own memory."""
    labels = re.findall(r"^\[(C\d+)\]", text, re.M)
    obj = {"memories": [{"members": [lab]} for lab in labels], "deleted": []}
    return ExtractResult(
        json.dumps(obj), 0.0, 1, "stub-model-1", ok=True, meta={"canonical_model": "stub-model-1"}
    )


class _StubClient:
    """A no-network ModelClient stand-in. Emits one gate-surviving memory per window, citing the
    window's user turn (a verbatim quote planted in the turn content). The per-conversation reverse
    delay makes concurrent completion order the opposite of selection order, so a 'write in
    completion order' regression would reorder the output."""

    model_id = "stub-model-1"
    cache_model_id = "stub-model-1"

    def __init__(self, delay_unit: float = 0.0):
        self.delay_unit = delay_unit

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        if "CANDIDATE MEMORIES" in conversation_text:
            return _reconcile_keep_all(conversation_text)
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        k = int(cid.rsplit("-", 1)[1])
        if self.delay_unit:
            time.sleep(self.delay_unit * (10 - k))
        obj = {
            "memories": [
                {
                    "statement": f"memory-{cid}",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": ["t"],
                    "disposition": "confident",
                    "evidence": [{"quote": f"quotable-phrase-{cid}", "turn": 1, "speaker": "user"}],
                }
            ]
        }
        return ExtractResult(
            raw_text=json.dumps(obj),
            cost_usd=0.0,
            duration_ms=1,
            model_id=self.model_id,
            ok=True,
            meta={"canonical_model": self.model_id},
        )


class _FailingClient(_StubClient):
    """Fails on one target conversation, by a model-level error (ok=False) or a raised exception."""

    def __init__(self, fail_cid: str, mode: str = "model", delay_unit: float = 0.0):
        super().__init__(delay_unit=delay_unit)
        self.fail_cid = fail_cid
        self.mode = mode

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        m = re.search(r"CONVID=(\S+)", conversation_text)
        if m and m.group(1) == self.fail_cid:
            if self.mode == "exception":
                raise RuntimeError(f"boom on {m.group(1)}")
            return ExtractResult(
                raw_text="", cost_usd=0.0, duration_ms=1, model_id=self.model_id, ok=False, error="model-fail"
            )
        return super().extract(system_prompt, conversation_text)


class _UsageClient(_StubClient):
    """Also reports a Gemini-shaped usageMetadata block, and counts its calls."""

    USAGE = {
        "promptTokenCount": 1_000,
        "candidatesTokenCount": 200,
        "thoughtsTokenCount": 4_000,
        "cachedContentTokenCount": 600,
    }
    http_retries = 0

    def __init__(self):
        super().__init__()
        self.calls = 0

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        self.calls += 1
        r = super().extract(system_prompt, conversation_text)
        r.meta["usage"] = dict(self.USAGE)
        r.meta["http_retries"] = self.http_retries
        return r


def _single_turn(cid: str):
    """One genuine person turn (human_prose) carrying the plantable quote."""
    return [
        transcript.Turn(
            i=1,
            role="user",
            authorship="human_prose",
            content=f"CONVID={cid} quotable-phrase-{cid} alpha beta",
        )
    ]


def _multi_turn(cid: str, n_turns: int = 3, size: int = 7000):
    """`n_turns` user turns each above the near-budget mark, so every turn opens its own window."""
    return [
        transcript.Turn(
            i=j + 1,
            role="user",
            authorship="human_prose",
            content=f"CONVID={cid} quotable-phrase " + ("x" * size),
        )
        for j in range(n_turns)
    ]


def _run_stub(
    client,
    workers: int,
    n_conv: int = 6,
    turns_fn=_single_turn,
    skip_ids: set | None = None,
    reconcile_client=None,
):
    """Drive extract.run over `n_conv` synthetic conversations (conv-0 .. conv-(n-1)) with
    transcript.resolve_with_reason patched to build them in memory: no session files, no network. Returns
    (metrics, ok, memory lines). `skip_ids` resolve to None, exercising the skipped-session path."""
    skip_ids = skip_ids or set()
    data_dir = tempfile.mkdtemp(prefix=f"extract_w{workers}_")
    out_dir = os.path.join(data_dir, "runs", f"run_w{workers}", "arms", "facts")
    rows = [
        {
            "conversation_id": f"conv-{k}",
            "native_id": f"n-{k}",
            "tool": "claude-code",
            "source": "local",
            "path": "unused",
        }
        for k in range(n_conv)
    ]

    def _fake_resolve(row):
        cid = row["conversation_id"]
        if cid in skip_ids:
            return None
        return transcript.Conversation(
            conversation_id=cid,
            tool="claude-code",
            source="local",
            date="2026-01-01",
            turns=turns_fn(cid),
            truncated=False,
        )

    with mock.patch.object(transcript, "resolve_with_reason", _with_reason(_fake_resolve)):
        metrics, ok = rc.run(
            rows,
            out_dir,
            client,
            arm="facts",
            cache_dir=os.path.join(data_dir, "cache", "extractions"),
            workers=workers,
            reconcile_client=reconcile_client,
            log=lambda m: None,
        )
    memp = os.path.join(out_dir, "memories.jsonl")
    lines = []
    if os.path.exists(memp):
        with open(memp, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return metrics, ok, lines


class ConversationParallelDeterminism(unittest.TestCase):
    """The whole point of the WINDOW->CONVERSATION parallelism refactor: output must be identical
    regardless of --workers. workers=1 (the byte-identical Claude default) and workers>1 must
    produce the same memories.jsonl in the same manifest order, and the same stable metrics."""

    N_CONV = 6

    def _run(self, workers: int, delay_unit: float):
        metrics, ok, lines = _run_stub(_StubClient(delay_unit=delay_unit), workers, self.N_CONV)
        self.assertTrue(ok)
        return metrics, lines

    def test_workers_1_vs_3_identical_ordered_output(self):
        m1, lines1 = self._run(workers=1, delay_unit=0.0)
        m3, lines3 = self._run(workers=3, delay_unit=0.01)

        # Output is byte-identical and in MANIFEST order (conv-0 .. conv-N), never completion order.
        expected_order = [f"memory-conv-{k}" for k in range(self.N_CONV)]
        self.assertEqual([json.loads(x)["statement"] for x in lines1], expected_order)
        self.assertEqual(lines1, lines3)

        # Stable metrics match too (drop only the wall-clock timing that legitimately varies).
        def _stable(m):
            return {k: v for k, v in m.items() if k not in ("wall_clock_s",)}

        self.assertEqual(m1["conversations_run"], self.N_CONV)
        self.assertEqual(m1["memories_total"], self.N_CONV)
        self.assertEqual(_stable(m1), _stable(m3))


class _MultiWindowClient(_StubClient):
    """Emits one gate-surviving memory per WINDOW (citing that window's first turn), and can fail
    at a target window index via a model-level fatal — to exercise the multi-window partial fold on
    the fatal path (pre-fatal windows' duration/drops counted, windows_total still counts them all,
    candidates NOT folded on an extract-stage fatal)."""

    def __init__(self, fail_cid: str | None = None, fail_window: int | None = None):
        super().__init__()
        self.fail_cid = fail_cid
        self.fail_window = fail_window

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        if "CANDIDATE MEMORIES" in conversation_text:
            return _reconcile_keep_all(conversation_text)
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        meta = json.loads(conversation_text.splitlines()[1])
        widx = meta["window"] - 1
        turn = meta["window_turns"][0]
        if cid == self.fail_cid and widx == self.fail_window:
            return ExtractResult("", 0.0, 1, self.model_id, ok=False, error="model-fail")
        obj = {
            "memories": [
                {
                    "statement": f"memory-{cid}-w{widx}",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": ["t"],
                    "disposition": "confident",
                    "evidence": [{"quote": "quotable-phrase", "turn": turn, "speaker": "user"}],
                }
            ]
        }
        return ExtractResult(
            json.dumps(obj), 0.0, 1, self.model_id, ok=True, meta={"canonical_model": self.model_id}
        )


class _ReconcileFatalClient(_StubClient):
    """Emits TWO memories on every extract (so conv_pre>=2 triggers the reconcile stage), then fails
    every reconcile call — to exercise the reconcile-stage fold (candidates_pre_reconcile IS folded
    for the failing conversation)."""

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        if "CANDIDATE MEMORIES" in conversation_text:  # this is the reconcile call
            return ExtractResult("", 0.0, 1, self.model_id, ok=False, error="recon-fail")
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        obj = {
            "memories": [
                {
                    "statement": f"memory-{cid}-a",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": ["t"],
                    "disposition": "confident",
                    "evidence": [{"quote": "alpha", "turn": 1, "speaker": "user"}],
                },
                {
                    "statement": f"memory-{cid}-b",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": ["t"],
                    "disposition": "confident",
                    "evidence": [{"quote": "beta", "turn": 1, "speaker": "user"}],
                },
            ]
        }
        return ExtractResult(
            json.dumps(obj), 0.0, 1, self.model_id, ok=True, meta={"canonical_model": self.model_id}
        )


class ConversationParallelFatalStop(unittest.TestCase):
    """A fatal in one conversation must STOP the run cleanly at the right manifest position:
    conversations before it are written, it and everything after are dropped, ok=False, and the
    fatal is surfaced in metrics — identical whether the batch ran 1 or 3 wide. Covers both a
    model-level fatal (ok=False) and an unexpected raised exception (must be handled, never
    propagate out of run())."""

    def _assert_clean_stop(self, mode: str, expected_stage: str):
        for workers in (1, 3):
            client = _FailingClient(fail_cid="conv-2", mode=mode)
            metrics, ok, lines = _run_stub(client, workers, n_conv=6)
            ctx = f"mode={mode} workers={workers}"
            self.assertFalse(ok, ctx)
            # conv-0 and conv-1 (before the fatal) are written, in order; conv-2+ are dropped.
            self.assertEqual(
                [json.loads(x)["statement"] for x in lines], ["memory-conv-0", "memory-conv-1"], ctx
            )
            self.assertEqual(metrics["conversations_run"], 2, ctx)
            fe = metrics.get("fatal_error") or {}
            self.assertEqual(fe.get("conversation_id"), "conv-2", ctx)
            self.assertEqual(fe.get("stage"), expected_stage, ctx)

    def test_model_fatal_stops_cleanly(self):
        self._assert_clean_stop("model", expected_stage="extract")

    def test_unexpected_exception_is_handled_as_fatal_not_raised(self):
        # If _safe_process did not catch, the raw RuntimeError would unwind out of run() (dropping
        # conv-0/conv-1 under workers=3) instead of a clean STOP — this asserts the wrapper holds.
        self._assert_clean_stop("exception", expected_stage="exception")

    def test_multiwindow_extract_fatal_folds_windows_not_candidates(self):
        # conv-0 is 3 windows; extraction succeeds on windows 0,1 then hits a model fatal on window
        # 2. The fold must count ALL 3 windows (windows_total) and must NOT fold
        # candidates_pre_reconcile (extract-stage fatal). conv-0 is first, so nothing is written.
        for workers in (1, 3):
            ctx = f"workers={workers}"
            client = _MultiWindowClient(fail_cid="conv-0", fail_window=2)
            metrics, ok, lines = _run_stub(
                client, workers, n_conv=3, turns_fn=lambda cid: _multi_turn(cid, 3)
            )
            self.assertFalse(ok, ctx)
            self.assertEqual(lines, [], ctx)
            self.assertEqual(metrics["conversations_run"], 0, ctx)
            self.assertEqual(metrics["windows_total"], 3, ctx)  # full count folded
            self.assertEqual(metrics["candidates_pre_reconcile"], 0, ctx)  # extract fatal: NOT folded
            fe = metrics.get("fatal_error") or {}
            self.assertEqual(fe.get("stage"), "extract", ctx)
            self.assertEqual(fe.get("window"), 2, ctx)
            # Sequentially, only the failing conversation ran: its two pre-fatal windows and the
            # failed call itself. Under >1 worker the siblings ran too, and their real calls are
            # folded as well — see the spend tests below.
            if workers == 1:
                self.assertEqual(metrics["model_duration_ms_total"], 3, ctx)
            self.assertGreaterEqual(metrics["model_duration_ms_total"], 3, ctx)

    def _paid_call_client(self):
        """A multi-window client whose successful calls each cost real money, so a stopped run's
        metrics can be checked against what was actually billed."""

        class _Paid(_MultiWindowClient):
            USAGE = {"promptTokenCount": 1_000, "candidatesTokenCount": 50, "thoughtsTokenCount": 500}

            def extract(self, system_prompt, conversation_text):
                r = super().extract(system_prompt, conversation_text)
                if r.ok:
                    r.meta["usage"] = dict(self.USAGE)
                    r.meta["http_retries"] = 1
                return r

        return _Paid(fail_cid="conv-1", fail_window=2)

    def test_a_fatal_stop_reports_the_calls_it_paid_for(self):
        """A stop mid-conversation still accounts for every call it paid for."""
        client = self._paid_call_client()
        metrics, ok, lines = _run_stub(client, workers=1, n_conv=3, turns_fn=lambda cid: _multi_turn(cid, 3))
        self.assertFalse(ok)
        # conv-0 completed (3 window calls + 1 reconcile), conv-1 paid for 2 windows then hit the
        # fatal on window 2
        self.assertEqual([json.loads(x)["origin"]["conversation_id"] for x in lines], ["conv-0"] * 3)
        self.assertEqual(metrics["conversations_run"], 1)
        # the failed call is a call too; this stub bills nothing for it
        self.assertEqual(metrics["llm_calls"], 7, "4 completed + 2 paid before the fatal + the fatal call")
        per_call = mcl.gemini_call_cost(client.USAGE)
        self.assertGreater(per_call, 0.0)
        self.assertAlmostEqual(metrics["cost_usd_total"], 6 * per_call, places=9)
        self.assertEqual(metrics["prompt_tokens_total"], 6 * 1_000)
        self.assertEqual(metrics["thinking_tokens_total"], 6 * 500)
        self.assertEqual(metrics["http_retries"], 6)
        # and the abandoned work is named, not just summed
        self.assertEqual(
            [
                (p["conversation_id"], p["status"], p["windows_done"], p["calls"])
                for p in metrics["partial_conversations"]
            ],
            [("conv-1", "fatal", 3, 3)],  # two paid windows and the failed call
        )
        self.assertAlmostEqual(
            metrics["partial_conversations"][0]["cost_usd"], round(2 * per_call, 6), places=6
        )
        # per_conversation still describes only what was WRITTEN
        self.assertEqual([p["conversation_id"] for p in metrics["per_conversation"]], ["conv-0"])

    def test_conversations_abandoned_in_flight_are_accounted_too(self):
        """Under workers the stop also abandons siblings that had already run. Their calls were
        billed as surely as the fatal one's, so they must reach the record."""
        client = self._paid_call_client()
        metrics, ok, _ = _run_stub(client, workers=3, n_conv=4, turns_fn=lambda cid: _multi_turn(cid, 3))
        self.assertFalse(ok)
        per_call = mcl.gemini_call_cost(client.USAGE)
        calls = metrics["llm_calls"]
        # conv-0 completes (3) + conv-1 pays 2 before failing, then the failed call = 6 at minimum;
        # whichever siblings were in flight add theirs, and every one of them is accounted for. The
        # one failed call bills nothing in this stub.
        self.assertGreaterEqual(calls, 6)
        self.assertAlmostEqual(metrics["cost_usd_total"], (calls - 1) * per_call, places=9)
        self.assertEqual(metrics["prompt_tokens_total"], (calls - 1) * 1_000)
        statuses = {p["status"] for p in metrics["partial_conversations"]}
        self.assertIn("fatal", statuses)
        # the totals equal the completed rows plus the partial rows — nothing is unattributed
        accounted = sum(p["llm_calls"] for p in metrics["partial_conversations"]) + sum(
            p["llm_calls"] for p in metrics["per_conversation"]
        )
        self.assertEqual(accounted, calls)

    def test_reconcile_fatal_folds_candidates_pre(self):
        # conv-0 extracts 2 candidates (so reconcile runs) then the reconcile call fails. The fold
        # MUST add candidates_pre_reconcile for the failing conversation (the reconcile-stage branch
        # of _fold_fatal), which an extract-stage fatal must not.
        for workers in (1, 3):
            ctx = f"workers={workers}"
            metrics, ok, lines = _run_stub(_ReconcileFatalClient(), workers, n_conv=3)
            self.assertFalse(ok, ctx)
            self.assertEqual(lines, [], ctx)
            self.assertEqual(metrics["conversations_run"], 0, ctx)
            self.assertEqual(metrics["windows_total"], 1, ctx)
            self.assertEqual(metrics["candidates_pre_reconcile"], 2, ctx)  # reconcile fatal FOLDS it
            fe = metrics.get("fatal_error") or {}
            self.assertEqual(fe.get("stage"), "reconcile", ctx)


class _RetryingStubClient(_StubClient):
    """Cold-cache retry simulation. A conversation's extract 'retries' a VARYING number of times
    (raising a retryable error then succeeding), each retry waiting a small REQUEST-LOCAL
    deterministic backoff — the same helper the real Gemini client uses, seeded per conversation so
    the wait never depends on thread interleaving. Because the retry count differs per conversation,
    completion order diverges from submission order by a per-worker-count amount; the runner must
    still emit memories in MANIFEST order at any width. Mirrors the client-level retry loop INSIDE
    extract (the runner does not retry), so the runner sees one successful result per conversation."""

    def __init__(self, base_delay: float = 0.003):
        super().__init__(delay_unit=0.0)
        self.base_delay = base_delay

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        k = int(cid.rsplit("-", 1)[1])
        fail_times = (k % 3) + 1  # 1..3 retryable failures before success (varies)
        for attempt in range(1, fail_times + 1):
            # request-local deterministic backoff (fix #3), scaled tiny so the test stays fast
            delay = GeminiClient._backoff_delay(attempt, None, cid) * self.base_delay
            time.sleep(delay)
        return super().extract(system_prompt, conversation_text)


class StreamingSchedulerDeterminism(unittest.TestCase):
    """The rolling worker pool must fold/write STRICTLY by manifest (submission) index, never by
    completion order, so its output is byte-identical to workers=1 even when refill runs past a
    head-of-line-blocking early conversation and when manifest rows are skipped. These force
    scheduling shapes a barrier'd batch runner would never produce."""

    def _stable(self, m):
        return {k: v for k, v in m.items() if k not in ("wall_clock_s",)}

    def test_cold_cache_forced_retries_deterministic_workers_1_vs_6(self):
        # Cold cache (each _run_stub gets a fresh data dir) + forced retries: every conversation
        # retries a varying number of times before succeeding, so completion order diverges from
        # submission order — and differently at workers=1 vs workers=6. The emitted memories and
        # their order must still be byte-identical, and the stable metrics must match: determinism
        # does not depend on retry timing.
        N = 8
        m1, ok1, lines1 = _run_stub(_RetryingStubClient(), 1, n_conv=N)
        m6, ok6, lines6 = _run_stub(_RetryingStubClient(), 6, n_conv=N)
        self.assertTrue(ok1 and ok6)
        self.assertEqual([json.loads(x)["statement"] for x in lines1], [f"memory-conv-{k}" for k in range(N)])
        self.assertEqual(lines1, lines6)  # byte-identical, manifest order
        self.assertEqual(self._stable(m1), self._stable(m6))

    def test_refill_past_head_of_line_block_matches_sequential(self):
        # A very slow conv-0 (the _StubClient delay is delay_unit*(10-k), so k=0 is slowest) blocks
        # next_merge while workers=3 keeps refilling convs 3..7 — exactly the "submit past the
        # barrier" the batch runner could not do. Output + stable metrics must equal workers=1.
        for n_conv, workers in ((8, 3), (8, 6)):
            with self.subTest(n_conv=n_conv, workers=workers):
                m1, ok1, lines1 = _run_stub(_StubClient(delay_unit=0.0), 1, n_conv=n_conv)
                mp, okp, linesp = _run_stub(_StubClient(delay_unit=0.005), workers, n_conv=n_conv)
                self.assertTrue(ok1 and okp)
                self.assertEqual(
                    [json.loads(x)["statement"] for x in lines1], [f"memory-conv-{k}" for k in range(n_conv)]
                )
                self.assertEqual(lines1, linesp)
                self.assertEqual(self._stable(m1), self._stable(mp))

    def test_fatal_behind_a_slower_earlier_conversation_drains_first(self):
        # conv-2 fails FAST (the failing branch never sleeps) while the earlier conv-0/conv-1 are
        # still running (they sleep delay_unit*(10-k)). The fatal must sit buffered until it reaches
        # next_merge: conv-0 and conv-1 are written in order first, then the run STOPs at conv-2.
        for workers in (3, 6):
            with self.subTest(workers=workers):
                client = _FailingClient(fail_cid="conv-2", mode="model", delay_unit=0.01)
                metrics, ok, lines = _run_stub(client, workers, n_conv=6)
                ctx = f"workers={workers}"
                self.assertFalse(ok, ctx)
                self.assertEqual(
                    [json.loads(x)["statement"] for x in lines], ["memory-conv-0", "memory-conv-1"], ctx
                )
                self.assertEqual(metrics["conversations_run"], 2, ctx)
                fe = metrics.get("fatal_error") or {}
                self.assertEqual(fe.get("conversation_id"), "conv-2", ctx)
                self.assertEqual(fe.get("stage"), "extract", ctx)

    def test_skipped_rows_interleaved_identical_workers_1_vs_3(self):
        # Some manifest rows resolve to None (transcript.resolve_with_reason -> None). On the success path
        # conversations_skipped IS worker-invariant, so skip counting AND output must match at
        # workers 1 vs 3, with the surviving conversations still in manifest order.
        skip = {"conv-1", "conv-3"}
        m1, ok1, lines1 = _run_stub(_StubClient(delay_unit=0.0), 1, n_conv=6, skip_ids=skip)
        m3, ok3, lines3 = _run_stub(_StubClient(delay_unit=0.005), 3, n_conv=6, skip_ids=skip)
        self.assertTrue(ok1 and ok3)
        self.assertEqual(
            [json.loads(x)["statement"] for x in lines1],
            ["memory-conv-0", "memory-conv-2", "memory-conv-4", "memory-conv-5"],
        )
        self.assertEqual(lines1, lines3)
        self.assertEqual(m1["conversations_skipped"], 2)
        self.assertEqual(m1["conversations_skipped"], m3["conversations_skipped"])
        self.assertEqual(self._stable(m1), self._stable(m3))


class _TemporalStubClient(_StubClient):
    """Emits one gate-surviving memory per window carrying the OPTIONAL temporal rich-meta fields,
    to prove they survive window-extraction -> (no-)reconcile -> memories.jsonl end to end."""

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        obj = {
            "memories": [
                {
                    "statement": f"memory-{cid}",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": ["t"],
                    "disposition": "confident",
                    "valid_as_of": "2026-07",
                    "temporal_kind": "dated",
                    "volatility": "high",
                    "evidence": [{"quote": f"quotable-phrase-{cid}", "turn": 1, "speaker": "user"}],
                }
            ]
        }
        return ExtractResult(
            raw_text=json.dumps(obj),
            cost_usd=0.0,
            duration_ms=1,
            model_id=self.model_id,
            ok=True,
            meta={"canonical_model": self.model_id},
        )


class TemporalFieldsEndToEnd(unittest.TestCase):
    """The three OPTIONAL temporal fields must reach memories.jsonl. Covers the pass-through of a
    single candidate and the reconcile-assembly carry (by recency)."""

    def test_noreconcile_carries_temporal_fields_to_output(self):
        _, ok, lines = _run_stub(_TemporalStubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertTrue(lines)
        for ln in lines:
            m = json.loads(ln)
            self.assertEqual(m["valid_as_of"], "2026-07")
            self.assertEqual(m["temporal_kind"], "dated")
            self.assertEqual(m["volatility"], "high")

    def test_absent_temporal_fields_default_null_in_output(self):
        # The base stub never emits the temporal fields -> they must be present as null, not missing.
        _, ok, lines = _run_stub(_StubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        for ln in lines:
            m = json.loads(ln)
            self.assertIsNone(m["valid_as_of"])
            self.assertIsNone(m["temporal_kind"])
            self.assertIsNone(m["volatility"])

    def test_reconcile_assembly_carries_latest_member_temporal_fields(self):
        # Two candidates merged into one reconciled memory: the temporal fields follow recency
        # (the latest member, C2), mirroring how scope/type/review fall back to the latest member.
        conv = transcript.Conversation(
            conversation_id="conv-x",
            tool="claude-code",
            source="local",
            date="2026-01-01",
            turns=[transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            truncated=False,
        )

        def _cand(label, wi, turn, quote, **temporal):
            c = {
                "label": label,
                "window_index": wi,
                "statement": "merged-stmt",
                "type": "fact",
                "scope_level": "personal",
                "scope_ref": None,
                "generality": "instance-specific",
                "tags": ["t"],
                "decision_outcome": None,
                "disposition": "confident",
                "intention_meta": None,
                "review": {"status": "active"},
                "valid_as_of": None,
                "temporal_kind": None,
                "volatility": None,
                "evidence": [{"quote": quote, "turn": turn, "speaker": "user", "conversation_id": "conv-x"}],
            }
            c.update(temporal)
            return c

        candidates = [
            _cand("C1", 0, 1, "alpha", temporal_kind="timeless", volatility="low"),
            _cand("C2", 1, 2, "beta", valid_as_of="2026-09", temporal_kind="dated", volatility="high"),
        ]
        recon_obj = {
            "memories": [
                {
                    "members": ["C1", "C2"],
                    "type": "fact",
                    "statement": "merged-stmt",
                    "generality": "instance-specific",
                }
            ],
            "deleted": [],
        }
        finals, stats = rc._assemble_reconciled(conv, candidates, recon_obj, "facts")
        self.assertEqual(len(finals), 1, stats)
        m = finals[0]
        self.assertEqual(m["valid_as_of"], "2026-09")  # latest member (C2) wins
        self.assertEqual(m["temporal_kind"], "dated")
        self.assertEqual(m["volatility"], "high")

    def test_candidates_the_reconcile_model_leaves_out_are_counted_deleted(self):
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            False,
        )
        base = {
            "type": "fact",
            "statement": "s",
            "scope_level": "personal",
            "scope_ref": None,
            "generality": "instance-specific",
            "tags": ["t"],
            "decision_outcome": None,
            "disposition": "confident",
            "intention_meta": None,
            "review": {"status": "active"},
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
        }
        cands = [
            {
                **base,
                "label": "C1",
                "window_index": 0,
                "evidence": [{"quote": "alpha", "turn": 1, "speaker": "user", "conversation_id": "conv-x"}],
            },
            {
                **base,
                "label": "C2",
                "window_index": 1,
                "evidence": [{"quote": "beta", "turn": 2, "speaker": "user", "conversation_id": "conv-x"}],
            },
        ]
        recon = {"memories": [{"members": ["C1"], "type": "fact", "statement": "s"}], "deleted": []}
        _, stats = rc._assemble_reconciled(conv, cands, recon, "facts")
        self.assertEqual((stats["n_deleted"], stats["deleted_labels"]), (1, ["C2"]))

    def test_reconcile_review_and_disposition_note(self):
        # The model's review is ignored; a borderline memory takes the note of its first borderline
        # member, and a confident memory carries no note.
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(i, "user", q) for i, q in ((1, "alpha"), (2, "beta"), (3, "gamma"))],
            False,
        )
        base = {
            "statement": "s",
            "scope_level": "personal",
            "scope_ref": None,
            "generality": "instance-specific",
            "tags": ["t"],
            "decision_outcome": None,
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
            "review": {"status": "active"},
        }

        def cand(label, wi, turn, quote, mtype, disposition, note, meta=None):
            return {
                **base,
                "label": label,
                "window_index": wi,
                "type": mtype,
                "disposition": disposition,
                "disposition_note": note,
                "intention_meta": meta,
                "evidence": [{"quote": quote, "turn": turn, "speaker": "user", "conversation_id": "conv-x"}],
            }

        cands = [
            cand(
                "C1",
                0,
                1,
                "alpha",
                "intention",
                "confident",
                "a confident aside",
                {"what": "w", "done_check": "d"},
            ),
            cand("C2", 1, 2, "beta", "fact", "borderline", "seen once"),
            cand("C3", 2, 3, "gamma", "fact", "confident", "another aside"),
        ]
        recon = {
            "memories": [
                {"members": ["C1", "C2"], "type": "fact", "statement": "s1", "review": {"status": "active"}},
                {"members": ["C3"], "type": "fact", "statement": "s2"},
            ],
            "deleted": [],
        }
        (grouped, single), _ = rc._assemble_reconciled(conv, cands, recon, "facts")
        self.assertEqual(grouped["review"], {"status": "hold-for-review"})
        self.assertEqual((grouped["disposition"], grouped["disposition_note"]), ("borderline", "seen once"))
        self.assertEqual((single["disposition"], single["disposition_note"]), ("confident", None))

    def test_a_confident_memory_carries_no_note_on_the_no_reconcile_path(self):
        conv = transcript.Conversation(
            "conv-x", "claude-code", "local", "2026-01-01", [transcript.Turn(1, "user", "alpha")], False
        )
        cand = {
            "label": "C1",
            "window_index": 0,
            "statement": "s",
            "type": "fact",
            "scope_level": "personal",
            "scope_ref": None,
            "generality": "instance-specific",
            "tags": ["t"],
            "decision_outcome": None,
            "intention_meta": None,
            "review": {"status": "active"},
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
            "evidence": [{"quote": "alpha", "turn": 1, "speaker": "user", "conversation_id": "conv-x"}],
        }
        for disposition, note, kept in (
            ("confident", "an aside", None),
            ("borderline", "seen once", "seen once"),
        ):
            with self.subTest(disposition=disposition):
                c = {**cand, "disposition": disposition, "disposition_note": note}
                (final,) = rc._candidates_to_finals_noreconcile(conv, [c], "facts")
                self.assertEqual(final["disposition_note"], kept)

    def test_noreconcile_intention_fold_clears_foreign_fields(self):
        # When a fold makes the memory an intention, a secondary_type equal to it and a decision's
        # outcome go: the format never pairs them with an intention.
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            False,
        )
        meta = {"what": "a", "done_check": "ca"}
        common = {
            "statement": "Add retries",
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "tags": ["a"],
            "disposition": "confident",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
        }
        firsts = [
            {"type": "fact", "secondary_type": "intention", "decision_outcome": None, "intention_meta": None},
            {
                "type": "decision",
                "secondary_type": None,
                "decision_outcome": "chose X",
                "intention_meta": None,
            },
        ]
        second = {
            "type": "intention",
            "secondary_type": None,
            "decision_outcome": None,
            "intention_meta": meta,
        }

        def cand(k, fields, quote):
            return {
                **common,
                **fields,
                "label": f"C{k + 1}",
                "window_index": k,
                "evidence": [{"quote": quote, "turn": k + 1, "speaker": "user", "conversation_id": "conv-x"}],
            }

        for first in firsts:
            cands = [cand(0, first, "alpha"), cand(1, second, "beta")]
            with self.subTest(path="no-reconcile", first=first["type"]):
                (final,) = rc._candidates_to_finals_noreconcile(conv, cands, "facts")
                self.assertEqual(
                    (final["type"], final["secondary_type"], final["decision_outcome"]),
                    ("intention", None, None),
                )
            with self.subTest(path="reconcile", first=first["type"]):
                recon = {
                    "memories": [{"members": ["C1", "C2"], "type": first["type"], "statement": "s"}],
                    "deleted": [],
                }
                (final,), _ = rc._assemble_reconciled(conv, cands, recon, "facts")
                self.assertEqual(
                    (final["type"], final["secondary_type"], final["decision_outcome"]),
                    ("intention", None, None),
                )
            with self.subTest(path="gate", first=first["type"]):
                obj = {
                    "memories": [
                        {
                            **common,
                            **{k: v for k, v in first.items() if v is not None},
                            "evidence": [{"quote": "alpha", "turn": 1, "speaker": "user"}],
                        },
                        {
                            **common,
                            "type": "intention",
                            "intention_meta": meta,
                            "evidence": [{"quote": "beta", "turn": 2, "speaker": "user"}],
                        },
                    ]
                }
                (gm,) = mc.validate_and_gate(obj, conv, "m").memories
                self.assertEqual((gm.type, gm.secondary_type, gm.decision_outcome), ("intention", None, None))

    def test_which_restatements_become_an_intention(self):
        # Only an intention member makes the fold an intention; a fact then a preference stays a fact.
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(i, "user", q) for i, q in ((1, "alpha"), (2, "beta"), (3, "gamma"))],
            False,
        )
        common = {
            "statement": "Add retries",
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "tags": ["a"],
            "disposition": "confident",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
            "decision_outcome": None,
            "secondary_type": None,
        }
        meta = {"what": "w", "done_check": "d"}
        for types, want in (
            (("fact", "preference"), "fact"),
            (("intention", "preference"), "intention"),
            (("fact", "intention", "preference"), "intention"),
        ):
            quotes = ("alpha", "beta", "gamma")[: len(types)]
            with self.subTest(path="no-reconcile", types=types):
                cands = [
                    {
                        **common,
                        "type": t,
                        "intention_meta": meta if t == "intention" else None,
                        "label": f"C{k + 1}",
                        "window_index": k,
                        "evidence": [
                            {"quote": q, "turn": k + 1, "speaker": "user", "conversation_id": "conv-x"}
                        ],
                    }
                    for k, (t, q) in enumerate(zip(types, quotes, strict=True))
                ]
                (final,) = rc._candidates_to_finals_noreconcile(conv, cands, "facts")
                self.assertEqual(final["type"], want)
            with self.subTest(path="gate", types=types):
                obj = {
                    "memories": [
                        {
                            **common,
                            "type": t,
                            "intention_meta": meta if t == "intention" else None,
                            "evidence": [{"quote": q, "turn": k + 1, "speaker": "user"}],
                        }
                        for k, (t, q) in enumerate(zip(types, quotes, strict=True))
                    ]
                }
                (gm,) = mc.validate_and_gate(obj, conv, "m").memories
                self.assertEqual(gm.type, want)

    def test_two_intentions_keep_one_meta_on_every_path(self):
        # Two intentions fold: the latest one that has meta wins, and one without meta never erases
        # the other's.
        A = {"what": "a", "done_check": "ca"}
        B = {"what": "b", "done_check": "cb"}
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            False,
        )
        for first, second, want in ((A, B, B), (None, B, B), (A, None, A)):
            metas = (first, second)
            base = {
                "statement": "Revisit tz handling",
                "type": "intention",
                "scope_level": "repo",
                "scope_ref": "acme/shop",
                "tags": ["a"],
                "disposition": "confident",
            }
            with self.subTest(path="gate", metas=metas):
                obj = {
                    "memories": [
                        {
                            **base,
                            "intention_meta": m,
                            "evidence": [{"quote": q, "turn": t, "speaker": "user"}],
                        }
                        for m, (t, q) in zip(metas, ((1, "alpha"), (2, "beta")), strict=True)
                    ]
                }
                (gm,) = mc.validate_and_gate(obj, conv, "m").memories
                self.assertEqual(gm.intention_meta, want)
            cands = [
                {
                    **base,
                    "label": f"C{k + 1}",
                    "window_index": k,
                    "generality": "instance-specific",
                    "decision_outcome": None,
                    "intention_meta": m,
                    "review": {"status": "hold-for-review"},
                    "valid_as_of": None,
                    "temporal_kind": None,
                    "volatility": None,
                    "evidence": [{"quote": q, "turn": k + 1, "speaker": "user", "conversation_id": "conv-x"}],
                }
                for k, (m, q) in enumerate(zip(metas, ("alpha", "beta"), strict=True))
            ]
            with self.subTest(path="no-reconcile", metas=metas):
                (final,) = rc._candidates_to_finals_noreconcile(conv, cands, "facts")
                self.assertEqual(final["intention_meta"], want)
            with self.subTest(path="reconcile", metas=metas):
                recon = {
                    "memories": [{"members": ["C1", "C2"], "type": "intention", "statement": "s"}],
                    "deleted": [],
                }
                (final,), _ = rc._assemble_reconciled(conv, cands, recon, "facts")
                self.assertEqual(final["intention_meta"], want)
            with self.subTest(path="merge", metas=metas):
                mems = [
                    {
                        "_arm": "facts",
                        "statement": f"revisit tz {k}",
                        "type": "intention",
                        "scope": "personal",
                        "tags": ["t"],
                        "memory_id": f"m{k}",
                        "when": f"2026-0{k + 1}-01",
                        "origin": {"conversation_id": f"cc:{k}", "tool": "claude-code"},
                        "evidence": [{"quote": "q", "turn": 1, "speaker": "user"}],
                        "review": {"status": "hold-for-review"},
                        "intention_meta": m,
                    }
                    for k, m in enumerate(metas)
                ]
                for order in ([0, 1], [1, 0]):
                    synth = merge._passthrough(mems, 0)
                    synth["_member_idxs"] = order
                    self.assertEqual(merge.build_final(synth, mems)["intention_meta"], want)

    def test_an_exact_restatement_fold_keeps_an_intention(self):
        # The same statement as a fact and as an intention folds to one memory, and that memory is
        # the intention, held for review with its meta, in either order and on both fold paths.
        meta = {"what": "add retries", "done_check": "a retry test passes"}
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(1, "user", "add retries to the billing job")],
            False,
        )
        ev = [{"quote": "add retries to the billing job", "turn": 1, "speaker": "user"}]
        fact = {
            "statement": "Add retries to the billing job",
            "type": "fact",
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "tags": ["a"],
            "disposition": "confident",
            "evidence": ev,
        }
        intention = {**fact, "type": "intention", "intention_meta": meta}
        for order in ((fact, intention), (intention, fact)):
            names = [m["type"] for m in order]
            with self.subTest(path="gate", order=names):
                rep = mc.validate_and_gate({"memories": [dict(m) for m in order]}, conv, "m")
                (gm,) = rep.memories
                self.assertEqual(
                    (gm.type, gm.review, gm.intention_meta),
                    ("intention", {"status": "hold-for-review"}, meta),
                )
            with self.subTest(path="no-reconcile", order=names):
                cands = []
                for wi, m in enumerate(order):
                    cands.append(
                        {
                            **m,
                            "label": f"C{wi + 1}",
                            "window_index": wi,
                            "generality": "instance-specific",
                            "scope_ref": "acme/shop",
                            "decision_outcome": None,
                            "intention_meta": m.get("intention_meta"),
                            "review": {"status": "active"},
                            "evidence": [{**ev[0], "conversation_id": "conv-x"}],
                        }
                    )
                (final,) = rc._candidates_to_finals_noreconcile(conv, cands, "facts")
                self.assertEqual(
                    (final["type"], final["review"], final["intention_meta"]),
                    ("intention", {"status": "hold-for-review"}, meta),
                )

    def test_a_reconciled_group_with_an_intention_stays_an_intention(self):
        # The reconcile model may name the group a fact; an intention member keeps it an intention,
        # held for review with its meta.
        conv = transcript.Conversation(
            "conv-x",
            "claude-code",
            "local",
            "2026-01-01",
            [transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            False,
        )
        base = {
            "statement": "s",
            "scope_level": "personal",
            "scope_ref": None,
            "generality": "instance-specific",
            "tags": ["t"],
            "decision_outcome": None,
            "disposition": "confident",
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
        }
        cands = [
            {
                **base,
                "label": "C1",
                "window_index": 0,
                "type": "intention",
                "review": {"status": "hold-for-review"},
                "intention_meta": {"what": "revisit tz", "done_check": "tz test passes"},
                "evidence": [{"quote": "alpha", "turn": 1, "speaker": "user", "conversation_id": "conv-x"}],
            },
            {
                **base,
                "label": "C2",
                "window_index": 1,
                "type": "fact",
                "review": {"status": "active"},
                "intention_meta": None,
                "evidence": [{"quote": "beta", "turn": 2, "speaker": "user", "conversation_id": "conv-x"}],
            },
        ]
        recon = {
            "memories": [{"members": ["C2", "C1"], "type": "fact", "statement": "fix tz"}],
            "deleted": [],
        }
        (m,), _ = rc._assemble_reconciled(conv, cands, recon, "facts")
        self.assertEqual((m["type"], m["review"]), ("intention", {"status": "hold-for-review"}))
        self.assertEqual(m["intention_meta"]["what"], "revisit tz")


class _DescriptionStubClient(_StubClient):
    """Emits one gate-surviving memory per window carrying the OPTIONAL `description` field, to prove
    it survives window-extraction -> (no-)reconcile -> memories.jsonl end to end."""

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        obj = {
            "memories": [
                {
                    "statement": f"memory-{cid}",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": ["t"],
                    "disposition": "confident",
                    "description": f"longer prose for {cid}: the why and the nuance behind the statement.",
                    "evidence": [{"quote": f"quotable-phrase-{cid}", "turn": 1, "speaker": "user"}],
                }
            ]
        }
        return ExtractResult(
            raw_text=json.dumps(obj),
            cost_usd=0.0,
            duration_ms=1,
            model_id=self.model_id,
            ok=True,
            meta={"canonical_model": self.model_id},
        )


class DescriptionFieldEndToEnd(unittest.TestCase):
    """The OPTIONAL `description` must reach memories.jsonl. Covers the no-reconcile pass-through,
    the absent-default-null case, and the reconcile-assembly carry (by recency, like temporal)."""

    def test_noreconcile_carries_description_to_output(self):
        _, ok, lines = _run_stub(_DescriptionStubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertTrue(lines)
        for ln in lines:
            m = json.loads(ln)
            self.assertIn("longer prose for", m["description"])

    def test_absent_description_default_null_in_output(self):
        _, ok, lines = _run_stub(_StubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        for ln in lines:
            self.assertIsNone(json.loads(ln)["description"])

    def test_reconcile_assembly_carries_latest_member_description(self):
        # Two candidates merged into one reconciled memory: description follows recency (latest
        # member C2), mirroring how scope/temporal fall back to the latest member.
        conv = transcript.Conversation(
            conversation_id="conv-x",
            tool="claude-code",
            source="local",
            date="2026-01-01",
            turns=[transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            truncated=False,
        )

        def _cand(label, wi, turn, quote, description):
            return {
                "label": label,
                "window_index": wi,
                "statement": "merged-stmt",
                "type": "fact",
                "scope_level": "personal",
                "scope_ref": None,
                "generality": "instance-specific",
                "tags": ["t"],
                "decision_outcome": None,
                "disposition": "confident",
                "intention_meta": None,
                "review": {"status": "active"},
                "valid_as_of": None,
                "temporal_kind": None,
                "volatility": None,
                "description": description,
                "evidence": [{"quote": quote, "turn": turn, "speaker": "user", "conversation_id": "conv-x"}],
            }

        candidates = [
            _cand("C1", 0, 1, "alpha", "older description"),
            _cand("C2", 1, 2, "beta", "newer description wins"),
        ]
        recon_obj = {
            "memories": [
                {
                    "members": ["C1", "C2"],
                    "type": "fact",
                    "statement": "merged-stmt",
                    "generality": "instance-specific",
                }
            ],
            "deleted": [],
        }
        finals, stats = rc._assemble_reconciled(conv, candidates, recon_obj, "facts")
        self.assertEqual(len(finals), 1, stats)
        self.assertEqual(finals[0]["description"], "newer description wins")


class _TypedTagsStubClient(_StubClient):
    """Emits one gate-surviving memory per window whose `tags` is the TYPED tag OBJECT (the form the
    model emits under constrained decoding), to prove the typed object survives window-extraction ->
    (no-)reconcile -> memories.jsonl as the memory's one `tags` field."""

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        obj = {
            "memories": [
                {
                    "statement": f"memory-{cid}",
                    "type": "fact",
                    "scope_level": "personal",
                    "generality": "instance-specific",
                    "tags": {
                        "domain": "backend",
                        "activity": "coding",
                        "component": ["cart"],
                        "tech": ["go"],
                        "concept": ["idempotency"],
                        "keyword": ["retry"],
                    },
                    "disposition": "confident",
                    "evidence": [{"quote": f"quotable-phrase-{cid}", "turn": 1, "speaker": "user"}],
                }
            ]
        }
        return ExtractResult(
            raw_text=json.dumps(obj),
            cost_usd=0.0,
            duration_ms=1,
            model_id=self.model_id,
            ok=True,
            meta={"canonical_model": self.model_id},
        )


class TagsTypedEndToEnd(unittest.TestCase):
    """A memory's tags reach memories.jsonl as one typed object, `tags`, in the no-reconcile
    pass-through and the reconcile assembly, whichever form the model answered in."""

    def test_noreconcile_carries_the_typed_object(self):
        _, ok, lines = _run_stub(_TypedTagsStubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertTrue(lines)
        for ln in lines:
            m = json.loads(ln)
            self.assertNotIn("tags_typed", m)
            # canonical output shape: single-value axes are scalars, open axes are lists
            self.assertEqual(
                m["tags"],
                {
                    "domain": "backend",
                    "activity": "coding",
                    "component": ["cart"],
                    "tech": ["go"],
                    "concept": ["idempotency"],
                    "keyword": ["retry"],
                },
            )

    def test_a_list_answer_reaches_the_output_typed(self):
        # The base stub answers tags as the list ["t"]: a bare word, so it lands in keyword.
        _, ok, lines = _run_stub(_StubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        for ln in lines:
            m = json.loads(ln)
            self.assertNotIn("tags_typed", m)
            self.assertEqual(m["tags"], {**rc.mc.empty_tags(), "keyword": ["t"]})

    def _typed_cand(self, label, wi, turn, quote, tags):
        return {
            "label": label,
            "window_index": wi,
            "statement": "merged-stmt",
            "type": "fact",
            "scope_level": "personal",
            "scope_ref": None,
            "generality": "instance-specific",
            "tags": tags,
            "tags_line": "",
            "decision_outcome": None,
            "disposition": "confident",
            "intention_meta": None,
            "review": {"status": "active"},
            "valid_as_of": None,
            "temporal_kind": None,
            "volatility": None,
            "evidence": [{"quote": quote, "turn": turn, "speaker": "user", "conversation_id": "conv-x"}],
        }

    def _conv(self):
        return transcript.Conversation(
            conversation_id="conv-x",
            tool="claude-code",
            source="local",
            date="2026-01-01",
            turns=[transcript.Turn(1, "user", "alpha"), transcript.Turn(2, "user", "beta")],
            truncated=False,
        )

    def test_reconcile_assembly_unions_typed_objects(self):
        # Two candidates merged into one reconciled memory: single-value domain/activity keep the first
        # non-null in member order; open axes union first-seen.
        candidates = [
            self._typed_cand(
                "C1",
                0,
                1,
                "alpha",
                {"domain": ["backend"], "activity": ["coding"], "component": ["cart"], "keyword": ["retry"]},
            ),
            self._typed_cand(
                "C2",
                1,
                2,
                "beta",
                {"domain": ["database"], "component": ["rate-limiter", "cart"], "tech": ["go"]},
            ),
        ]
        recon_obj = {
            "memories": [
                {
                    "members": ["C1", "C2"],
                    "type": "fact",
                    "statement": "merged-stmt",
                    "generality": "instance-specific",
                }
            ],
            "deleted": [],
        }
        finals, stats = rc._assemble_reconciled(self._conv(), candidates, recon_obj, "facts")
        self.assertEqual(len(finals), 1, stats)
        self.assertEqual(
            finals[0]["tags"],
            {
                "domain": "backend",  # first non-null domain in member order (C1)
                "activity": "coding",
                "component": ["cart", "rate-limiter"],
                "tech": ["go"],
                "concept": [],
                "keyword": ["retry", "database"],  # the losing domain moves to keyword
            },
        )

    def test_noreconcile_fold_unions_complementary_typed_tags(self):
        # Two verbatim restatements (same statement -> same memory_id) with COMPLEMENTARY typed
        # tags. The exact-restatement fold must UNION their tags, not keep only the first candidate's.
        candidates = [
            self._typed_cand("C1", 0, 1, "alpha", {"domain": ["backend"]}),
            self._typed_cand("C2", 1, 2, "beta", {"tech": ["go"]}),
        ]
        finals = rc._candidates_to_finals_noreconcile(self._conv(), candidates, "facts")
        self.assertEqual(len(finals), 1)  # same statement -> folded into one
        self.assertEqual(finals[0]["tags"], {**rc.mc.empty_tags(), "domain": "backend", "tech": ["go"]})


class LeversAndTokensAppearInMetrics(unittest.TestCase):
    """metrics.json must record which window shape a run used (so two lever arms are
    distinguishable after the fact) and where the arm's tokens went."""

    def test_explicit_cache_defaults_are_recorded(self):
        """A client without an explicit cache records the empty cache fields, not nulls."""
        metrics, ok, _ = _run_stub(_StubClient(), workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertFalse(metrics["explicit_cache"])
        self.assertEqual(metrics["explicit_caches"], [])
        self.assertEqual(metrics["explicit_cache_prompts"], [])
        self.assertEqual(metrics["explicit_cache_tokens"], 0)
        self.assertEqual(metrics["explicit_cache_storage_usd"], 0.0)
        self.assertEqual(metrics["http_retries"], 0)

    def test_explicit_cache_identity_and_storage_cost_are_recorded(self):
        client = _StubClient()
        client.explicit_cache = True
        client.explicit_caches = [
            {"name": "cachedContents/xyz", "tokens": 4_700, "lifetime_s": 600.0, "live": False}
        ]
        client.explicit_cache_prompts = [
            {
                "role": "extract",
                "status": _b.CACHE_STATUS_CACHED,
                "tokens": 4_700,
                "name": "cachedContents/xyz",
                "reason": None,
            },
            {
                "role": "reconcile",
                "status": _b.CACHE_STATUS_INLINE,
                "tokens": 1_490,
                "name": None,
                "reason": "provider refused as too small: 952 tokens, its minimum is 1024",
            },
        ]
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertTrue(metrics["explicit_cache"])
        self.assertEqual(
            [(c["name"], c["tokens"]) for c in metrics["explicit_caches"]], [("cachedContents/xyz", 4_700)]
        )
        self.assertEqual(metrics["explicit_cache_tokens"], 4_700)
        # The per-prompt decisions ride along, so a reader sees WHICH stages the lever reached.
        self.assertEqual(
            [(p["role"], p["status"]) for p in metrics["explicit_cache_prompts"]],
            [("extract", "cached"), ("reconcile", "inline")],
        )
        # Storage is priced on THAT RESOURCE's own 600s lifetime, not the full TTL and not the
        # run's elapsed, and stays BESIDE cost_usd_total rather than folded into it.
        self.assertAlmostEqual(
            metrics["explicit_cache_storage_usd"], mcl.gemini_cache_storage_cost(4_700, 600.0), places=6
        )
        self.assertLess(metrics["explicit_cache_storage_usd"], 0.00235)
        self.assertNotAlmostEqual(metrics["cost_usd_total"], 0.00235, places=6)

    def test_every_stages_cache_is_recorded_not_only_the_extract_clients(self):
        """A structured run gives reconcile its OWN unconstrained client, and each stage's prompt
        gets its own resource — so recording just the extraction client's cache under-reports both
        the identities and the storage bill."""
        extract = _StubClient()
        extract.explicit_cache = True
        extract.explicit_caches = [
            {"name": "cachedContents/extract", "tokens": 4_700, "lifetime_s": 900.0, "live": False}
        ]
        reconcile = _StubClient()
        reconcile.explicit_cache = True
        reconcile.explicit_caches = [
            {"name": "cachedContents/reconcile", "tokens": 5_100, "lifetime_s": 300.0, "live": False}
        ]
        metrics, ok, _ = _run_stub(extract, workers=1, n_conv=2, reconcile_client=reconcile)
        self.assertTrue(ok)
        self.assertEqual(
            [c["name"] for c in metrics["explicit_caches"]],
            ["cachedContents/extract", "cachedContents/reconcile"],
        )
        self.assertEqual(metrics["explicit_cache_tokens"], 4_700 + 5_100)
        # Each resource on ITS OWN lifetime — 900s and 300s here, not one run-wide elapsed for
        # both, which is what billing them together would assume.
        self.assertAlmostEqual(
            metrics["explicit_cache_storage_usd"],
            mcl.gemini_cache_storage_cost(4_700, 900.0) + mcl.gemini_cache_storage_cost(5_100, 300.0),
            places=6,
        )
        self.assertLess(
            metrics["explicit_cache_storage_usd"],
            mcl.gemini_cache_storage_cost(9_800, mcl.GEMINI_EXPLICIT_CACHE_TTL_S),
        )

    def test_recorded_knobs_let_an_analysis_rebuild_the_cache_id(self):
        """thinking_level and the responseSchema DIGEST fold into the cache id and are derivable
        from nothing else in the run, so they are recorded rather than left to be guessed."""
        client = _StubClient()
        client.thinking_level = "high"
        client.response_schema = {"type": "object"}
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertEqual(metrics["thinking_level"], "high")
        self.assertTrue(metrics["structured_tags"])
        self.assertEqual(metrics["response_schema_digest"], "687c20a6")

        bare = _StubClient()
        metrics, ok, _ = _run_stub(bare, workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertIsNone(metrics["thinking_level"])
        self.assertFalse(metrics["structured_tags"])
        self.assertIsNone(metrics["response_schema_digest"])

    def test_one_client_serving_both_stages_is_not_double_counted(self):
        # An UNSTRUCTURED run reuses one client for both stages, so the same resource must not be
        # billed twice just because it was reached through two references.
        client = _StubClient()
        client.explicit_cache = True
        client.explicit_caches = [
            {"name": "cachedContents/shared", "tokens": 4_700, "lifetime_s": 120.0, "live": False}
        ]
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=2, reconcile_client=client)
        self.assertTrue(ok)
        self.assertEqual(
            [(c["name"], c["tokens"]) for c in metrics["explicit_caches"]], [("cachedContents/shared", 4_700)]
        )
        self.assertAlmostEqual(
            metrics["explicit_cache_storage_usd"], mcl.gemini_cache_storage_cost(4_700, 120.0), places=6
        )
        self.assertEqual(metrics["explicit_cache_tokens"], 4_700)

    def test_http_retries_fold_from_the_per_call_records(self):
        client = _UsageClient()
        client.http_retries = 2
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=3)
        self.assertTrue(ok)
        self.assertEqual(metrics["http_retries"], 3 * 2)

    def test_arm_cost_is_repriced_from_usage_not_replayed(self):
        client = _UsageClient()
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=3)
        self.assertTrue(ok)
        # The stub reports cost_usd=0.0 but a usage block worth real money; the arm must bill the
        # usage. This is the assertion a warm replay could otherwise never make.
        per_call = mcl.gemini_call_cost(client.USAGE)
        self.assertGreater(per_call, 0.0)
        self.assertAlmostEqual(metrics["cost_usd_total"], 3 * per_call, places=9)

    def test_token_totals_fold_from_the_per_call_usage(self):
        client = _UsageClient()
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=3)
        self.assertTrue(ok)
        self.assertEqual(metrics["llm_calls"], 3)
        self.assertEqual(metrics["prompt_tokens_total"], 3 * 1_000)
        self.assertEqual(metrics["answer_tokens_total"], 3 * 200)
        self.assertEqual(metrics["thinking_tokens_total"], 3 * 4_000)
        self.assertEqual(metrics["cached_tokens_total"], 3 * 600)
        self.assertEqual(metrics["cache_creation_tokens_total"], 0)  # gemini reports none
        self.assertEqual(metrics["usage_missing_calls"], 0)

    def test_calls_without_usage_are_counted_not_silently_zeroed(self):
        # A cache entry written before usage was persisted replays without it. The run must SAY so
        # rather than report token totals that look complete and are not.
        metrics, ok, _ = _run_stub(_StubClient(), workers=1, n_conv=3)
        self.assertTrue(ok)
        self.assertEqual(metrics["usage_missing_calls"], 3)
        self.assertEqual(metrics["prompt_tokens_total"], 0)


class UsagePersistsIntoTheCacheRecord(unittest.TestCase):
    """The whole point of persisting usage: a cost post-mortem can run off the cache alone, and a
    warm replay reports the same token anatomy as the cold run — without the fingerprint moving."""

    def test_cold_call_stores_usage_and_warm_replay_reports_the_same_tokens(self):
        cache = cache_mod.Cache(tempfile.mkdtemp(prefix="usage_cache_"))
        client = _UsageClient()
        args = ("SYSPROMPT", "prompt-v", "cc:probe#w0", "CONVID=conv-0 window text")
        fp = cache_mod.fingerprint(
            "cc:probe#w0", "prompt-v", client.cache_model_id, "SYSPROMPT", "CONVID=conv-0 window text"
        )

        res, hit = rc._cached_call(cache, client, *args)
        self.assertFalse(hit)
        self.assertEqual(res["usage"]["promptTokenCount"], 1_000)
        self.assertEqual(res["cost_formula"], _b.COST_FORMULA)
        # Persisted under the same fingerprint as an entry without usage: usage rides alongside the
        # key, never inside it, so recording usage never orphans an entry.
        self.assertEqual(cache.get(fp)["usage"], res["usage"])
        self.assertEqual(cache.get(fp)["cost_formula"], _b.COST_FORMULA)

        replayed, hit = rc._cached_call(cache, client, *args)
        self.assertTrue(hit)
        self.assertEqual(replayed["usage"], res["usage"])
        self.assertEqual(client.calls, 1)  # the replay made no model call
        # Re-priced from the record's own usage on BOTH reads, so a warm replay exercises the live
        # formula instead of trusting the number the entry was written with.
        want = mcl.gemini_call_cost(res["usage"])
        self.assertAlmostEqual(mcl.cost_from_cache_record(res), want, places=9)
        self.assertAlmostEqual(mcl.cost_from_cache_record(replayed), want, places=9)

    def test_warm_and_cold_token_totals_match(self):
        acc_cold, acc_warm = rc._fresh_acc(), rc._fresh_acc()
        cache = cache_mod.Cache(tempfile.mkdtemp(prefix="usage_cache2_"))
        client = _UsageClient()
        args = ("SYSPROMPT", "prompt-v", "cc:probe#w0", "CONVID=conv-0 window text")
        _b.fold_usage(acc_cold, rc._cached_call(cache, client, *args)[0].get("usage"))
        _b.fold_usage(acc_warm, rc._cached_call(cache, client, *args)[0].get("usage"))
        self.assertEqual(acc_cold, acc_warm)
        self.assertEqual(acc_warm["thinking_tokens_total"], 4_000)


class APaidCallSurvivesLocalFailure(unittest.TestCase):
    """The provider bills when it answers. Everything after that — persisting the entry, parsing the
    response — is LOCAL work, and a failure there must not erase the call from the accounting. A
    cache that cannot write (full disk, bad permissions) is the case pinned here."""

    class _ExplodingCache:
        """Real Cache semantics for reads; put() raises the way mkstemp/write/rename can."""

        def __init__(self, root, fail_after=0):
            self._inner = cache_mod.Cache(root)
            self.fail_after = fail_after
            self.puts = 0

        def get(self, fp):
            return self._inner.get(fp)

        def put(self, fp, record):
            self.puts += 1
            if self.puts > self.fail_after:
                raise OSError(28, "No space left on device")
            return self._inner.put(fp, record)

    def test_a_cache_write_failure_does_not_un_spend_the_call(self):
        client = _UsageClient()
        store = self._ExplodingCache(tempfile.mkdtemp(prefix="boom_cache_"))
        acc = rc._fresh_acc()
        spent = {"cost": 0.0, "calls": 0, "hits": 0}

        def _account(res, hit):
            spent["hits" if hit else "calls"] += 1
            spent["cost"] += mcl.cost_from_cache_record(res)
            _b.fold_usage(acc, res.get("usage"))

        with self.assertRaises(OSError):
            rc._cached_call(
                store, client, "SYSPROMPT", "pv", "cc:x#w0", "CONVID=conv-0 window", account=_account
            )
        # the provider answered and billed, so the accounting must already have happened
        self.assertEqual(client.calls, 1)
        self.assertEqual(spent["calls"], 1)
        self.assertAlmostEqual(spent["cost"], mcl.gemini_call_cost(client.USAGE), places=9)
        self.assertEqual(acc["prompt_tokens_total"], 1_000)

    def test_a_stopped_run_still_reports_a_call_whose_write_failed(self):
        """End to end: the run stops (the write failure is fatal — continuing would re-pay for
        everything) and its metrics carry the call it paid for, named in partial_conversations."""
        data_dir = tempfile.mkdtemp(prefix="boom_run_")
        out_dir = os.path.join(data_dir, "runs", "boom")
        rows = [
            {
                "conversation_id": f"conv-{k}",
                "native_id": f"n-{k}",
                "tool": "claude-code",
                "source": "local",
                "path": "unused",
            }
            for k in range(2)
        ]

        def _fake_resolve(row):
            return transcript.Conversation(
                row["conversation_id"],
                "claude-code",
                "local",
                "2026-01-01",
                _single_turn(row["conversation_id"]),
                False,
            )

        client = _UsageClient()
        exploding = self._ExplodingCache(os.path.join(data_dir, "cache", "extractions"))
        with (
            mock.patch.object(transcript, "resolve_with_reason", _with_reason(_fake_resolve)),
            mock.patch.object(rc.cache_mod, "Cache", lambda root: exploding),
        ):
            metrics, ok = rc.run(
                rows,
                out_dir,
                client,
                arm="facts",
                cache_dir=os.path.join(data_dir, "cache", "extractions"),
                workers=1,
                log=lambda m: None,
            )

        self.assertFalse(ok, "an unwritable cache must stop the run, not re-pay for everything")
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertAlmostEqual(metrics["cost_usd_total"], mcl.gemini_call_cost(client.USAGE), places=9)
        self.assertEqual(metrics["prompt_tokens_total"], 1_000)
        self.assertEqual((metrics.get("fatal_error") or {}).get("stage"), "exception")
        self.assertEqual(
            [(p["conversation_id"], p["status"], p["calls"]) for p in metrics["partial_conversations"]],
            [("conv-0", "fatal", 1)],
        )
        self.assertAlmostEqual(
            metrics["partial_conversations"][0]["cost_usd"],
            round(mcl.gemini_call_cost(client.USAGE), 6),
            places=6,
        )

    def test_merge_folds_a_calls_spend_before_persisting_it(self):
        gm = merge._fresh_group_metrics()
        res = {
            "raw_text": "{}",
            "cost_usd": 0.01,
            "duration_ms": 5,
            "usage": {"promptTokenCount": 900, "candidatesTokenCount": 10, "thoughtsTokenCount": 100},
            "http_retries": 2,
        }
        merge._fold_call_spend(gm, res)
        self.assertAlmostEqual(gm["cost_usd_total"], mcl.gemini_call_cost(res["usage"]), places=9)
        self.assertEqual(gm["prompt_tokens_total"], 900)
        self.assertEqual(gm["http_retries"], 2)
        self.assertEqual(gm["model_duration_ms_total"], 5)


class _UnparseableClient(_StubClient):
    """Answers one target conversation's window with text that is not JSON (a truncated answer)."""

    def __init__(self, bad_cid: str):
        super().__init__()
        self.bad_cid = bad_cid

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        m = re.search(r"CONVID=(\S+)", conversation_text)
        if m and m.group(1) == self.bad_cid:
            return ExtractResult(
                'I could not finish the JSON {"memories": [',
                0.0,
                1,
                self.model_id,
                ok=True,
                meta={"canonical_model": self.model_id},
            )
        return super().extract(system_prompt, conversation_text)


class UnparseableWindowsAreCounted(unittest.TestCase):
    """A window whose answer cannot be parsed yields no memory; the run must say so rather than
    read as 'nothing to remember'."""

    def test_an_unparseable_window_is_counted_per_run_and_per_conversation(self):
        metrics, ok, lines = _run_stub(_UnparseableClient("conv-1"), workers=1, n_conv=3)
        self.assertTrue(ok)
        self.assertEqual(len(lines), 2)
        self.assertEqual(metrics["windows_unparseable"], 1)
        per = {p["conversation_id"]: p["windows_unparseable"] for p in metrics["per_conversation"]}
        self.assertEqual(per, {"conv-0": 0, "conv-1": 1, "conv-2": 0})

    def test_a_parseable_answer_with_no_memories_is_not_counted(self):
        class _Empty(_StubClient):
            def extract(self, system_prompt, conversation_text):
                return ExtractResult('{"memories": []}', 0.0, 1, self.model_id, ok=True)

        metrics, ok, _ = _run_stub(_Empty(), workers=1, n_conv=2)
        self.assertTrue(ok)
        self.assertEqual(metrics["windows_unparseable"], 0)
        self.assertEqual(metrics["conversations_empty"], 2)


class PriceCardIsRecorded(unittest.TestCase):
    """metrics.json says which price card a Gemini run was priced at, so a reader can tell when the
    provider's prices have moved since."""

    def test_a_gemini_client_records_the_rates_and_the_card_date(self):
        client = _StubClient()
        client.response_schema = None
        metrics, ok, _ = _run_stub(client, workers=1, n_conv=1, reconcile_client=None)
        self.assertTrue(ok)
        self.assertIsNone(metrics["price_card"])  # not a Gemini client: the provider reports cost

        gem = GeminiClient(api_key="fake-key-for-test", model_id="gemini-3.7-flash")
        card = rc.price_card_for(gem)
        self.assertEqual(
            card,
            {
                "source": "ai.google.dev/gemini-api/docs/pricing",
                "read_on": mcl.GEMINI_PRICE_CARD_DATE,
                "input_usd_per_1m": mcl.GEMINI_INPUT_USD_PER_1M,
                "cached_input_usd_per_1m": mcl.GEMINI_CACHED_INPUT_USD_PER_1M,
                "output_usd_per_1m": mcl.GEMINI_OUTPUT_USD_PER_1M,
                "cache_storage_usd_per_1m_token_hour": mcl.GEMINI_CACHE_STORAGE_USD_PER_1M_TOKEN_HOUR,
            },
        )


class _TwoCandidatesClient(_StubClient):
    """Two memories per window; the reconcile answer is whatever `reconcile_answer` says."""

    def __init__(self, reconcile_answer: str):
        super().__init__()
        self.reconcile_answer = reconcile_answer

    def extract(self, system_prompt, conversation_text):
        if "CANDIDATE MEMORIES" in conversation_text:
            return ExtractResult(self.reconcile_answer, 0.0, 1, self.model_id, ok=True)
        cid = re.search(r"CONVID=(\S+)", conversation_text).group(1)
        mems = [
            {
                "statement": f"memory-{cid}-{x}",
                "type": "fact",
                "scope_level": "personal",
                "tags": ["t"],
                "disposition": "confident",
                "evidence": [{"quote": x, "turn": 1, "speaker": "user"}],
            }
            for x in ("alpha", "beta")
        ]
        return ExtractResult(json.dumps({"memories": mems}), 0.0, 1, self.model_id, ok=True)


class ReconcileFallbacks(unittest.TestCase):
    def test_unparseable_reconcile_keeps_candidates(self):
        metrics, ok, lines = _run_stub(
            _TwoCandidatesClient('{"memories": [ {"members": ["C1"'), workers=1, n_conv=2
        )
        self.assertTrue(ok)
        self.assertEqual(len(lines), 4)  # every candidate kept, unreconciled
        self.assertEqual(metrics["reconcile_unparseable"], 2)
        self.assertEqual(metrics["reconcile_deleted_total"], 0)
        self.assertEqual([p["reconcile_unparseable"] for p in metrics["per_conversation"]], [1, 1])

    def _candidates(self, n, size):
        return [
            {
                "label": f"C{k + 1}",
                "window_index": 0,
                "statement": "s" * size,
                "type": "fact",
                "scope_level": "personal",
                "scope_ref": None,
                "generality": "instance-specific",
                "tags": {"keyword": ["t"]},
                "tags_line": "t",
                "disposition": "confident",
                "evidence": [],
            }
            for k in range(n)
        ]

    def test_candidates_within_the_budget_are_all_sent(self):
        conv = transcript.Conversation("c", "claude-code", "local", None, [], False)
        _, sent = rc.build_reconcile_text(conv, self._candidates(40, 400))  # about 21,000 chars
        self.assertEqual(len(sent), 40)

    def test_candidates_past_the_budget_pass_through_and_are_counted(self):
        conv = transcript.Conversation("c", "claude-code", "local", None, [], False)
        cands = self._candidates(140, 400)  # about 72,000 chars against the 60,000 budget
        text, sent = rc.build_reconcile_text(conv, cands)
        self.assertLess(len(sent), len(cands))
        self.assertEqual(sent, cands[len(cands) - len(sent) :])  # the newest ones are sent
        body = text.split("oldest window first) ---\n", 1)[1]
        self.assertLessEqual(len(body), 60_000)
        # one more (older) candidate would not have fit
        one_more, _ = rc.build_reconcile_text(conv, cands[-len(sent) - 1 :], budget=10**9)
        self.assertGreater(len(one_more.split("oldest window first) ---\n", 1)[1]) + 1, 60_000)

    def test_overflow_reaches_the_metrics(self):
        class _Many(_StubClient):
            def extract(self, system_prompt, conversation_text):
                if "CANDIDATE MEMORIES" in conversation_text:
                    return _reconcile_keep_all(conversation_text)
                mems = [
                    {
                        "statement": f"m{k} " + "w" * 600,
                        "type": "fact",
                        "scope_level": "personal",
                        "tags": ["t"],
                        "disposition": "confident",
                        "evidence": [{"quote": "quotable-phrase", "turn": 1, "speaker": "user"}],
                    }
                    for k in range(120)
                ]
                return ExtractResult(json.dumps({"memories": mems}), 0.0, 1, self.model_id, ok=True)

        metrics, ok, lines = _run_stub(_Many(), workers=1, n_conv=1)
        self.assertTrue(ok)
        self.assertGreater(metrics["reconcile_overflow"], 0)
        self.assertEqual(len(lines), 120)  # nothing lost, sent or not


class _TagFormsClient(_StubClient):
    """Answers three memories per window in the tag forms a model can send, and records the reconcile
    request it receives."""

    TAGS = (
        {"keyword": ["Retry"], "tech": ["go", "go"], "platform": "k8s", "domain": "backend"},
        ["domain:backend", "Bare", "topic:x"],
        {},
        ["tech:go", "tech:go", "Tech:PNPM"],
    )

    def __init__(self):
        super().__init__()
        self.reconcile_texts = []

    def extract(self, system_prompt, conversation_text):
        if "CANDIDATE MEMORIES" in conversation_text:
            self.reconcile_texts.append(conversation_text)
            return _reconcile_keep_all(conversation_text)
        mems = [
            {
                "statement": f"memory-{k}",
                "type": "fact",
                "scope_level": "personal",
                "tags": tags,
                "disposition": "confident",
                "evidence": [{"quote": "quotable-phrase", "turn": 1, "speaker": "user"}],
            }
            for k, tags in enumerate(self.TAGS)
        ]
        return ExtractResult(json.dumps({"memories": mems}), 0.0, 1, self.model_id, ok=True)


class ReconcileRequestTagsLine(unittest.TestCase):
    def test_the_tags_line_shows_the_answer_as_sent(self):
        # The reconcile request shows each candidate's tags as the model answered them, in answer
        # order, duplicates and all; the output carries the typed object instead. These lines are
        # pinned from memory format v2, so the request bytes, and every reconcile cache key, hold.
        client = _TagFormsClient()
        _, ok, lines = _run_stub(client, workers=1, n_conv=1)
        self.assertTrue(ok)
        (text,) = client.reconcile_texts
        self.assertEqual(
            [ln for ln in text.splitlines() if ln.startswith("    tags:")],
            [
                "    tags: keyword:retry, tech:go, tech:go, keyword:k8s, domain:backend",
                "    tags: domain:backend, Bare, topic:x",
                "    tags: ",
                "    tags: tech:go, tech:go, Tech:PNPM",
            ],
        )
        self.assertEqual(
            [json.loads(ln)["tags"] for ln in lines],
            [
                {**rc.mc.empty_tags(), "domain": "backend", "tech": ["go"], "keyword": ["retry", "k8s"]},
                {**rc.mc.empty_tags(), "domain": "backend", "keyword": ["bare", "topic:x"]},
                rc.mc.empty_tags(),
                {**rc.mc.empty_tags(), "tech": ["go", "pnpm"]},
            ],
        )


class WindowBudget(unittest.TestCase):
    def test_ten_thousand_characters_fit_one_window(self):
        self.assertEqual(rc.WINDOW_BUDGET, 10_000)
        self.assertEqual(len(rc.build_windows(_turns([("assistant", 5000), ("assistant", 5000)]))), 1)
        self.assertEqual(len(rc.build_windows(_turns([("assistant", 5000), ("assistant", 5001)]))), 2)
