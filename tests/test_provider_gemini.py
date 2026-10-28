# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""GeminiClient without any network call: urllib is stubbed, so nothing here talks to the provider.

Pinned: the request body shape and POST URL; the key's loading (file or environment, stripped of a
trailing newline); a normal response parses to ok=True with the joined candidate text and usage; a
prompt-level safety block, empty candidates and a candidate-level safety finishReason are terminal,
while a 5xx is retried; per-call pricing and token accounting; the explicit prompt cache, decided per
prompt, recovered once, and deleted at close; and the local cache identity."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from agent_memory_engine import providers as _providers
from agent_memory_engine.providers import base as _base
from agent_memory_engine.providers import gemini as _gemini

# One namespace over the provider modules, so the tests read like the module they guard.
mcl = types.SimpleNamespace(
    **{k: getattr(m, k) for m in (_base, _gemini, _providers) for k in dir(m) if not k.startswith("__")}
)


def _key(path: str) -> str:
    return _gemini.load_gemini_key(path)


class _FakeResp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _write_key(text: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "api.key")  # inside the test's tmp_path (conftest)
    with open(path, "w") as f:
        f.write(text)
    return path


_OK_PAYLOAD = {
    # two parts to prove the client JOINS them.
    "candidates": [
        {"content": {"parts": [{"text": '{"memories":'}, {"text": "[]}"}]}, "finishReason": "STOP"}
    ],
    # thoughtsTokenCount present — Gemini 3.7 Flash is a thinking model; these are billed as output.
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "thoughtsTokenCount": 20,
        "totalTokenCount": 35,
    },
    "modelVersion": "example-model-001",
}


class GeminiClientTest(unittest.TestCase):
    def setUp(self):
        # key file carries a trailing newline — the client must strip it.
        self.key_path = _write_key("SECRET-KEY-123\n")
        self.addCleanup(lambda: os.path.exists(self.key_path) and os.unlink(self.key_path))

    def test_key_read_from_file_and_stripped(self):
        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        self.assertEqual(c._api_key, "SECRET-KEY-123")  # trailing newline gone

    def test_response_schema_in_request_and_cache_key(self):
        schema = {"type": "object", "properties": {"memories": {"type": "array"}}, "required": ["memories"]}
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model", response_schema=schema)
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            c.extract("S", "C")
        self.assertEqual(captured["body"]["generationConfig"]["responseSchema"], schema)
        self.assertIn("+schema-", c.cache_model_id)  # constrained run must not share cache w/ unconstrained
        # no schema -> not in the request, no schema hash in the cache id (unchanged behavior)
        c2 = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        captured.clear()
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            c2.extract("S", "C")
        self.assertNotIn("responseSchema", captured["body"]["generationConfig"])
        self.assertNotIn("+schema-", c2.cache_model_id)

    def test_build_client_threads_response_schema(self):
        schema = {"type": "object", "required": ["memories"]}
        key = _key(self.key_path)
        self.assertEqual(
            mcl.build_client("gemini", "example-model", api_key=key, response_schema=schema).response_schema,
            schema,
        )
        self.assertIsNone(mcl.build_client("gemini", "example-model", api_key=key).response_schema)

    def test_request_shape_and_ok_response(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["content_type"] = req.get_header("Content-type")
            captured["api_key_header"] = req.get_header("X-goog-api-key")
            captured["timeout"] = timeout
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            res = c.extract("SYS PROMPT", "CONV TEXT")

        # --- request shape ---
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["content_type"], "application/json")
        # model id in the URL; the stripped key rides the x-goog-api-key HEADER, never the URL
        # (a key in the URL leaks into logs/error lines).
        self.assertIn("example-model:generateContent", captured["url"])
        self.assertNotIn("key=", captured["url"])
        self.assertEqual(captured["api_key_header"], "SECRET-KEY-123")
        self.assertNotIn("\n", captured["api_key_header"])
        body = captured["body"]
        self.assertEqual(body["system_instruction"]["parts"][0]["text"], "SYS PROMPT")
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "CONV TEXT")
        self.assertEqual(body["generationConfig"]["temperature"], 0)
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

        # --- parsed result ---
        self.assertTrue(res.ok, res.error)
        self.assertEqual(res.raw_text, '{"memories":[]}')  # parts joined
        self.assertEqual(res.model_id, "example-model")
        self.assertEqual(res.meta["usage"]["promptTokenCount"], 10)
        self.assertEqual(res.meta["canonical_model"], "example-model-001")
        # cost: output billed = candidates + thoughts (thinking tokens count as output).
        # 10 in, (5 + 20) out at 0.75/3.75 per 1M.
        expected = (
            10 / 1_000_000 * mcl.GEMINI_INPUT_USD_PER_1M + (5 + 20) / 1_000_000 * mcl.GEMINI_OUTPUT_USD_PER_1M
        )
        self.assertAlmostEqual(res.cost_usd, expected)

    def test_prompt_safety_block_is_not_ok_and_not_retried(self):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            return _FakeResp({"promptFeedback": {"blockReason": "SAFETY"}})

        c = mcl.GeminiClient(api_key=_key(self.key_path))
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            res = c.extract("s", "c")
        self.assertFalse(res.ok)
        self.assertIn("prompt-blocked", res.error)
        self.assertEqual(calls["n"], 1)  # terminal — no retry

    def test_empty_candidates_not_ok(self):
        def fake_urlopen(req, timeout=None):
            return _FakeResp({"candidates": []})

        c = mcl.GeminiClient(api_key=_key(self.key_path))
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            res = c.extract("s", "c")
        self.assertFalse(res.ok)
        self.assertIn("no-candidates", res.error)

    def test_candidate_safety_finish_not_ok(self):
        def fake_urlopen(req, timeout=None):
            return _FakeResp({"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]})

        c = mcl.GeminiClient(api_key=_key(self.key_path))
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            res = c.extract("s", "c")
        self.assertFalse(res.ok)
        self.assertIn("safety-blocked", res.error)

    def test_how_each_finish_reason_reads(self):
        # A missing reason and STOP in any case are a complete answer; RECITATION is a block; a cut
        # answer keeps the cost of the tokens it billed.
        text = '{"memories": []}'
        cases = [
            (None, True, None),
            ("STOP", True, None),
            ("stop", True, None),
            ("Stop", True, None),
            ("RECITATION", False, "safety-blocked: finish=RECITATION"),
            ("recitation", False, "safety-blocked: finish=recitation"),
            ("MAX_TOKENS", False, "incomplete: finish=MAX_TOKENS"),
            ("SOMETHING_NEW", False, "incomplete: finish=SOMETHING_NEW"),
            (1, False, "incomplete: finish=1"),
        ]
        for finish, ok, error in cases:
            with self.subTest(finish=finish):
                cand = {"content": {"parts": [{"text": text}]}}
                if finish is not None:
                    cand["finishReason"] = finish
                body = {"candidates": [cand], "usageMetadata": _OK_PAYLOAD["usageMetadata"]}
                c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=1)
                with mock.patch.object(
                    mcl.urllib.request, "urlopen", lambda req, timeout=None, b=body: _FakeResp(b)
                ):
                    res = c.extract("s", "t")
                self.assertEqual((res.ok, res.error), (ok, error))
                self.assertAlmostEqual(
                    res.cost_usd, mcl.gemini_call_cost(_OK_PAYLOAD["usageMetadata"]), places=12
                )

    def test_an_empty_complete_answer_reports_its_billed_cost(self):
        body = {
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
            "usageMetadata": _OK_PAYLOAD["usageMetadata"],
        }
        c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=1)
        with mock.patch.object(mcl.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(body)):
            res = c.extract("s", "t")
        self.assertEqual((res.ok, res.error), (False, "empty-candidate: finish=STOP"))
        self.assertAlmostEqual(res.cost_usd, mcl.gemini_call_cost(_OK_PAYLOAD["usageMetadata"]), places=12)

    def test_an_answer_cut_short_is_a_terminal_error_and_is_not_cached(self):
        # A truncated answer is deterministic for its input: retrying or replaying it from the cache
        # only returns the same cut. It fails the call, with its billed cost kept.
        from agent_memory_engine import cache as cache_mod
        from agent_memory_engine import extract as rc

        for finish in ("MAX_TOKENS", "SAFETY", "OTHER"):
            with self.subTest(finish=finish):
                calls = {"n": 0}

                def fake_urlopen(req, timeout=None, finish=finish, calls=calls):
                    calls["n"] += 1
                    body = {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": '{"memories":[{"statement":"Prefers tabs","type":"pref'}
                                    ]
                                },
                                "finishReason": finish,
                            }
                        ],
                        "usageMetadata": _OK_PAYLOAD["usageMetadata"],
                    }
                    return _FakeResp(body)

                c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=3)
                root = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, root)
                cache = cache_mod.Cache(root)
                with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
                    res, err = rc._cached_call(cache, c, "s", "v1", "conv", "t")
                self.assertIsNone(res)
                self.assertIn(f"finish={finish}", err)
                self.assertEqual(calls["n"], 1)
                self.assertEqual(os.listdir(root), [])

    def test_5xx_is_retried_then_succeeds(self):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise mcl.urllib.error.HTTPError(req.full_url, 503, "overloaded", {}, io.BytesIO(b"busy"))
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=3)
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda *_a: None),
        ):
            res = c.extract("s", "c")
        self.assertTrue(res.ok, res.error)
        self.assertEqual(calls["n"], 3)  # two 503s retried, third succeeds
        self.assertEqual(res.meta["retries"], 2)

    def test_4xx_client_error_not_retried(self):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise mcl.urllib.error.HTTPError(req.full_url, 400, "bad request", {}, io.BytesIO(b"nope"))

        c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=3)
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda *_a: None),
        ):
            res = c.extract("s", "c")
        self.assertFalse(res.ok)
        self.assertIn("http 400", res.error)
        self.assertEqual(calls["n"], 1)  # terminal — no retry

    def test_build_client_switch(self):
        gem = mcl.build_client("gemini", model=None, api_key=_key(self.key_path))
        self.assertIsInstance(gem, mcl.GeminiClient)
        self.assertEqual(gem.model_id, mcl.GeminiClient.DEFAULT_MODEL)
        # the default configuration: thinking medium and the explicit cache on
        self.assertEqual(gem.thinking_level, "medium")
        self.assertTrue(gem.explicit_cache)
        cla = mcl.build_client("claude", model=None)
        self.assertIsInstance(cla, mcl.ClaudeCLIClient)
        with self.assertRaises(mcl.GeminiKeyError):
            mcl.build_client("gemini", model=None, api_key=None)
        with self.assertRaises(ValueError):
            mcl.build_client("bogus")

    # --- thinking_level knob ------------------------------------------------ #
    def test_thinking_level_absent_by_default(self):
        """No thinking_level -> thinkingConfig omitted entirely (API default = medium), and
        cache_model_id stays the bare id."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            c.extract("s", "c")
        self.assertNotIn("thinkingConfig", captured["body"]["generationConfig"])
        self.assertEqual(c.cache_model_id, "example-model")

    def test_thinking_level_in_request_body(self):
        """thinking_level set -> generationConfig.thinkingConfig.thinkingLevel carries it."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model", thinking_level="high")
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            c.extract("s", "c")
        tc = captured["body"]["generationConfig"]["thinkingConfig"]
        self.assertEqual(tc["thinkingLevel"], "high")
        # bare id for reporting, level-encoded id for the cache.
        self.assertEqual(c.model_id, "example-model")
        self.assertEqual(c.cache_model_id, "example-model@high")

    def test_thinking_level_37_rejects_minimal(self):
        """gemini-3.7-flash + minimal must fail FAST at construction, not mid-run."""
        with self.assertRaises(ValueError):
            mcl.GeminiClient(
                api_key=_key(self.key_path), model_id="gemini-3.7-flash", thinking_level="minimal"
            )
        # 3.6 accepts minimal; 3.7 accepts the other levels.
        mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model", thinking_level="minimal")
        mcl.GeminiClient(api_key=_key(self.key_path), model_id="gemini-3.7-flash", thinking_level="low")

    def test_invalid_thinking_level_rejected(self):
        with self.assertRaises(ValueError):
            mcl.GeminiClient(api_key=_key(self.key_path), thinking_level="turbo")

    def test_cache_fingerprint_isolated_across_thinking_levels(self):
        """Two clients differing ONLY in thinking_level must produce DIFFERENT cache fingerprints
        for the same input — otherwise one sweep variant replays the other's stale output."""
        from agent_memory_engine import cache as cache_mod

        c_min = mcl.GeminiClient(
            api_key=_key(self.key_path), model_id="example-model", thinking_level="minimal"
        )
        c_high = mcl.GeminiClient(
            api_key=_key(self.key_path), model_id="example-model", thinking_level="high"
        )
        fp_min = cache_mod.fingerprint("conv-1", "v1", c_min.cache_model_id, "SYS", "TXT")
        fp_high = cache_mod.fingerprint("conv-1", "v1", c_high.cache_model_id, "SYS", "TXT")
        self.assertNotEqual(fp_min, fp_high)
        # And a bare (no-level) client differs from both.
        c_none = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        fp_none = cache_mod.fingerprint("conv-1", "v1", c_none.cache_model_id, "SYS", "TXT")
        self.assertNotEqual(fp_none, fp_min)
        self.assertNotEqual(fp_none, fp_high)

    def test_build_client_threads_thinking_level(self):
        gem = mcl.build_client(
            "gemini",
            model="example-model",
            api_key=_key(self.key_path),
            thinking_level="low",
            explicit_cache=False,
        )
        self.assertEqual(gem.thinking_level, "low")
        self.assertEqual(gem.cache_model_id, "example-model@low")
        # claude ignores the knob (no equivalent), and still builds.
        cla = mcl.build_client("claude", model=None, thinking_level="high")
        self.assertIsInstance(cla, mcl.ClaudeCLIClient)
        self.assertEqual(cla.cache_model_id, cla.model_id)

    # --- retry hardening ---------------------------------------------------- #
    def test_retry_after_header_seconds_honored(self):
        """A 429 with Retry-After: N (seconds) -> we sleep ~N (capped), not a fixed backoff."""
        calls = {"n": 0}
        slept = []

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise mcl.urllib.error.HTTPError(
                    req.full_url, 429, "slow down", {"Retry-After": "7"}, io.BytesIO(b"rate")
                )
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=5)
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda s: slept.append(s)),
        ):
            res = c.extract("s", "c")
        self.assertTrue(res.ok, res.error)
        self.assertEqual(slept, [7.0])  # honored the header exactly

    def test_retry_after_capped(self):
        """A huge Retry-After is capped so a run cannot stall indefinitely."""
        slept = []
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise mcl.urllib.error.HTTPError(
                    req.full_url, 503, "busy", {"Retry-After": "9999"}, io.BytesIO(b"x")
                )
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=5)
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda s: slept.append(s)),
        ):
            res = c.extract("s", "c")
        self.assertTrue(res.ok, res.error)
        self.assertEqual(slept, [mcl._GEMINI_RETRY_AFTER_CAP_S])

    def test_retry_delay_from_error_body_honored(self):
        """No Retry-After header, but a google.rpc.RetryInfo.retryDelay in the JSON body -> honored."""
        calls = {"n": 0}
        slept = []
        err_body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}],
                }
            }
        ).encode()

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise mcl.urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(err_body))
            return _FakeResp(_OK_PAYLOAD)

        c = mcl.GeminiClient(api_key=_key(self.key_path), max_attempts=5)
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda s: slept.append(s)),
        ):
            res = c.extract("s", "c")
        self.assertTrue(res.ok, res.error)
        self.assertEqual(slept, [3.0])

    def test_backoff_jitter_when_no_hint(self):
        """No server hint -> full-jitter exponential backoff: each sleep in [0, 2^attempt]."""
        for attempt in (1, 2, 3, 4):
            base = min(2.0**attempt, mcl._GEMINI_RETRY_AFTER_CAP_S)
            for _ in range(50):
                d = mcl.GeminiClient._backoff_delay(attempt, None)
                self.assertGreaterEqual(d, 0.0)
                self.assertLessEqual(d, base)

    def test_default_max_attempts_is_five(self):
        c = mcl.GeminiClient(api_key=_key(self.key_path))
        self.assertEqual(c.max_attempts, 5)

    # --- malformed 200 guard ------------------------------------------------ #
    def test_malformed_200_degrades_not_raises(self):
        """A 200 body with a wrong-typed field (usageMetadata as a list) must degrade to a terminal
        ok=False, never an uncaught AttributeError mid-run."""
        bad = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": [1, 2, 3],
        }  # a LIST where a dict is expected

        def fake_urlopen(req, timeout=None):
            return _FakeResp(bad)

        c = mcl.GeminiClient(api_key=_key(self.key_path))
        with mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen):
            res = c.extract("s", "c")  # must not raise
        self.assertFalse(res.ok)
        self.assertIn("unexpected-response-shape", res.error)


class CachedPromptBilling(unittest.TestCase):
    """Cost arithmetic per call. Thinking tokens bill as OUTPUT and are NOT in
    candidatesTokenCount; cachedContentTokenCount is the SUBSET of promptTokenCount the provider
    served from its own context cache, billed at a tenth of the fresh input rate."""

    def _cases(self):
        M = 1_000_000
        IN, CACHED, OUT = (
            mcl.GEMINI_INPUT_USD_PER_1M,
            mcl.GEMINI_CACHED_INPUT_USD_PER_1M,
            mcl.GEMINI_OUTPUT_USD_PER_1M,
        )
        return [
            # (name, usageMetadata, expected USD)
            (
                "no cached field at all — every prompt token at the fresh rate",
                {"promptTokenCount": M, "candidatesTokenCount": 0, "thoughtsTokenCount": 0},
                IN,
            ),
            (
                "cached reported as zero",
                {
                    "promptTokenCount": M,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": 0,
                    "cachedContentTokenCount": 0,
                },
                IN,
            ),
            (
                "cached is a subset: only the remainder bills at the fresh rate",
                {
                    "promptTokenCount": M,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": 0,
                    "cachedContentTokenCount": 400_000,
                },
                0.6 * IN + 0.4 * CACHED,
            ),
            (
                "the whole prompt served from cache",
                {
                    "promptTokenCount": M,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": 0,
                    "cachedContentTokenCount": M,
                },
                CACHED,
            ),
            (
                "thinking tokens bill as output alongside candidates",
                {"promptTokenCount": 0, "candidatesTokenCount": 200_000, "thoughtsTokenCount": 800_000},
                OUT,
            ),
            (
                "cached prompt + thinking together",
                {
                    "promptTokenCount": M,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": M,
                    "cachedContentTokenCount": M,
                },
                CACHED + OUT,
            ),
            # A provider reporting cached > prompt must never bill a negative amount.
            (
                "cached over-reported is clamped to the prompt total, never negative",
                {
                    "promptTokenCount": 100_000,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": 0,
                    "cachedContentTokenCount": M,
                },
                0.1 * CACHED,
            ),
            ("empty usage is free", {}, 0.0),
        ]

    def test_cost_table(self):
        for name, usage, want in self._cases():
            self.assertAlmostEqual(mcl.gemini_call_cost(usage), want, places=9, msg=name)
            self.assertGreaterEqual(mcl.gemini_call_cost(usage), 0.0, name)

    def test_cost_on_a_live_response_uses_the_cached_rate(self):
        """End-to-end through the client: a 200 body reporting a cache hit must bill it at the
        cached rate, and hand the raw usage block up in meta so it can be persisted."""
        usage = {
            "promptTokenCount": 1_000_000,
            "candidatesTokenCount": 0,
            "thoughtsTokenCount": 0,
            "cachedContentTokenCount": 1_000_000,
        }
        payload = {
            "candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
            "usageMetadata": usage,
            "modelVersion": "gemini-3.7-flash",
        }
        c = mcl.GeminiClient(api_key=_key(self.key_path))
        with mock.patch.object(mcl.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(payload)):
            res = c.extract("s", "c")
        self.assertTrue(res.ok, res.error)
        self.assertAlmostEqual(res.cost_usd, mcl.GEMINI_CACHED_INPUT_USD_PER_1M, places=9)
        self.assertEqual(res.meta["usage"], usage)

    def setUp(self):
        self.key_path = _write_key("k\n")


class UsageNormalization(unittest.TestCase):
    """model_client.normalize_usage turns a provider's raw usage block into the counters
    metrics.json reports, with `prompt` always the TOTAL prompt tokens.

    The trap the flag exists for: Gemini's cachedContentTokenCount is INSIDE promptTokenCount,
    while the Anthropic/Claude-CLI input_tokens EXCLUDES both cache reads and cache writes — so the
    real Claude prompt is input + cache_read + cache_creation. Reading one shape with the other's
    assumption misstates the prompt in both directions.

    An unreadable block returns None, which callers count as a usage-missing call rather than
    folding zeros — a cache entry written before usage was persisted, or a counter that arrives as
    a string, must not make an arm's token totals look complete.
    """

    def _cases(self):
        return [
            (
                "gemini: cached is a SUBSET of prompt, so prompt is unchanged",
                {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 40,
                    "cachedContentTokenCount": 5,
                },
                {
                    "family": "gemini",
                    "prompt": 10,
                    "answer": 3,
                    "thinking": 40,
                    "cached": 5,
                    "cache_creation": 0,
                },
            ),
            (
                "gemini without the optional cached field",
                {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 40},
                {
                    "family": "gemini",
                    "prompt": 10,
                    "answer": 3,
                    "thinking": 40,
                    "cached": 0,
                    "cache_creation": 0,
                },
            ),
            (
                "gemini with nulls",
                {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": None,
                    "thoughtsTokenCount": None,
                    "cachedContentTokenCount": None,
                },
                {
                    "family": "gemini",
                    "prompt": 10,
                    "answer": 0,
                    "thinking": 0,
                    "cached": 0,
                    "cache_creation": 0,
                },
            ),
            # input_tokens 7 EXCLUDES the 1 cache read and the 4 cache writes -> prompt total 12.
            (
                "claude cli: cached is ADDITIVE, so prompt = input + read + creation",
                {
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 1,
                    "cache_creation_input_tokens": 4,
                },
                {
                    "family": "claude-cli",
                    "prompt": 12,
                    "answer": 2,
                    "thinking": 0,
                    "cached": 1,
                    "cache_creation": 4,
                },
            ),
            (
                "claude cli with no cache fields at all",
                {"input_tokens": 7, "output_tokens": 2},
                {
                    "family": "claude-cli",
                    "prompt": 7,
                    "answer": 2,
                    "thinking": 0,
                    "cached": 0,
                    "cache_creation": 0,
                },
            ),
            ("absent usage", None, None),
            ("wrong type", [1, 2, 3], None),
            ("a dict in no shape we know", {"tokens": 5}, None),
            # A counter that is PRESENT but not a number is a provider/format surprise. Folding it
            # as 0 would shrink the arm's token totals while every call still looked accounted for,
            # so the whole block reads as unreadable instead.
            (
                "a non-numeric prompt count makes the block unreadable",
                {"promptTokenCount": "lots", "candidatesTokenCount": 3},
                None,
            ),
            (
                "a non-numeric secondary count makes the block unreadable",
                {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": ["1"]},
                None,
            ),
            (
                "a bool count makes the block unreadable",
                {"promptTokenCount": 10, "candidatesTokenCount": True},
                None,
            ),
        ]

    def test_normalize_table(self):
        for name, raw, want in self._cases():
            self.assertEqual(mcl.normalize_usage(raw), want, name)

    def test_fold_accumulates_and_counts_the_unreadable(self):
        acc = {k: 0 for k in mcl.USAGE_METRIC_KEYS}
        for _name, raw, _ in self._cases():
            mcl.fold_usage(acc, raw)
        self.assertEqual(acc["usage_missing_calls"], 6)  # 3 shapeless + 3 non-numeric
        self.assertEqual(acc["prompt_tokens_total"], 10 + 10 + 10 + 12 + 7)
        self.assertEqual(acc["thinking_tokens_total"], 40 + 40)
        self.assertEqual(acc["cached_tokens_total"], 5 + 1)
        self.assertEqual(acc["cache_creation_tokens_total"], 4)

    def test_fold_refuses_an_undeclared_accumulator(self):
        # Strict indexing: a caller that forgot to declare the keys must fail loudly rather than
        # start a fresh counter that later reads as a complete total.
        with self.assertRaises(KeyError):
            mcl.fold_usage({}, {"promptTokenCount": 4})
        with self.assertRaises(KeyError):
            mcl.fold_usage({}, None)

    def test_token_keys_exclude_the_call_counter(self):
        # A future "sum the tokens" loop must not be able to pick up a call count.
        self.assertNotIn("usage_missing_calls", mcl.USAGE_TOKEN_KEYS)
        self.assertEqual(set(mcl.USAGE_METRIC_KEYS), set(mcl.USAGE_TOKEN_KEYS) | set(mcl.USAGE_CALL_KEYS))
        self.assertTrue(all(k.endswith("_tokens_total") for k in mcl.USAGE_TOKEN_KEYS))


class CacheRecordCostProvenance(unittest.TestCase):
    """Two cost formulas now exist on disk, and a warm replay is the one path that never runs the
    live one. So a record stamps WHICH arithmetic priced it, and a record carrying a readable
    Gemini usage block is re-priced from that block on every read."""

    USAGE = {
        "promptTokenCount": 1_000_000,
        "candidatesTokenCount": 0,
        "thoughtsTokenCount": 0,
        "cachedContentTokenCount": 1_000_000,
    }

    def _cases(self):
        flat = 1_000_000 / 1_000_000 * mcl.GEMINI_INPUT_USD_PER_1M  # a flat price, as if nothing were cached
        split = mcl.GEMINI_CACHED_INPUT_USD_PER_1M  # the split price
        return [
            # (name, record, expected cost, expectation about which formula ran)
            (
                "a gemini record is RE-PRICED from its usage, not replayed from the stored number",
                {"cost_usd": flat, "cost_formula": "v1-flat", "usage": self.USAGE},
                split,
                "live",
            ),
            (
                "a record priced by the current formula re-prices to the same number it stored",
                {"cost_usd": split, "cost_formula": mcl.COST_FORMULA, "usage": self.USAGE},
                split,
                "live",
            ),
            # A record without a usage block: the stored number is all it has.
            ("a record without usage falls back to the stored cost", {"cost_usd": 0.4242}, 0.4242, "stored"),
            (
                "a record with an unreadable usage block falls back",
                {"cost_usd": 0.4242, "usage": {"tokens": 5}},
                0.4242,
                "stored",
            ),
            # The Claude CLI prices its own calls; there is no formula here to re-apply.
            (
                "a claude-cli record always falls back to the CLI's own total",
                {
                    "cost_usd": 0.4242,
                    "cost_formula": mcl.COST_FORMULA,
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                },
                0.4242,
                "stored",
            ),
            ("a record with no cost at all reads as free", {}, 0.0, "stored"),
            ("a null cost reads as free", {"cost_usd": None}, 0.0, "stored"),
        ]

    def test_cost_table(self):
        for name, record, want, which in self._cases():
            got = mcl.cost_from_cache_record(record)
            self.assertAlmostEqual(got, want, places=9, msg=name)
            if which == "live":
                self.assertAlmostEqual(got, mcl.gemini_call_cost(record["usage"]), places=9, msg=name)

    def test_a_new_record_carries_usage_the_formula_and_the_retries(self):
        r = mcl.ExtractResult(
            "{}", 0.5, 1, "gemini-3.7-flash", ok=True, meta={"usage": self.USAGE, "http_retries": 2}
        )
        self.assertEqual(
            mcl.usage_fields(r),
            {
                "usage": self.USAGE,
                "cost_formula": mcl.COST_FORMULA,
                "http_retries": 2,
                "cache_route": mcl.CACHE_STATUS_INLINE,
            },
        )
        self.assertEqual(mcl.COST_FORMULA, "v2-split")

    def test_a_result_with_no_usage_still_stamps_the_formula(self):
        r = mcl.ExtractResult("{}", 0.5, 1, "m", ok=True, meta={})
        self.assertEqual(
            mcl.usage_fields(r),
            {
                "usage": None,
                "cost_formula": mcl.COST_FORMULA,
                "http_retries": 0,
                "cache_route": mcl.CACHE_STATUS_INLINE,
            },
        )

    def test_key_from_file_wins_over_the_environment_and_is_stripped(self):
        path = _write_key("  file-key\n")
        self.addCleanup(os.unlink, path)
        self.assertEqual(_gemini.load_gemini_key(path, environ={"GEMINI_API_KEY": "env-key"}), "file-key")

    def test_key_from_the_environment(self):
        self.assertEqual(_gemini.load_gemini_key(None, environ={"GEMINI_API_KEY": " env-key\n"}), "env-key")

    def test_missing_or_empty_key_is_refused(self):
        empty = _write_key("\n")
        self.addCleanup(os.unlink, empty)
        cases = [
            (None, {}),
            (None, {"GEMINI_API_KEY": "   "}),
            (empty, {}),
            ("/nonexistent/key", {"GEMINI_API_KEY": "env-key"}),
        ]
        for path, env in cases:
            with self.subTest(path=path, env=env), self.assertRaises(mcl.GeminiKeyError):
                _gemini.load_gemini_key(path, environ=env)

    def test_the_key_never_appears_in_an_error(self):
        with self.assertRaises(mcl.GeminiKeyError) as ctx:
            _gemini.load_gemini_key(None, environ={})
        self.assertNotIn("SECRET", str(ctx.exception))
        with self.assertRaises(mcl.GeminiKeyError):
            mcl.GeminiClient(api_key="   ")


class CachedPrefixIsAlreadyPricedCorrectly(unittest.TestCase):
    """The point of the placement lever is that a cache-served prefix bills at a tenth of the input
    rate. Nothing new is needed to capture that: `cachedContentTokenCount` is a SUBSET of
    `promptTokenCount` (the probe's two calls reported the same 5506 prompt tokens, one with 3356
    of them cached), which is exactly what gemini_call_cost already assumes. A cache hit therefore
    lowers the bill without moving the prompt total, and the saving reads straight off
    cached_tokens_total — under either placement, so the two arms are directly comparable."""

    # The two probe calls, verbatim, prompt at the head of `contents`: call 1 cold, call 2 a hit.
    PROBE_COLD = {
        "promptTokenCount": 5506,
        "candidatesTokenCount": 43,
        "thoughtsTokenCount": 671,
        "totalTokenCount": 6220,
    }
    PROBE_HIT = {
        "promptTokenCount": 5506,
        "candidatesTokenCount": 39,
        "thoughtsTokenCount": 562,
        "totalTokenCount": 6107,
        "cachedContentTokenCount": 3356,
    }

    def test_the_cached_share_is_billed_at_the_cached_rate(self):
        cold = mcl.gemini_call_cost(self.PROBE_COLD)
        hit = mcl.gemini_call_cost(self.PROBE_HIT)
        # Isolate the INPUT side: the two calls spent different output tokens, so compare the
        # input-side saving against the rate difference on exactly the cached tokens.
        want_input_saving = (
            3356 * (mcl.GEMINI_INPUT_USD_PER_1M - mcl.GEMINI_CACHED_INPUT_USD_PER_1M) / 1_000_000
        )
        cold_input = mcl.gemini_call_cost({"promptTokenCount": 5506})
        hit_input = mcl.gemini_call_cost({"promptTokenCount": 5506, "cachedContentTokenCount": 3356})
        self.assertAlmostEqual(cold_input - hit_input, want_input_saving, places=9)
        self.assertAlmostEqual(want_input_saving, 0.0022653, places=7)
        # Both calls still price above zero, and the prompt total is unchanged by the hit.
        self.assertGreater(hit, 0.0)
        self.assertLess(hit, cold)
        self.assertEqual(self.PROBE_COLD["promptTokenCount"], self.PROBE_HIT["promptTokenCount"])

    def test_a_cache_hit_folds_into_cached_tokens_total(self):
        acc = {k: 0 for k in mcl.USAGE_METRIC_KEYS}
        mcl.fold_usage(acc, self.PROBE_COLD)
        mcl.fold_usage(acc, self.PROBE_HIT)
        # cached is a SUBSET, so the prompt total counts both calls in full and the saving is
        # readable directly off cached_tokens_total.
        self.assertEqual(acc["prompt_tokens_total"], 5506 * 2)
        self.assertEqual(acc["cached_tokens_total"], 3356)
        self.assertEqual(acc["cache_creation_tokens_total"], 0)
        self.assertEqual(acc["usage_missing_calls"], 0)


def _long(label: str, chars: int = 900) -> str:
    """A prompt long enough that the client will actually attempt a create for it. Below
    GEMINI_CACHE_SKIP_BELOW_CHARS the client skips the round trip by design, so a test about what
    the PROVIDER decides has to hand it something the provider would get to see."""
    body = f"[{label}] "
    return (body * (chars // len(body) + 1))[:chars]


class ExplicitCacheIsPerPromptAndOpportunistic(unittest.TestCase):
    """Eligibility is the PROVIDER's call, decided per prompt.

    The provider accepts the extraction prompts and refuses the reconcile prompt as too small (its
    floor is about 1,024 tokens). A local floor taken from the docs instead would delete resources the
    provider had already created and charged for. So this code applies no token floor of its own; a
    prompt the provider will not cache is sent INLINE, and the run is never aborted over one.
    """

    # Verbatim from a live create, the reconcile prompt:
    TOO_SMALL_BODY = (
        '{"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": '
        '"Cached content is too small. total_token_count=952, '
        'min_total_token_count=1024"}}'
    )

    def setUp(self):
        self.key_path = _write_key("k\n")
        self.calls = []
        self.minted = 0
        self.tokens_for = {}  # prompt text -> tokens the create should report
        self.fail_create = None  # (status, times, body) to fail creates with
        self.no_name = False  # make the create return a body with no resource name

    def _serve(self, delete_fails=False):
        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            self.calls.append((req.get_method(), req.full_url, body))
            if req.get_method() == "DELETE":
                if delete_fails:
                    raise mcl.urllib.error.HTTPError(req.full_url, 500, "e", {}, io.BytesIO(b"x"))
                return _FakeResp({})
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                if self.fail_create and self.fail_create[1] > 0:
                    status, left, fbody = self.fail_create
                    self.fail_create = (status, left - 1, fbody)
                    raise mcl.urllib.error.HTTPError(
                        req.full_url, status, "e", {}, io.BytesIO(fbody.encode())
                    )
                prompt = body["systemInstruction"]["parts"][0]["text"]
                self.minted += 1
                if self.no_name:
                    return _FakeResp({"usageMetadata": {"totalTokenCount": 4_700}})
                return _FakeResp(
                    {
                        "name": f"cachedContents/c{self.minted}",
                        "usageMetadata": {"totalTokenCount": self.tokens_for.get(prompt, 4_700)},
                        "model": "models/example-model",
                    }
                )
            return _FakeResp(_OK_PAYLOAD)

        return fake_urlopen

    def _client(self, **kwargs):
        kwargs.setdefault("explicit_cache", True)
        return mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model", **kwargs)

    def _creates(self):
        return [c for c in self.calls if c[1].endswith("/cachedContents") and c[0] == "POST"]

    def _generates(self):
        return [c for c in self.calls if "generateContent" in c[1]]

    def _mixed_cases(self):
        # (role, prompt, what the provider does, expected status)
        return [
            ("extract", _long("EXTRACT"), 4_776, mcl.CACHE_STATUS_CACHED),
            ("reconcile", _long("RECONCILE"), "too-small", mcl.CACHE_STATUS_INLINE),
            ("dedup", "tiny dedup prompt", "skip", mcl.CACHE_STATUS_INLINE),
        ]

    def _serve_mixed(self, cases):
        """Serve creates per prompt: a token count, the real too-small refusal, or nothing (the
        client skips it locally)."""
        outcome = {p: o for _, p, o, _ in cases}

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            self.calls.append((req.get_method(), req.full_url, body))
            if req.get_method() == "DELETE":
                return _FakeResp({})
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                prompt = body["systemInstruction"]["parts"][0]["text"]
                if outcome.get(prompt) == "too-small":
                    raise mcl.urllib.error.HTTPError(
                        req.full_url, 400, "e", {}, io.BytesIO(self.TOO_SMALL_BODY.encode())
                    )
                self.minted += 1
                return _FakeResp(
                    {
                        "name": f"cachedContents/c{self.minted}",
                        "usageMetadata": {"totalTokenCount": outcome[prompt]},
                    }
                )
            return _FakeResp(_OK_PAYLOAD)

        return fake_urlopen

    def test_a_refused_prompt_goes_inline_instead_of_aborting(self):
        cases = self._mixed_cases()
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve_mixed(cases)):
            records = c.preflight_explicit_caches([(r, p) for r, p, _, _ in cases])
            for role, prompt, _, _ in cases:
                self.assertTrue(c.extract(prompt, f"CONV {role}").ok, role)

        self.assertEqual([r["status"] for r in records], [e for _, _, _, e in cases])
        self.assertEqual([r["role"] for r in records], [r for r, _, _, _ in cases])
        cached = [r for r in records if r["status"] == mcl.CACHE_STATUS_CACHED]
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]["tokens"], 4_776)
        self.assertIsNone(cached[0]["reason"])
        # the refused prompt carries the PROVIDER's numbers, not this code's estimate of them
        refused = records[1]
        self.assertIn("952 tokens", refused["reason"])
        self.assertIn("minimum is 1024", refused["reason"])
        self.assertIsNone(refused["name"])
        # the tiny one never even asked
        self.assertIn("skipped without a create call", records[2]["reason"])
        self.assertEqual([x["tokens"] for x in c.explicit_caches], [4_776])
        # nothing the provider accepted was deleted again
        self.assertEqual([x for x in self.calls if x[0] == "DELETE"], [])

    def test_each_request_uses_the_route_its_own_prompt_resolved_to(self):
        cases = self._mixed_cases()
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve_mixed(cases)):
            for role, prompt, _, _ in cases:
                self.assertTrue(c.extract(prompt, f"CONV {role}").ok, role)
        gens = self._generates()
        self.assertEqual(len(gens), 3)
        big, *smalls = gens
        # the cacheable prompt travels BY REFERENCE, with no inline copy
        self.assertEqual(big[2]["cachedContent"], "cachedContents/c1")
        self.assertNotIn("system_instruction", big[2])
        self.assertNotIn("EXTRACT", json.dumps(big[2]))
        # every refused prompt travels inline, exactly as with the lever off
        for (role, prompt, _, _), (_, _, body) in zip(cases[1:], smalls, strict=False):
            self.assertNotIn("cachedContent", body, role)
            self.assertEqual(body["system_instruction"]["parts"][0]["text"], prompt, role)

    def test_a_prompt_the_docs_floor_would_have_rejected_is_kept(self):
        """The concrete harm of the docs' 4,096 figure: a resource the API accepted at 1,920 tokens
        (the default extract prompt) deleted again for being under a floor that is not the
        provider's — after paying to create it."""
        prompt = _long("MID")
        self.tokens_for = {prompt: 1_920}
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            name = c.resolve_explicit_cache(prompt)
        self.assertIsNotNone(name, "an accepted resource must not be second-guessed locally")
        self.assertEqual([x["tokens"] for x in c.explicit_caches], [1_920])
        self.assertEqual([x for x in self.calls if x[0] == "DELETE"], [])

    def test_a_prompt_far_below_any_floor_skips_the_create_entirely(self):
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            self.assertIsNone(c.resolve_explicit_cache("tiny"))
        self.assertEqual(self._creates(), [], "no point paying for a certain refusal")
        rec = next(iter(c._caches.values()))
        self.assertIn("skipped without a create call", rec["reason"])
        self.assertIn("4 chars", rec["reason"])

    def test_the_skip_threshold_cannot_pre_empt_the_provider(self):
        # Safe by construction: no text is denser than ~1 char/token, so 512 chars cannot reach the
        # 1,024-token floor. The threshold saves a round trip; it never makes the call.
        self.assertLess(mcl.GEMINI_CACHE_SKIP_BELOW_CHARS, mcl.GEMINI_EXPLICIT_CACHE_OBSERVED_MIN_TOKENS)
        self.assertEqual(mcl.GEMINI_EXPLICIT_CACHE_OBSERVED_MIN_TOKENS, 1_024)

    def _refusal_cases(self):
        return [
            # (name, status, body, reason fragment, retried?)
            (
                "the provider's real too-small refusal",
                400,
                self.TOO_SMALL_BODY,
                "952 tokens, its minimum is 1024",
                False,
            ),
            (
                "a too-small refusal with no parseable numbers",
                400,
                '{"error": {"message": "Cached content is too small."}}',
                "refused as too small",
                False,
            ),
            ("an unrelated 4xx", 403, "permission denied", "http 403", False),
            ("exhausted 5xx retries", 500, "boom", "http 500", True),
        ]

    def test_the_provider_decides_and_its_words_are_recorded(self):
        for name, status, body, needle, retried in self._refusal_cases():
            self.calls.clear()
            self.minted = 0
            self.fail_create = (status, 99, body)
            c = self._client()
            with (
                mock.patch.object(mcl.urllib.request, "urlopen", self._serve()),
                mock.patch.object(mcl.time, "sleep", lambda *_: None),
            ):
                res = c.extract(_long("ANY"), "CONV")
            self.assertTrue(res.ok, name)  # never fatal
            rec = next(iter(c._caches.values()))
            self.assertEqual(rec["status"], mcl.CACHE_STATUS_INLINE, name)
            self.assertIn(needle, rec["reason"], name)
            self.assertEqual(len(self._creates()) > 1, retried, name)
            self.assertNotIn("cachedContent", self._generates()[0][2], name)

    def test_a_create_with_no_resource_name_falls_back_inline(self):
        self.no_name = True
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            self.assertIsNone(c.resolve_explicit_cache(_long("NONAME")))
        self.assertIn("no resource name", next(iter(c._caches.values()))["reason"])

    def test_the_decision_is_made_once_per_prompt(self):
        big, small = _long("BIG"), _long("SMALL")
        self.tokens_for = {big: 4_776, small: 2_000}
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            for _ in range(3):
                self.assertTrue(c.extract(big, "X").ok)
                self.assertTrue(c.extract(small, "X").ok)
        self.assertEqual(len(self._creates()), 2, "two prompts, two creates — not six")

    def test_off_by_default_sends_no_cache_reference(self):
        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            self.assertTrue(c.extract("SYS PROMPT", "CONV").ok)
        self.assertEqual(self._creates(), [])
        self.assertNotIn("cachedContent", self.calls[0][2])
        self.assertEqual(self.calls[0][2]["system_instruction"]["parts"][0]["text"], "SYS PROMPT")
        self.assertEqual(c.explicit_caches, [])
        self.assertEqual(c.explicit_cache_prompts, [])

    def test_preflight_on_a_client_with_the_lever_off_reports_everything_inline(self):
        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        recs = c.preflight_explicit_caches([("extract", _long("P1")), ("reconcile", _long("P2"))])
        self.assertEqual([r["status"] for r in recs], [mcl.CACHE_STATUS_INLINE, mcl.CACHE_STATUS_INLINE])
        self.assertEqual(self._creates(), [])

    def test_create_body_shape(self):
        prompt = _long("EXTRACT")
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            c.resolve_explicit_cache(prompt)
        _, url, body = self._creates()[0]
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/cachedContents")
        self.assertEqual(body["model"], "models/example-model")  # Required, "models/" prefixed
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], prompt)
        self.assertEqual(body["ttl"], f"{mcl.GEMINI_EXPLICIT_CACHE_TTL_S}s")
        self.assertNotIn("contents", body)  # Optional, Input only
        self.assertLessEqual(len(body["displayName"]), 128)

    def test_transient_create_failures_retry_then_succeed(self):
        for status in (500, 429, 503):
            self.calls.clear()
            self.minted = 0
            self.fail_create = (status, 2, "blip")
            c = self._client()
            with (
                mock.patch.object(mcl.urllib.request, "urlopen", self._serve()),
                mock.patch.object(mcl.time, "sleep", lambda *_: None),
            ):
                res = c.extract(_long("ANY"), "CONV")
            self.assertTrue(res.ok, status)
            rec = next(iter(c._caches.values()))
            self.assertEqual(rec["status"], mcl.CACHE_STATUS_CACHED, status)
            self.assertEqual(len(self._creates()), 3, status)

    def test_a_missing_cache_mid_run_is_recreated_once(self):
        """A referenced cache can expire (TTL) or be evicted. The provider reports a plain 4xx that
        the transient rule would call terminal, so it is matched specifically and recovered by
        re-creating the resource — losing an arm to an expired cache would be absurd."""
        prompt = _long("BIG")
        state = {"gen": 0}

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            self.calls.append((req.get_method(), req.full_url, body))
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                self.minted += 1
                return _FakeResp(
                    {"name": f"cachedContents/c{self.minted}", "usageMetadata": {"totalTokenCount": 4_776}}
                )
            if req.get_method() == "DELETE":
                return _FakeResp({})
            state["gen"] += 1
            if state["gen"] == 1:
                raise mcl.urllib.error.HTTPError(
                    req.full_url, 403, "e", {}, io.BytesIO(b'{"error":{"message":"CachedContent not found"}}')
                )
            return _FakeResp(_OK_PAYLOAD)

        c = self._client()
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda *_: None),
        ):
            res = c.extract(prompt, "CONV")
        self.assertTrue(res.ok, res.error)
        self.assertEqual(len(self._creates()), 2, "the cache must be re-created once")
        gens = self._generates()
        self.assertEqual(len(gens), 2)
        self.assertEqual(gens[0][2]["cachedContent"], "cachedContents/c1")
        self.assertEqual(gens[1][2]["cachedContent"], "cachedContents/c2")
        # the FINAL route is what the result reports, and what a caller stores under
        self.assertEqual(res.meta["cached_content_name"], "cachedContents/c2")
        self.assertEqual(res.meta["cache_route"], mcl.CACHE_STATUS_CACHED)
        # a cache-missing failure is not a transient retry, so neither counter moves
        self.assertEqual(res.meta["http_retries"], 0)

    def test_a_recovery_whose_recreate_is_refused_reroutes_the_call_inline(self):
        """The route CAN change within a call: if the recreate is refused, the retry goes inline —
        and everything reported about the call must describe that, so the caller stores the answer
        under the inline identity rather than a cached-route one it never used."""
        prompt = _long("BIG")
        state = {"gen": 0, "created": 0}

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            self.calls.append((req.get_method(), req.full_url, body))
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                state["created"] += 1
                if state["created"] == 1:
                    return _FakeResp(
                        {"name": "cachedContents/c1", "usageMetadata": {"totalTokenCount": 4_776}}
                    )
                raise mcl.urllib.error.HTTPError(
                    req.full_url, 400, "e", {}, io.BytesIO(self.TOO_SMALL_BODY.encode())
                )
            if req.get_method() == "DELETE":
                return _FakeResp({})
            state["gen"] += 1
            if state["gen"] == 1:
                raise mcl.urllib.error.HTTPError(
                    req.full_url, 403, "e", {}, io.BytesIO(b"CachedContent not found")
                )
            return _FakeResp(_OK_PAYLOAD)

        c = self._client()
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(mcl.time, "sleep", lambda *_: None),
        ):
            route = c.call_route(prompt)
            self.assertEqual(route.route, mcl.CACHE_STATUS_CACHED)
            res = c.extract(prompt, "CONV", route=route)
        self.assertTrue(res.ok, res.error)
        # the retry went out with the prompt INLINE
        gens = self._generates()
        self.assertEqual(len(gens), 2)
        self.assertEqual(gens[0][2]["cachedContent"], "cachedContents/c1")
        self.assertNotIn("cachedContent", gens[1][2])
        # ...and the reported route follows, so the storage identity is the plain one
        self.assertEqual(res.meta["cache_route"], mcl.CACHE_STATUS_INLINE)
        self.assertIsNone(res.meta["cached_content_name"])
        self.assertNotEqual(mcl.storage_cache_model_id(res, route), route.cache_model_id)
        self.assertNotIn("+explicit-cache", mcl.storage_cache_model_id(res, route))

    def test_a_missing_cache_is_recovered_only_once_then_surfaces(self):
        def always_missing(req, timeout=None):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            self.calls.append((req.get_method(), req.full_url, body))
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                self.minted += 1
                return _FakeResp(
                    {"name": f"cachedContents/c{self.minted}", "usageMetadata": {"totalTokenCount": 4_776}}
                )
            raise mcl.urllib.error.HTTPError(
                req.full_url, 403, "e", {}, io.BytesIO(b"CachedContent not found")
            )

        c = self._client()
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", always_missing),
            mock.patch.object(mcl.time, "sleep", lambda *_: None),
        ):
            res = c.extract(_long("BIG"), "CONV")
        self.assertFalse(res.ok)
        self.assertIn("http 403", res.error)
        self.assertEqual(len(self._generates()), 2, "one recovery, then it surfaces")

    def test_concurrent_workers_on_one_prompt_mint_one_resource(self):
        import threading as _t

        c = self._client()
        errors = []
        prompt = _long("ONE")

        def _go():
            try:
                c.resolve_explicit_cache(prompt)
            except Exception as e:  # pragma: no cover - a failure here IS the assertion
                errors.append(e)

        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            threads = [_t.Thread(target=_go) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self._creates()), 1)
        self.assertEqual(len(c.explicit_caches), 1)

    def test_close_deletes_every_cached_resource_and_is_idempotent(self):
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            for p in (_long("A"), _long("B")):
                c.resolve_explicit_cache(p)
            self.calls.clear()
            c.close()
            c.close()
        deleted = sorted(u.rsplit("/v1beta/", 1)[1] for m, u, _ in self.calls if m == "DELETE")
        self.assertEqual(deleted, ["cachedContents/c1", "cachedContents/c2"])
        # the ledger is KEPT — a closed client still knows what it paid for — with every life ended
        self.assertEqual(len(c.explicit_caches), 2)
        self.assertTrue(all(not x["live"] for x in c.explicit_caches))

    def test_a_failed_delete_does_not_fail_the_run(self):
        c = self._client()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve(delete_fails=True)):
            c.resolve_explicit_cache(_long("A"))
            c.close()  # must not raise
        self.assertTrue(all(not x["live"] for x in c.explicit_caches))

    def test_close_on_a_client_with_no_cache_is_a_no_op(self):
        c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        c.close()
        self.assertEqual(self.calls, [])

    def test_storage_is_priced_on_how_long_the_cache_actually_lived(self):
        cases = [
            ("a ten-minute run", 4_776, 600, 0.000398),
            ("a full hour", 4_776, 3_600, 0.002388),
            ("longer than the TTL is capped at the TTL", 4_776, 99_999, 0.002388),
            ("a 1M-token cache for an hour", 1_000_000, 3_600, 0.50),
            ("half an hour", 1_000_000, 1_800, 0.25),
            ("no tokens", 0, 3_600, 0.0),
            ("negative elapsed cannot bill", 4_776, -5, 0.0),
        ]
        for name, tokens, held, want in cases:
            self.assertAlmostEqual(mcl.gemini_cache_storage_cost(tokens, held), want, places=6, msg=name)


class CacheIdentity(unittest.TestCase):
    """Every output-shaping knob folds into the local cache id through ONE builder, appended only when
    set, so an explicit-cache call never replays into an inline-prompt run and two thinking levels
    never share an answer."""

    def setUp(self):
        self.key_path = _write_key("k\n")
        self.addCleanup(os.unlink, self.key_path)

    def _fp(self, cache_model_id):
        from agent_memory_engine import cache as cache_mod

        return cache_mod.fingerprint("conv-1", "v1", cache_model_id, "SYS", "TXT")

    def _cases(self):
        # (name, kwargs to gemini_cache_model_id, expected id)
        return [
            ("bare", {}, "m"),
            ("thinking", {"thinking_level": "high"}, "m@high"),
            ("schema", {"response_schema": {"type": "object"}}, "m+schema-687c20a6"),
            ("explicit cache", {"explicit_cache": True}, "m+explicit-cache"),
            (
                "every knob at once",
                {"thinking_level": "high", "response_schema": {"type": "object"}, "explicit_cache": True},
                "m@high+schema-687c20a6+explicit-cache",
            ),
            ("explicit defaults spell the bare id", {"explicit_cache": False}, "m"),
        ]

    def test_suffix_table(self):
        for name, kwargs, want in self._cases():
            with self.subTest(name=name):
                self.assertEqual(mcl.gemini_cache_model_id("m", **kwargs), want)

    def test_the_default_configuration_keeps_its_id(self):
        from agent_memory_engine import contract as mc

        schema = mc.structured_output_schema(sorted(mc.VALID_TYPES))
        self.assertEqual(
            mcl.gemini_cache_model_id(
                "gemini-3.7-flash", thinking_level="medium", response_schema=schema, explicit_cache=True
            ),
            "gemini-3.7-flash@medium+schema-4c07912b+explicit-cache",
        )

    def test_schema_digest_is_pinned_to_a_literal(self):
        """8 hex chars of a sha1 over the schema's canonical JSON. A changed recipe would orphan
        every cache entry on disk, so it fails here instead of as a silent full-price re-run."""
        self.assertEqual(mcl.gemini_schema_digest({"type": "object"}), "687c20a6")
        self.assertEqual(
            mcl.gemini_schema_digest({"b": 1, "a": 2}), mcl.gemini_schema_digest({"a": 2, "b": 1})
        )

    def test_each_knob_gives_a_distinct_fingerprint(self):
        fps = {name: self._fp(mcl.gemini_cache_model_id("m", **kwargs)) for name, kwargs, _ in self._cases()}
        distinct = {n: fp for n, fp in fps.items() if n != "explicit defaults spell the bare id"}
        self.assertEqual(len(set(distinct.values())), len(distinct), distinct)
        self.assertEqual(fps["bare"], fps["explicit defaults spell the bare id"])

    def test_the_live_client_uses_the_shared_builder(self):
        combos = [
            {},
            {"explicit_cache": True},
            {"thinking_level": "high"},
            {"thinking_level": "high", "response_schema": {"type": "object"}, "explicit_cache": True},
        ]
        for kwargs in combos:
            with self.subTest(knobs=sorted(kwargs)):
                c = mcl.GeminiClient(api_key=_key(self.key_path), model_id="m", **kwargs)
                self.assertEqual(c.cache_model_id, mcl.gemini_cache_model_id("m", **kwargs))

    def test_builder_refuses_unknown_knob_values(self):
        with self.assertRaises(ValueError):
            mcl.gemini_cache_model_id("m", thinking_level="turbo")


class CacheRecoveryIsScopedToTheResourceThatFailed(unittest.TestCase):
    """Two workers can be holding the SAME cache name. When the first recreates after a
    cache-missing error, the second's already-in-flight failure arrives late referring to the dead
    name. Evicting by prompt alone would then discard the fresh resource the first worker just paid
    for — orphaning it, since only the live map was consulted at teardown — and mint a third."""

    def setUp(self):
        self.key_path = _write_key("k\n")
        self.minted = 0
        self.deleted = []

    def _client(self):
        return mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model", explicit_cache=True)

    def _serve(self, c):
        def fake_urlopen(req, timeout=None):
            if req.get_method() == "DELETE":
                self.deleted.append(req.full_url.rsplit("/v1beta/", 1)[1])
                return _FakeResp({})
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                self.minted += 1
                return _FakeResp(
                    {"name": f"cachedContents/c{self.minted}", "usageMetadata": {"totalTokenCount": 7_400}}
                )
            return _FakeResp(_OK_PAYLOAD)

        return fake_urlopen

    def test_a_late_failure_on_a_superseded_name_is_a_no_op(self):
        c = self._client()
        P = _long("P")
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve(c)):
            first = c.resolve_explicit_cache(P)  # both workers see c1
            self.assertEqual(first, "cachedContents/c1")

            # worker A: its request failed on c1, so it invalidates c1 and re-resolves -> c2
            self.assertTrue(c.invalidate_explicit_cache(P, used_name="cachedContents/c1"))
            second = c.resolve_explicit_cache(P)
            self.assertEqual(second, "cachedContents/c2")

            # worker B: its in-flight request ALSO failed on c1, arriving late. Scoped to the name,
            # this must change nothing — c2 stays live and no third resource is minted.
            self.assertFalse(c.invalidate_explicit_cache(P, used_name="cachedContents/c1"))
            self.assertEqual(c.resolve_explicit_cache(P), "cachedContents/c2")
            self.assertEqual(self.minted, 2)

    def test_every_resource_ever_minted_is_deleted_and_accounted(self):
        c = self._client()
        P = _long("P")
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve(c)):
            c.resolve_explicit_cache(P)
            c.invalidate_explicit_cache(P, used_name="cachedContents/c1")
            c.resolve_explicit_cache(P)  # c1 superseded by c2
            # the superseded resource is still in the ledger: it existed, so it was paid for
            self.assertEqual(
                [x["name"] for x in c.explicit_caches], ["cachedContents/c1", "cachedContents/c2"]
            )
            self.assertEqual(sum(x["tokens"] for x in c.explicit_caches), 14_800)
            # c1's life ENDED when the provider said it was gone, so it is billed for less than
            # c2 (still live) and teardown has nothing to delete for it.
            c1, c2 = c.explicit_caches
            self.assertFalse(c1["live"])
            self.assertTrue(c2["live"])
            self.assertLessEqual(c1["lifetime_s"], c2["lifetime_s"])
            c.close()
        # only the resource that still existed is deleted; the ledger survives for accounting
        self.assertEqual(self.deleted, ["cachedContents/c2"])
        self.assertEqual([x["name"] for x in c.explicit_caches], ["cachedContents/c1", "cachedContents/c2"])
        self.assertTrue(all(not x["live"] for x in c.explicit_caches))

    def test_the_per_prompt_record_follows_the_recreate(self):
        c = self._client()
        P = _long("P")
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve(c)):
            c.preflight_explicit_caches([("extract", P)])
            self.assertEqual([r["name"] for r in c.explicit_cache_prompts], ["cachedContents/c1"])
            c.invalidate_explicit_cache(P, used_name="cachedContents/c1")
            c.resolve_explicit_cache(P)
            # the record must name the LIVE resource, not the one preflight happened to see
            self.assertEqual([r["name"] for r in c.explicit_cache_prompts], ["cachedContents/c2"])
            self.assertEqual([r["role"] for r in c.explicit_cache_prompts], ["extract"])

    def test_invalidating_an_unknown_prompt_or_name_reports_nothing_dropped(self):
        c = self._client()
        P = _long("P")
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve(c)):
            self.assertFalse(c.invalidate_explicit_cache(_long("never-resolved")))
            c.resolve_explicit_cache(P)
            self.assertFalse(c.invalidate_explicit_cache(P, used_name="cachedContents/nope"))
            self.assertTrue(c.invalidate_explicit_cache(P))  # unscoped still drops


class InlineRouteKeysOnThePlainIdentity(unittest.TestCase):
    """A prompt the provider refuses to cache is sent INLINE — a request byte-identical to a
    run's without the cache. Keying it under `+explicit-cache` would refuse the entry that already
    holds that exact answer and re-pay full price for it."""

    TOO_SMALL_BODY = ExplicitCacheIsPerPromptAndOpportunistic.TOO_SMALL_BODY
    BIG = _long("BIG")
    SMALL = _long("SMALL")

    def setUp(self):
        self.key_path = _write_key("k\n")
        self.minted = 0

    def _serve(self):
        """Accepts BIG, refuses SMALL exactly as the live API refuses the reconcile prompt."""

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            if req.get_method() == "DELETE":
                return _FakeResp({})
            if "cachedContents" in req.full_url and req.get_method() == "POST":
                if body["systemInstruction"]["parts"][0]["text"] == self.SMALL:
                    raise mcl.urllib.error.HTTPError(
                        req.full_url, 400, "e", {}, io.BytesIO(self.TOO_SMALL_BODY.encode())
                    )
                self.minted += 1
                return _FakeResp(
                    {"name": f"cachedContents/c{self.minted}", "usageMetadata": {"totalTokenCount": 4_776}}
                )
            return _FakeResp(_OK_PAYLOAD)

        return fake_urlopen

    def _clients(self):
        plain = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model")
        levered = mcl.GeminiClient(api_key=_key(self.key_path), model_id="example-model", explicit_cache=True)
        return plain, levered

    def _id(self, client, prompt):
        return mcl.resolve_call_route(client, prompt).cache_model_id

    def test_a_refused_prompt_keys_exactly_as_a_run_without_the_cache(self):
        plain, levered = self._clients()
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", self._serve()),
            mock.patch.object(mcl.time, "sleep", lambda *_: None),
        ):
            # the inline route keys IDENTICALLY to a run without the lever
            self.assertEqual(self._id(levered, self.SMALL), self._id(plain, self.SMALL))
            # the cached route does not — it is a different request
            self.assertNotEqual(self._id(levered, self.BIG), self._id(plain, self.BIG))
            self.assertIn("+explicit-cache", self._id(levered, self.BIG))
            self.assertNotIn("+explicit-cache", self._id(levered, self.SMALL))

    def test_the_local_cache_actually_hits_across_the_two_runs(self):
        """End to end through the real store: a run without the cache leaves an entry, and a --explicit-cache
        run whose prompt the provider refuses must REPLAY it instead of calling the model."""
        from agent_memory_engine import cache as cache_mod

        store = cache_mod.Cache(tempfile.mkdtemp(prefix="inline_route_"))
        plain, levered = self._clients()
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", self._serve()),
            mock.patch.object(mcl.time, "sleep", lambda *_: None),
        ):
            fp_plain = cache_mod.fingerprint(
                "conv#w0", "pv", self._id(plain, self.SMALL), self.SMALL, "WINDOW"
            )
            store.put(fp_plain, {"raw_text": "{}", "cost_usd": 0.5, "duration_ms": 1})
            fp_levered = cache_mod.fingerprint(
                "conv#w0", "pv", self._id(levered, self.SMALL), self.SMALL, "WINDOW"
            )
        self.assertEqual(fp_levered, fp_plain)
        self.assertIsNotNone(store.get(fp_levered), "the inline route must reuse the entry")

    def test_a_cached_prompt_does_not_reuse_the_inline_entry(self):
        from agent_memory_engine import cache as cache_mod

        store = cache_mod.Cache(tempfile.mkdtemp(prefix="cached_route_"))
        plain, levered = self._clients()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            fp_plain = cache_mod.fingerprint("conv#w0", "pv", self._id(plain, self.BIG), self.BIG, "W")
            store.put(fp_plain, {"raw_text": "{}", "cost_usd": 0.5, "duration_ms": 1})
            fp_levered = cache_mod.fingerprint("conv#w0", "pv", self._id(levered, self.BIG), self.BIG, "W")
        self.assertNotEqual(fp_levered, fp_plain)
        self.assertIsNone(store.get(fp_levered))

    def test_the_route_is_persisted_in_the_record(self):
        _, levered = self._clients()
        with (
            mock.patch.object(mcl.urllib.request, "urlopen", self._serve()),
            mock.patch.object(mcl.time, "sleep", lambda *_: None),
        ):
            for prompt, want in ((self.BIG, mcl.CACHE_STATUS_CACHED), (self.SMALL, mcl.CACHE_STATUS_INLINE)):
                res = levered.extract(prompt, "CONV")
                self.assertTrue(res.ok, want)
                self.assertEqual(res.meta["cache_route"], want, want)
                self.assertEqual(mcl.usage_fields(res)["cache_route"], want, want)

    def test_the_lever_off_client_never_reports_a_cached_route(self):
        plain, _ = self._clients()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            res = plain.extract(self.BIG, "CONV")
        self.assertEqual(res.meta["cache_route"], mcl.CACHE_STATUS_INLINE)
        self.assertIsNone(res.meta["cached_content_name"])

    def test_one_resolution_serves_both_the_fingerprint_and_the_request(self):
        """The route object is resolved ONCE and passed to both, so no recovery in between can make
        the fingerprint and the request disagree about which route was taken."""
        _, levered = self._clients()
        with mock.patch.object(mcl.urllib.request, "urlopen", self._serve()):
            route = mcl.resolve_call_route(levered, self.BIG)
            self.assertEqual(route.route, mcl.CACHE_STATUS_CACHED)
            self.assertEqual(route.cache_name, "cachedContents/c1")
            self.assertIn("+explicit-cache", route.cache_model_id)
            res = mcl.extract_with_route(levered, self.BIG, "CONV", route)
        self.assertTrue(res.ok)
        # no recovery happened, so the stored identity is the one the lookup used
        self.assertEqual(mcl.storage_cache_model_id(res, route), route.cache_model_id)

    def test_a_single_route_client_answers_with_its_client_wide_identity(self):
        stub = mcl.ClaudeCLIClient(model_id="example-model")
        route = mcl.resolve_call_route(stub, "any prompt")
        self.assertEqual(route.route, mcl.CACHE_STATUS_INLINE)
        self.assertIsNone(route.cache_name)
        self.assertEqual(route.cache_model_id, stub.cache_model_id)
        self.assertFalse(stub.accepts_call_route)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class UnusableRetryHintsMeanNoHint(unittest.TestCase):
    """A 429/5xx whose retry hint cannot be read is still an ordinary transient failure: it is retried
    on the normal backoff, never raised out of the client."""

    def _client_with(self, responses):
        calls = iter(responses)

        def fake_urlopen(req, timeout=None):
            r = next(calls)
            if isinstance(r, BaseException):
                raise r
            return r

        c = mcl.GeminiClient(api_key="fake-key-for-test", model_id="example-model", max_attempts=3)
        return c, fake_urlopen

    def test_table(self):
        cases = [
            ("unparseable Retry-After date", {"Retry-After": "soon"}, b"busy"),
            ("error body is a JSON list", {}, b"[1, 2]"),
            ("error body is a JSON string", {}, b'"busy"'),
            ("retryDelay is not a duration", {}, b'{"error":{"details":[{"retryDelay":"later"}]}}'),
        ]
        for name, headers, body in cases:
            with self.subTest(name):
                err = mcl.urllib.error.HTTPError("u", 503, "busy", headers, io.BytesIO(body))
                c, fake = self._client_with([err, _FakeResp(_OK_PAYLOAD)])
                with (
                    mock.patch.object(mcl.urllib.request, "urlopen", fake),
                    mock.patch.object(mcl.time, "sleep") as slept,
                ):
                    r = c.extract("p", "t")
                self.assertTrue(r.ok, r.error)
                self.assertEqual(r.meta["http_retries"], 1)
                self.assertEqual(slept.call_count, 1)

    def test_parsers_return_none(self):
        self.assertIsNone(mcl._parse_retry_after_header("soon"))
        self.assertIsNone(mcl._retry_delay_from_error_body("[1, 2]"))
        self.assertIsNone(mcl._retry_delay_from_error_body('"x"'))
        self.assertEqual(mcl._parse_retry_after_header("7"), 7.0)


class HttpErrorsAreClosed(unittest.TestCase):
    def test_the_error_response_is_closed(self):
        body = io.BytesIO(b'{"error": {"message": "bad request"}}')
        err = mcl.urllib.error.HTTPError("u", 400, "bad", {}, body)
        c = mcl.GeminiClient(api_key="fake-key-for-test", model_id="example-model", max_attempts=1)
        with mock.patch.object(mcl.urllib.request, "urlopen", side_effect=err):
            r = c.extract("p", "t")
        self.assertFalse(r.ok)
        self.assertTrue(body.closed)
