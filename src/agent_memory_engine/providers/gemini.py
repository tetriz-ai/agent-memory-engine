# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""GeminiClient: the Google Generative Language REST API over stdlib urllib, no third-party package.

The API key comes from the caller (see load_gemini_key: `--gemini-key-file`, else the
GEMINI_API_KEY environment variable). It travels in the `x-goog-api-key` header, never the URL, so
it cannot leak into a logged request line or an error message.

Each system prompt can be held in a provider-side `cachedContents` resource for the run (the
explicit cache, on by default) and referenced by name instead of being re-sent on every call; the
cached share then bills at the cached-input rate. Whether a prompt can be cached is the provider's
call: it refuses prompts under about 1,024 tokens. The decision is made per prompt, a prompt the
provider will not cache is sent inline, and every resource is deleted when the run ends.
"""

from __future__ import annotations

import contextlib
import email.utils
import hashlib
import json
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

from .base import (
    CACHE_STATUS_CACHED,
    CACHE_STATUS_INLINE,
    CallRoute,
    ExtractResult,
    ModelClient,
    normalize_usage,
)

GEMINI_KEY_ENV = "GEMINI_API_KEY"

# The generateContent endpoint; the model id is appended per request.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Same service, one level up: the cachedContents collection is a sibling of models/.
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# A pinned concrete id, not a moving alias, so two runs never mix underlying versions.
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"

# gemini-3.7-flash pricing, USD per 1M tokens, from the provider's price card
# (ai.google.dev/gemini-api/docs/pricing) as read on GEMINI_PRICE_CARD_DATE. The card listed these
# as introductory rates through 2026-12-31, doubling on 2027-01-01, and the provider may change it at
# any time; every run records the rates it used in metrics.json ("price_card").
GEMINI_PRICE_CARD_DATE = "2026-08-26"
GEMINI_INPUT_USD_PER_1M = 0.75
GEMINI_OUTPUT_USD_PER_1M = 3.75
# Prompt tokens the provider served from a context cache bill at a tenth of the fresh input rate.
GEMINI_CACHED_INPUT_USD_PER_1M = 0.075

# Explicit cache storage bills for as long as the resource exists, at $0.50 / 1M cached tokens /
# hour. No per-call usage block reports it, so a run records it separately.
GEMINI_CACHE_STORAGE_USD_PER_1M_TOKEN_HOUR = 0.50
GEMINI_EXPLICIT_CACHE_TTL_S = 3_600

# The provider's own floor, measured from its refusal
# ("400 Cached content is too small. total_token_count=952, min_total_token_count=1024").
# This code applies no floor of its own: the create is attempted and a refusal is recorded verbatim.
GEMINI_EXPLICIT_CACHE_OBSERVED_MIN_TOKENS = 1_024

# The one local pre-check, only to skip a create that cannot succeed: this many characters is a few
# hundred tokens at most, far under the provider's floor.
GEMINI_CACHE_SKIP_BELOW_CHARS = 512

GEMINI_CACHE_DISPLAY_NAME = "agent-memory-engine-prompt"

# Reasoning knob: generationConfig.thinkingConfig.thinkingLevel. "minimal" is rejected by
# gemini-3.7-flash; the constructor turns that into a clear ValueError up front.
GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
DEFAULT_THINKING_LEVEL = "medium"

GEMINI_TIMEOUT_S = 300

# Candidate-level finishReasons that mean the response was suppressed. Deterministic for a given
# input, so terminal, never retried.
_GEMINI_BLOCK_FINISH = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION", "SPII"}

# A server-suggested wait (Retry-After header or RetryInfo.retryDelay) is honored, capped so a
# multi-minute suggestion cannot stall a run.
_GEMINI_RETRY_AFTER_CAP_S = 60.0
_RETRY_DELAY_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)s\s*$")

# Searched, not anchored: the generate path formats errors as "http 503: <detail>", a management
# call prefixes the method and path ("POST cachedContents -> http 503: ...").
_HTTP_STATUS_RE = re.compile(r"http (\d{3}):")

# The provider reporting that a referenced cache resource is gone (expired or evicted mid-run). A
# plain 4xx, matched by status and by the body naming the cache, and recovered once by re-creating.
_CACHE_MISSING_CODES = (400, 403, 404)
_CACHE_MISSING_HINT = re.compile(r"cached\s*content|cachedcontents", re.IGNORECASE)

_CACHE_TOO_SMALL_RE = re.compile(
    r"too small.*?total_token_count=(\d+).*?min_total_token_count=(\d+)",
    re.IGNORECASE | re.DOTALL,
)


class GeminiKeyError(ValueError):
    """No usable Gemini API key: the key file is missing or empty, or the variable is unset."""


def load_gemini_key(key_file: str | None = None, environ: dict | None = None) -> str:
    """The Gemini API key from `key_file` when given, else from GEMINI_API_KEY.

    Surrounding whitespace is stripped, a trailing newline included: the API rejects a key with a
    stray newline. A missing or empty key raises GeminiKeyError before any work starts."""
    env = os.environ if environ is None else environ
    if key_file:
        try:
            with open(os.path.expanduser(key_file), encoding="utf-8") as f:
                key = f.read().strip()
        except OSError as e:
            raise GeminiKeyError(f"cannot read the Gemini key file {key_file}: {e.strerror}") from None
        if not key:
            raise GeminiKeyError(f"the Gemini key file {key_file} is empty")
        return key
    key = (env.get(GEMINI_KEY_ENV) or "").strip()
    if not key:
        raise GeminiKeyError(f"no Gemini API key: set {GEMINI_KEY_ENV}, or pass --gemini-key-file PATH")
    return key


def _http_status(error: str | None) -> int | None:
    m = _HTTP_STATUS_RE.search(error) if error else None
    return int(m.group(1)) if m else None


def _is_transient_api_error(error: str | None) -> bool:
    """Whether a cache create/delete failure is worth retrying: 429 and 5xx yes, other 4xx no,
    transport blips yes. The same rule the generate path applies."""
    code = _http_status(error)
    if code is not None:
        return code == 429 or 500 <= code < 600
    return bool(error) and ("network:" in error or "transport:" in error)  # type: ignore[operator]


def _parse_cache_too_small(error: str | None) -> str | None:
    """A readable reason carrying the provider's own token numbers, or None if this failure is not
    a too-small refusal."""
    m = _CACHE_TOO_SMALL_RE.search(error) if error else None
    if m:
        return f"provider refused as too small: {m.group(1)} tokens, its minimum is {m.group(2)}"
    if error and "too small" in error.lower():
        return "provider refused as too small"
    return None


def _is_cache_missing_error(error: str | None) -> bool:
    code = _http_status(error)
    return bool(code in _CACHE_MISSING_CODES and error and _CACHE_MISSING_HINT.search(error))


def _parse_retry_after_header(value: str | None) -> float | None:
    """Retry-After is delta-seconds ("5") or an HTTP-date. Seconds from now, or None. A past
    HTTP-date clamps to 0."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(int(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None  # an unreadable hint is no hint: the normal backoff applies
    if dt is None:
        return None
    try:
        return max(0.0, dt.timestamp() - time.time())
    except (OverflowError, ValueError, OSError):
        return None


def _retry_delay_from_error_body(detail: str) -> float | None:
    """A google.rpc.RetryInfo.retryDelay ("5s") from a JSON error body, if present."""
    try:
        obj = json.loads(detail)
    except Exception:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("error"), dict):
        return None  # a body of another shape carries no hint
    details = obj["error"].get("details") or []
    if not isinstance(details, list):
        return None
    for d in details:
        if not isinstance(d, dict):
            continue
        rd = d.get("retryDelay")
        if isinstance(rd, str):
            m = _RETRY_DELAY_RE.match(rd)
            if m:
                return float(m.group(1))
    return None


def gemini_schema_digest(response_schema: dict) -> str:
    """The 8-char digest of a responseSchema that the cache id carries."""
    return hashlib.sha1(json.dumps(response_schema, sort_keys=True).encode()).hexdigest()[:8]


def gemini_request_body(
    system_prompt: str,
    conversation_text: str,
    gen_config: dict,
    cached_content_name: str | None = None,
) -> dict:
    """The generateContent request for one call. With an explicit cache the prompt is not in the
    request at all, only its cache name; otherwise it rides in `system_instruction`."""
    body: dict = {}
    if cached_content_name is not None:
        body["contents"] = [{"role": "user", "parts": [{"text": conversation_text}]}]
        body["cachedContent"] = cached_content_name
    else:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}
        body["contents"] = [{"role": "user", "parts": [{"text": conversation_text}]}]
    body["generationConfig"] = gen_config
    return body


def gemini_call_cost(usage: dict) -> float:
    """USD for one call from its usageMetadata.

    Reasoning tokens land in `thoughtsTokenCount`, bill as output, and are not part of
    `candidatesTokenCount`. `cachedContentTokenCount` is the subset of `promptTokenCount` served
    from a context cache at a tenth of the input rate, clamped to the prompt total so a provider
    reporting cached > prompt never produces a negative bill."""
    prompt_tokens = usage.get("promptTokenCount") or 0
    cached_tokens = min(usage.get("cachedContentTokenCount") or 0, prompt_tokens)
    fresh_tokens = prompt_tokens - cached_tokens
    out_tokens = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
    return (
        fresh_tokens / 1_000_000 * GEMINI_INPUT_USD_PER_1M
        + cached_tokens / 1_000_000 * GEMINI_CACHED_INPUT_USD_PER_1M
        + out_tokens / 1_000_000 * GEMINI_OUTPUT_USD_PER_1M
    )


def gemini_cache_storage_cost(
    cached_tokens: int, held_seconds: float, ttl_seconds: float = GEMINI_EXPLICIT_CACHE_TTL_S
) -> float:
    """USD to hold an explicit cache for as long as it existed, capped at its TTL."""
    billable = max(0.0, min(held_seconds, ttl_seconds))
    return (cached_tokens / 1_000_000) * GEMINI_CACHE_STORAGE_USD_PER_1M_TOKEN_HOUR * (billable / 3_600)


def cost_from_cache_record(record: dict) -> float:
    """What one (possibly replayed) cached call cost. A record with a readable Gemini usage block
    is re-priced from it at read time, so a replay exercises the live formula. Anything else (a
    Claude CLI record, which reports its own total) falls back to the stored number."""
    n = normalize_usage(record.get("usage"))
    if n is not None and n["family"] == "gemini":
        return gemini_call_cost(record["usage"])
    return float(record.get("cost_usd") or 0.0)


def gemini_cache_model_id(
    model_id: str,
    thinking_level: str | None = None,
    response_schema: dict | None = None,
    explicit_cache: bool = False,
) -> str:
    """The id the local result cache keys on: the model plus every knob that shapes a call's output
    and is not already covered by (model, system prompt, text). Each suffix is appended only when
    the knob is set, and the order is part of the contract: reordering orphans existing entries."""
    if thinking_level is not None and thinking_level not in GEMINI_THINKING_LEVELS:
        raise ValueError(
            f"invalid thinking_level {thinking_level!r}; expected one of "
            f"{sorted(GEMINI_THINKING_LEVELS)} or None"
        )
    cache_id = f"{model_id}@{thinking_level}" if thinking_level else model_id
    if response_schema is not None:
        cache_id = f"{cache_id}+schema-{gemini_schema_digest(response_schema)}"
    if explicit_cache:
        # The marker, never the resource name: the name is minted per run.
        cache_id = f"{cache_id}+explicit-cache"
    return cache_id


class GeminiClient(ModelClient):
    """ModelClient over the Gemini REST API. temperature 0, JSON response type, an optional
    responseSchema for constrained decoding, and an optional thinking level."""

    DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
    accepts_call_route = True

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_GEMINI_MODEL,
        api_base: str = GEMINI_API_BASE,
        timeout_s: int = GEMINI_TIMEOUT_S,
        max_attempts: int = 5,
        thinking_level: str | None = None,
        response_schema: dict | None = None,
        explicit_cache: bool = False,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise GeminiKeyError("empty Gemini API key")
        if thinking_level is not None and thinking_level not in GEMINI_THINKING_LEVELS:
            raise ValueError(
                f"invalid thinking_level {thinking_level!r}; expected one of "
                f"{sorted(GEMINI_THINKING_LEVELS)} or None (API default = medium)"
            )
        # gemini-3.7-flash rejects "minimal" with an API error mid-run; fail before any spend.
        if thinking_level == "minimal" and "-3.7" in model_id:
            raise ValueError(
                f"model {model_id!r} does not support thinking_level 'minimal'; use low/medium/high"
            )
        self.model_id = model_id
        self.api_base = api_base
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.thinking_level = thinking_level
        self.response_schema = response_schema
        self.explicit_cache = explicit_cache
        self._api_key = api_key.strip()
        # Caches are keyed by system prompt: one client can serve two stages with different prompts.
        # Created lazily and shared by every worker thread; the lock keeps concurrent workers from
        # minting one resource each for the same prompt.
        self._cache_lock = threading.Lock()
        self._caches: dict[str, dict] = {}  # prompt key -> the live decision
        self._cache_inflight: dict[str, threading.Event] = {}
        # Every resource ever minted, kept apart from the live map so a superseded one is still
        # deleted at close and still counted in storage.
        self._created: list[dict] = []
        self._preflight_roles: list[tuple[str, str]] = []
        # The inline-route identity carries no explicit-cache suffix: a prompt the provider will not
        # cache is sent inline, byte-identical to a run without the cache, and reuses that entry.
        self._cache_model_id_inline = gemini_cache_model_id(
            model_id, thinking_level=thinking_level, response_schema=response_schema
        )
        self._cache_model_id_cached = gemini_cache_model_id(
            model_id,
            thinking_level=thinking_level,
            response_schema=response_schema,
            explicit_cache=True,
        )
        self.cache_model_id = self._cache_model_id_cached if explicit_cache else self._cache_model_id_inline

    # ------------------------------------------------------------------ #
    # explicit prompt cache
    # ------------------------------------------------------------------ #
    def _api_call(
        self, path: str, method: str = "GET", payload: dict | None = None, timeout_s: int | None = None
    ) -> dict:
        """One cachedContents management call. Raises RuntimeError on an HTTP failure."""
        url = f"{GEMINI_API_ROOT}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            with e, contextlib.suppress(Exception):
                detail = e.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"{method} {path} -> http {e.code}: {detail}") from None
        return json.loads(raw) if raw.strip() else {}

    def call_route(self, system_prompt: str) -> CallRoute:
        """Resolve this prompt's route once; the caller hands it to both the cache key and the
        request, so the two cannot disagree."""
        if not self.explicit_cache:
            return CallRoute(CACHE_STATUS_INLINE, None, self._cache_model_id_inline)
        name = self.resolve_explicit_cache(system_prompt)
        if name:
            return CallRoute(CACHE_STATUS_CACHED, name, self._cache_model_id_cached)
        return CallRoute(CACHE_STATUS_INLINE, None, self._cache_model_id_inline)

    def _cache_key(self, system_prompt: str) -> str:
        h = hashlib.sha256()
        for part in (self.model_id, str(GEMINI_EXPLICIT_CACHE_TTL_S), system_prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _create_cache(self, system_prompt: str) -> dict:
        """Attempt the cachedContents create for one prompt and return the decision: cached, or
        inline with the reason. Never raises: a prompt that cannot be cached is sent inline.
        Transient failures are retried on the generate path's backoff."""
        if len(system_prompt) < GEMINI_CACHE_SKIP_BELOW_CHARS:
            return {
                "status": CACHE_STATUS_INLINE,
                "name": None,
                "tokens": 0,
                "reason": f"prompt is {len(system_prompt)} chars, far below anything the "
                f"provider caches; skipped without a create call",
            }
        body = {
            "model": f"models/{self.model_id}",
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "ttl": f"{GEMINI_EXPLICIT_CACHE_TTL_S}s",
            "displayName": GEMINI_CACHE_DISPLAY_NAME,
        }
        jitter_seed = hashlib.sha256(("cache\x00" + system_prompt).encode("utf-8")).hexdigest()
        last_error = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            try:
                created = self._api_call("cachedContents", method="POST", payload=body)
            except RuntimeError as e:
                last_error = str(e)
                too_small = _parse_cache_too_small(last_error)
                if too_small is not None:
                    return {"status": CACHE_STATUS_INLINE, "name": None, "tokens": 0, "reason": too_small}
                if not _is_transient_api_error(last_error) or attempt == self.max_attempts:
                    return {
                        "status": CACHE_STATUS_INLINE,
                        "name": None,
                        "tokens": 0,
                        "reason": f"create failed: {last_error[:200]}",
                    }
                time.sleep(self._backoff_delay(attempt, None, jitter_seed))
                continue
            name = created.get("name")
            tokens = int(((created.get("usageMetadata") or {}).get("totalTokenCount")) or 0)
            if not name:
                return {
                    "status": CACHE_STATUS_INLINE,
                    "name": None,
                    "tokens": 0,
                    "reason": f"create returned no resource name: {str(created)[:200]}",
                }
            with self._cache_lock:
                # Monotonic stamps, used only to derive a lifetime for the storage bill.
                self._created.append(
                    {
                        "name": name,
                        "key": self._cache_key(system_prompt),
                        "tokens": tokens,
                        "created_at": time.monotonic(),
                        "ended_at": None,
                        "gone": False,
                    }
                )
            return {"status": CACHE_STATUS_CACHED, "name": name, "tokens": tokens, "reason": None}
        return {
            "status": CACHE_STATUS_INLINE,
            "name": None,
            "tokens": 0,
            "reason": f"create failed: {last_error[:200]}",
        }

    def resolve_explicit_cache(self, system_prompt: str) -> str | None:
        """The cache resource to reference for this prompt, or None to send it inline. Decided once
        per prompt; concurrent workers wanting the same prompt wait for that one decision."""
        key = self._cache_key(system_prompt)
        while True:
            with self._cache_lock:
                decided = self._caches.get(key)
                if decided is not None:
                    return decided["name"]
                waiting = self._cache_inflight.get(key)
                if waiting is None:
                    self._cache_inflight[key] = threading.Event()
                    break
            waiting.wait()
        try:
            decision = self._create_cache(system_prompt)
        except Exception as e:  # _create_cache is written not to raise
            decision = {
                "status": CACHE_STATUS_INLINE,
                "name": None,
                "tokens": 0,
                "reason": f"create raised unexpectedly: {e!r}"[:200],
            }
        with self._cache_lock:
            self._caches[key] = decision
            event = self._cache_inflight.pop(key, None)
        if event is not None:
            event.set()
        return decision["name"]

    def invalidate_explicit_cache(self, system_prompt: str, used_name: str | None = None) -> bool:
        """Forget this prompt's decision so the next call re-decides it. Scoped to the resource the
        failing request used: a late failure naming an already-superseded resource is a no-op, so it
        cannot throw away the fresh one another worker just created."""
        with self._cache_lock:
            key = self._cache_key(system_prompt)
            current = self._caches.get(key)
            if current is None:
                return False
            if used_name is not None and current.get("name") != used_name:
                return False
            del self._caches[key]
            self._end_resource_life(current.get("name"), gone=True)
            return True

    def _end_resource_life(self, name: str | None, gone: bool = False) -> None:
        """Stamp when a resource stopped existing. `gone=True` means the provider says it is not
        there, so close() must not try to delete it. Caller holds the lock."""
        if not name:
            return
        stamp = time.monotonic()
        for entry in self._created:
            if entry["name"] == name:
                if entry["ended_at"] is None:
                    entry["ended_at"] = stamp
                entry["gone"] = entry["gone"] or gone

    def _bounded_lifetime(self, entry: dict, now: float) -> float:
        end = entry["ended_at"] if entry["ended_at"] is not None else now
        return max(0.0, min(end - entry["created_at"], GEMINI_EXPLICIT_CACHE_TTL_S))

    def cache_storage_cost_usd(self) -> float:
        now = time.monotonic()
        with self._cache_lock:
            return sum(
                gemini_cache_storage_cost(e["tokens"], self._bounded_lifetime(e, now)) for e in self._created
            )

    def preflight_explicit_caches(self, roles: list[tuple[str, str]]) -> list[dict]:
        """Decide every prompt a run will use before its first call, and return one role-labelled
        record per prompt, so the operator sees which prompts are cached before spending."""
        for role, prompt in roles:
            if self.explicit_cache:
                self.resolve_explicit_cache(prompt)
            with self._cache_lock:
                entry = (role, self._cache_key(prompt))
                if entry not in self._preflight_roles:
                    self._preflight_roles.append(entry)
        return self.explicit_cache_prompts

    @property
    def explicit_cache_prompts(self) -> list[dict]:
        """The per-prompt decisions as they stand now (derived, so a mid-run recreate shows)."""
        with self._cache_lock:
            out = []
            for role, key in self._preflight_roles:
                decided = self._caches.get(key) or {}
                out.append(
                    {
                        "role": role,
                        "status": decided.get("status", CACHE_STATUS_INLINE),
                        "tokens": decided.get("tokens", 0),
                        "name": decided.get("name"),
                        "reason": decided.get("reason"),
                    }
                )
            return out

    @property
    def explicit_cache_roles(self) -> dict[str, str]:
        """The preflight role (`extract-facts`, `reconcile`, ...) of every cache resource this client
        created, superseded ones included, keyed by resource name."""
        with self._cache_lock:
            role_of = {key: role for role, key in self._preflight_roles}
            return {e["name"]: role_of[e["key"]] for e in self._created if e.get("key") in role_of}

    @property
    def explicit_caches(self) -> list[dict]:
        """Every cache resource this client created, superseded ones included, with how long each
        existed (capped at the TTL): the billable quantity."""
        now = time.monotonic()
        with self._cache_lock:
            return [
                {
                    "name": e["name"],
                    "tokens": e["tokens"],
                    "lifetime_s": round(self._bounded_lifetime(e, now), 3),
                    "live": e["ended_at"] is None,
                }
                for e in self._created
            ]

    def close(self) -> None:
        """Delete every cache this client created. Failures are reported and swallowed: a cache
        expires on its own within its TTL, so a failed delete is a bounded overspend."""
        with self._cache_lock:
            stamp = time.monotonic()
            for entry in self._created:
                if entry["ended_at"] is None:
                    entry["ended_at"] = stamp
            entries = [e for e in self._created if not e["gone"]]
            for entry in entries:
                entry["gone"] = True
            self._caches = {}
        for entry in entries:
            try:
                self._api_call(entry["name"], method="DELETE")
            except Exception as e:
                print(
                    f"warning: could not delete {entry['name']} ({e}); it expires in at most "
                    f"{GEMINI_EXPLICIT_CACHE_TTL_S}s",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------ #
    # generateContent
    # ------------------------------------------------------------------ #
    def extract(
        self, system_prompt: str, conversation_text: str, route: CallRoute | None = None
    ) -> ExtractResult:
        """One call with retries. Network drops and HTTP 429/5xx are retried with exponential
        backoff and full jitter, honoring a server-suggested wait. A timeout, a safety block and any
        other 4xx are terminal: deterministic for a given input. The jitter is seeded from the
        request itself, so a request's backoff never depends on other threads."""
        jitter_seed = hashlib.sha256((system_prompt + "\x00" + conversation_text).encode("utf-8")).hexdigest()
        http_retries = 0
        cache_reset_used = False
        call_route = route
        while True:
            if call_route is None:
                call_route = self.call_route(system_prompt)
            used_cache = call_route.cache_name
            last, transient, retry_after = self._extract_once(
                system_prompt, conversation_text, cache_name=used_cache
            )
            last.meta["http_retries"] = http_retries
            last.meta["cache_route"] = call_route.route
            last.meta["cached_content_name"] = used_cache
            last.meta["cache_model_id"] = call_route.cache_model_id
            if last.ok:
                if http_retries:
                    last.meta["retries"] = http_retries
                return last
            # A referenced cache that expired or was evicted comes back as a plain 4xx. Recoverable
            # once: drop that resource's decision and re-resolve the route, which can turn inline.
            if used_cache and not cache_reset_used and _is_cache_missing_error(last.error):
                cache_reset_used = True
                self.invalidate_explicit_cache(system_prompt, used_name=used_cache)
                call_route = self.call_route(system_prompt)
                continue
            if not transient or http_retries + 1 >= self.max_attempts:
                return last
            delay = self._backoff_delay(http_retries + 1, retry_after, jitter_seed)
            http_retries += 1
            last.meta["http_retries"] = http_retries
            time.sleep(delay)

    @staticmethod
    def _backoff_delay(attempt: int, retry_after: float | None, jitter_seed: str = "") -> float:
        """Seconds to wait before the next attempt: a server-supplied wait wins (capped), else
        2^attempt capped, with full jitter from a request-local, deterministic RNG."""
        if retry_after is not None:
            return min(max(0.0, retry_after), _GEMINI_RETRY_AFTER_CAP_S)
        base = min(2.0**attempt, _GEMINI_RETRY_AFTER_CAP_S)
        rng = random.Random(f"{jitter_seed}#attempt={attempt}")
        return rng.uniform(0.0, base)

    def _extract_once(
        self, system_prompt: str, conversation_text: str, cache_name: str | None = None
    ) -> tuple[ExtractResult, bool, float | None]:
        """One HTTP call. Returns (result, transient, retry_after)."""
        gen_config: dict = {"temperature": 0, "responseMimeType": "application/json"}
        if self.thinking_level is not None:
            gen_config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        if self.response_schema is not None:
            gen_config["responseSchema"] = self.response_schema
        body = json.dumps(
            gemini_request_body(system_prompt, conversation_text, gen_config, cache_name)
        ).encode("utf-8")
        url = f"{self.api_base}/{self.model_id}:generateContent"
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            dur = int((time.time() - t0) * 1000)
            detail = ""
            with e, contextlib.suppress(Exception):
                detail = e.read().decode("utf-8", errors="replace")[:2000]
            transient = e.code == 429 or 500 <= e.code < 600
            retry_after = None
            if transient:
                hdr = None
                try:
                    hdr = e.headers.get("Retry-After") if e.headers else None
                except Exception:
                    hdr = None
                retry_after = _parse_retry_after_header(hdr)
                if retry_after is None:
                    retry_after = _retry_delay_from_error_body(detail)
            return (
                ExtractResult("", 0.0, dur, self.model_id, ok=False, error=f"http {e.code}: {detail}"),
                transient,
                retry_after,
            )
        except TimeoutError:
            return (
                ExtractResult(
                    "", 0.0, int((time.time() - t0) * 1000), self.model_id, ok=False, error="timeout"
                ),
                False,
                None,
            )
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                return (
                    ExtractResult(
                        "", 0.0, int((time.time() - t0) * 1000), self.model_id, ok=False, error="timeout"
                    ),
                    False,
                    None,
                )
            return (
                ExtractResult(
                    "",
                    0.0,
                    int((time.time() - t0) * 1000),
                    self.model_id,
                    ok=False,
                    error=f"network: {reason}",
                ),
                True,
                None,
            )
        except Exception as e:  # unexpected transport failure: worth a retry
            return (
                ExtractResult(
                    "", 0.0, int((time.time() - t0) * 1000), self.model_id, ok=False, error=f"transport: {e}"
                ),
                True,
                None,
            )
        dur = int((time.time() - t0) * 1000)
        try:
            d = json.loads(raw)
        except Exception as e:
            return (
                ExtractResult(
                    "", 0.0, dur, self.model_id, ok=False, error=f"api-json-parse: {e}: {raw[:500]}"
                ),
                False,
                None,
            )
        # A 200 body can still be shaped wrong; any surprise degrades to a terminal error.
        try:
            return self._parse_ok_body(d, dur)
        except Exception as e:
            return (
                ExtractResult("", 0.0, dur, self.model_id, ok=False, error=f"unexpected-response-shape: {e}"),
                False,
                None,
            )

    def _parse_ok_body(self, d: dict, dur: int) -> tuple[ExtractResult, bool, float | None]:
        pf = d.get("promptFeedback") or {}
        if pf.get("blockReason"):
            return (
                ExtractResult(
                    "",
                    0.0,
                    dur,
                    self.model_id,
                    ok=False,
                    error=f"prompt-blocked: {pf.get('blockReason')}",
                    meta={"prompt_feedback": pf},
                ),
                False,
                None,
            )
        candidates = d.get("candidates") or []
        usage = d.get("usageMetadata") or {}
        meta = {"usage": usage, "canonical_model": d.get("modelVersion")}
        if not candidates:
            return (
                ExtractResult("", 0.0, dur, self.model_id, ok=False, error="no-candidates", meta=meta),
                False,
                None,
            )
        cand = candidates[0]
        finish = cand.get("finishReason")
        parts = ((cand.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        meta["finish_reason"] = finish
        if finish is not None and str(finish).upper() != "STOP":
            # Cut short (length, safety, recitation, any other stop): the text is incomplete. The
            # cut is deterministic for this input, so it is terminal, and the billed cost is kept.
            reason = "safety-blocked" if str(finish).upper() in _GEMINI_BLOCK_FINISH else "incomplete"
            return (
                ExtractResult(
                    "",
                    gemini_call_cost(usage),
                    dur,
                    self.model_id,
                    ok=False,
                    error=f"{reason}: finish={finish}",
                    meta=meta,
                ),
                False,
                None,
            )
        if not text.strip():
            # A complete answer with no text: a terminal error, with the tokens it billed.
            return (
                ExtractResult(
                    "",
                    gemini_call_cost(usage),
                    dur,
                    self.model_id,
                    ok=False,
                    error=f"empty-candidate: finish={finish}",
                    meta=meta,
                ),
                False,
                None,
            )
        cost = gemini_call_cost(usage)
        return (
            ExtractResult(
                raw_text=text, cost_usd=cost, duration_ms=dur, model_id=self.model_id, ok=True, meta=meta
            ),
            False,
            None,
        )
