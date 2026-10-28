# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The provider seam: one method, extract(system_prompt, text) -> ExtractResult.

Every pipeline stage is "system prompt + text in, JSON text out", so a backend only has to turn
those two strings into a model answer and report what it cost. The two built-in backends are
GeminiClient (REST, the default) and ClaudeCLIClient (the local `claude -p` login). Anything else
subclasses ModelClient.

This module also owns the token accounting shared by every stage: how each provider names its
token counters, and how a call's usage folds into a run's totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CACHE_STATUS_CACHED = "cached"
CACHE_STATUS_INLINE = "inline"


@dataclass
class ExtractResult:
    raw_text: str  # the model's textual response (expected: JSON)
    cost_usd: float
    duration_ms: int
    model_id: str  # immutable id actually used
    ok: bool
    error: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CallRoute:
    """How one call travels, resolved once and then immutable.

    The explicit prompt cache is decided per prompt and can be re-decided mid-run (a recovery whose
    recreate fails routes a prompt inline). The local result cache and the request must agree on
    the route, so it is resolved once and handed to both.
    """

    route: str  # CACHE_STATUS_CACHED | CACHE_STATUS_INLINE
    cache_name: str | None  # the provider-side cache resource, when routed by reference
    cache_model_id: str  # the local-cache identity for this route


class ModelClient:
    """Interface every backend implements. `model_id` must be an immutable version id.

    `cache_model_id` is the id the local result cache keys on. It defaults to `model_id`, but a
    backend whose output varies with a knob that the key (model + system prompt + text) does not
    already cover must fold that knob in, or two settings would replay each other's answers.
    """

    model_id: str = ""
    cache_model_id: str = ""

    # Backends that accept a pre-resolved CallRoute set this True; see extract_with_route.
    accepts_call_route = False

    def call_route(self, system_prompt: str) -> CallRoute:
        """How a call carrying this prompt travels. One route for most backends."""
        return CallRoute(CACHE_STATUS_INLINE, None, self.cache_model_id or self.model_id)

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        raise NotImplementedError

    def close(self) -> None:
        """Release anything the backend holds for the run. A no-op by default."""


def resolve_call_route(client, system_prompt: str) -> CallRoute:
    """The route for one call, asked of any client, including duck-typed ones without call_route."""
    resolver = getattr(client, "call_route", None)
    if resolver is not None:
        return resolver(system_prompt)
    return CallRoute(CACHE_STATUS_INLINE, None, client.cache_model_id)


def extract_with_route(client, system_prompt: str, conversation_text: str, route: CallRoute):
    """Run one call on an already-resolved route, so the request cannot take a different one than
    the cache key was built from. Backends without route support are called plainly."""
    if getattr(client, "accepts_call_route", False):
        return client.extract(system_prompt, conversation_text, route=route)
    return client.extract(system_prompt, conversation_text)


def storage_cache_model_id(result, route: CallRoute) -> str:
    """The identity a result is stored under: the route the request actually used, which a
    recovery can have changed from the one it was looked up under."""
    return result.meta.get("cache_model_id") or route.cache_model_id


# --------------------------------------------------------------------------- #
# Per-call token accounting
# --------------------------------------------------------------------------- #
# Stamped into every cache record, so a reader can tell which arithmetic priced an entry:
# thinking billed as output, provider-cached prompt tokens billed at the cached rate.
COST_FORMULA = "v2-split"

# The additive token counters, and the call counter, kept separate so a caller summing "the tokens"
# cannot pick up a call count. USAGE_METRIC_KEYS is the union, for declaring or folding the set.
USAGE_TOKEN_KEYS = (
    "prompt_tokens_total",
    "answer_tokens_total",
    "thinking_tokens_total",
    "cached_tokens_total",
    "cache_creation_tokens_total",
)
USAGE_CALL_KEYS = ("usage_missing_calls",)
USAGE_METRIC_KEYS = USAGE_TOKEN_KEYS + USAGE_CALL_KEYS

# How each provider names its token counters, and whether its cached-prompt count is inside the
# prompt count or on top of it:
#   Gemini generateContent: cachedContentTokenCount is a subset of promptTokenCount.
#   Claude CLI: input_tokens excludes both cache reads and cache writes, so the real prompt is
#   input_tokens + cache_read_input_tokens + cache_creation_input_tokens.
_USAGE_SHAPES: tuple[dict[str, Any], ...] = (
    {
        "family": "gemini",
        "prompt": "promptTokenCount",
        "answer": "candidatesTokenCount",
        "thinking": "thoughtsTokenCount",
        "cached": "cachedContentTokenCount",
        "cache_creation": None,
        "cached_is_additive": False,
    },
    {
        "family": "claude-cli",
        "prompt": "input_tokens",
        "answer": "output_tokens",
        "thinking": None,
        "cached": "cache_read_input_tokens",
        "cache_creation": "cache_creation_input_tokens",
        "cached_is_additive": True,
    },
)


def _token_count(usage: dict, key: str | None) -> int | None:
    """One counter, or None if it is present but not a number. An unexpected shape reads as
    unreadable, never as zero, so a run's token totals cannot shrink quietly."""
    if key is None or key not in usage:
        return 0
    v = usage[key]
    if v is None:
        return 0
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return int(v)


def normalize_usage(usage) -> dict | None:
    """A provider's raw usage block -> {family, prompt, answer, thinking, cached, cache_creation},
    where `prompt` is always the total prompt tokens. None means unreadable: no block, an unknown
    shape, or a counter that is not a number. Callers count None (usage_missing_calls)."""
    if not isinstance(usage, dict):
        return None
    for shape in _USAGE_SHAPES:
        if shape["prompt"] not in usage:
            continue
        counts = {
            name: _token_count(usage, shape[name])
            for name in ("prompt", "answer", "thinking", "cached", "cache_creation")
        }
        if any(v is None for v in counts.values()):
            return None
        if shape["cached_is_additive"]:
            counts["prompt"] += counts["cached"] + counts["cache_creation"]  # type: ignore[operator]
        return {"family": shape["family"], **counts}
    return None


def fold_usage(acc: dict, usage) -> None:
    """Add one call's token counts into an accumulator that already declares USAGE_METRIC_KEYS.
    Strict indexing on purpose: an undeclared key fails here instead of starting a partial total."""
    n = normalize_usage(usage)
    if n is None:
        acc["usage_missing_calls"] += 1
        return
    acc["prompt_tokens_total"] += n["prompt"]
    acc["answer_tokens_total"] += n["answer"]
    acc["thinking_tokens_total"] += n["thinking"]
    acc["cached_tokens_total"] += n["cached"]
    acc["cache_creation_tokens_total"] += n["cache_creation"]


def usage_fields(result: ExtractResult) -> dict:
    """The provenance fields every cache record carries beside its raw text and cost: the
    provider's token block verbatim, the pricing formula, the retry count and the route taken."""
    return {
        "usage": result.meta.get("usage"),
        "cost_formula": COST_FORMULA,
        "http_retries": int(result.meta.get("http_retries") or 0),
        "cache_route": result.meta.get("cache_route", CACHE_STATUS_INLINE),
    }
