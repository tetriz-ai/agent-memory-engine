# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Model providers: the ModelClient interface and the two built-in backends."""

from __future__ import annotations

from .base import (
    CACHE_STATUS_CACHED,
    CACHE_STATUS_INLINE,
    USAGE_METRIC_KEYS,
    CallRoute,
    ExtractResult,
    ModelClient,
    extract_with_route,
    fold_usage,
    normalize_usage,
    resolve_call_route,
    storage_cache_model_id,
    usage_fields,
)
from .claude_cli import ClaudeCLIClient, ClaudeNotFoundError
from .gemini import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_THINKING_LEVEL,
    GEMINI_KEY_ENV,
    GeminiClient,
    GeminiKeyError,
    cost_from_cache_record,
    gemini_cache_storage_cost,
    gemini_schema_digest,
    load_gemini_key,
)

PROVIDERS = ("gemini", "claude")
DEFAULT_PROVIDER = "gemini"


def build_client(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    thinking_level: str | None = DEFAULT_THINKING_LEVEL,
    response_schema: dict | None = None,
    explicit_cache: bool = True,
) -> ModelClient:
    """A client for `provider`. `model` None picks the provider's pinned default. The thinking
    level, responseSchema and explicit cache are Gemini settings; the caller refuses them for
    claude before calling this (the CLI has no equivalent)."""
    if provider == "gemini":
        if not api_key:
            raise GeminiKeyError(f"no Gemini API key: set {GEMINI_KEY_ENV}, or pass --gemini-key-file PATH")
        return GeminiClient(
            api_key=api_key,
            model_id=model or GeminiClient.DEFAULT_MODEL,
            thinking_level=thinking_level,
            response_schema=response_schema,
            explicit_cache=explicit_cache,
        )
    if provider == "claude":
        return ClaudeCLIClient(model_id=model or ClaudeCLIClient.DEFAULT_MODEL)
    raise ValueError(f"unknown provider: {provider!r}")


__all__ = [
    "CACHE_STATUS_CACHED",
    "CACHE_STATUS_INLINE",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_THINKING_LEVEL",
    "GEMINI_KEY_ENV",
    "PROVIDERS",
    "USAGE_METRIC_KEYS",
    "CallRoute",
    "ClaudeCLIClient",
    "ClaudeNotFoundError",
    "ExtractResult",
    "GeminiClient",
    "GeminiKeyError",
    "ModelClient",
    "build_client",
    "cost_from_cache_record",
    "extract_with_route",
    "fold_usage",
    "gemini_cache_storage_cost",
    "gemini_schema_digest",
    "load_gemini_key",
    "normalize_usage",
    "resolve_call_route",
    "storage_cache_model_id",
    "usage_fields",
]
