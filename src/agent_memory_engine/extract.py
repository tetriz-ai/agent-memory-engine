# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""One extraction arm over a set of sessions: windows, per-window extraction, reconcile.

A conversation is read as a sequence of turn-aware windows instead of in one shot, so that nothing
lands in the middle of a long context and each window gets a small, fully attended one:

  Stage 1, windowing (deterministic, no model). Greedily pack consecutive turns into windows under
    WINDOW_BUDGET characters; never split a turn; prefer to start a new window at a user turn once the
    current one is near full (a natural task boundary). Real turn indices are kept on every window,
    so the evidence gate checks quotes at their true position.
  Stage 2, per-window extraction (one cached model call per window), run in order with a short
    "already extracted" context block. Each window's output is validated and evidence-gated against
    that window's turns, the exact text the model read.
  Stage 3, within-conversation reconcile (one cached model call per conversation). The model reads
    all of a conversation's window candidates oldest first and collapses duplicates or drops
    superseded ones (recency wins). It never sees transcripts or evidence; code unions the surviving
    candidates' already-verified evidence.

There are two arms, each with its own prompt: `facts` (facts, decisions, cautions, failed attempts,
intentions) and `preferences` (preferences and conventions). Every model call is cached on its exact
inputs, so a re-run over the same sessions costs nothing.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from importlib import resources

from . import cache as cache_mod
from . import contract as mc
from . import transcript
from .paths import run_lock
from .providers import (
    CACHE_STATUS_CACHED,
    USAGE_METRIC_KEYS,
    ModelClient,
    cost_from_cache_record,
    extract_with_route,
    fold_usage,
    gemini_cache_storage_cost,
    gemini_schema_digest,
    resolve_call_route,
    storage_cache_model_id,
    usage_fields,
)


@dataclass(frozen=True)
class Arm:
    """One extraction arm. `prompt_version` is the label the local result cache keys on. It is pinned
    here, never derived from the prompt file name, so renaming a file cannot silently change cache
    identity; editing the prompt text still forces fresh calls, because the key hashes the text."""

    name: str
    prompt_file: str
    prompt_version: str


# `+fmt-v2` names the request-bytes version the cache keys cover, not the output format.
ARMS = (
    Arm("facts", "extract_facts.md", "facts-v1+fmt-v2"),
    Arm("preferences", "extract_preferences.md", "preferences-v1+fmt-v2"),
)
ARMS_BY_NAME = {a.name: a for a in ARMS}
RECONCILE_PROMPT_FILE = "reconcile.md"
RECONCILE_PROMPT_VERSION = "chunked-reconcile-v2+fmt-v2"  # request-bytes version, as for ARMS
# The segmentation label inside every cache key; changing it orphans cached answers.
SEGMENTATION = "turn_aware"


# Char budget per window. Below the model's comfortable context, and above the per-turn cap
# (transcript.PER_TURN_MAX), so a single parsed turn always fits a window on its own.
WINDOW_BUDGET = 10_000
# Prefer to close a window at a user turn once it is at least this full.
NEAR_BUDGET_FRAC = 0.6
# Bound on the "context so far" block handed to each window (statements only, no evidence).
CONTEXT_BUDGET = 1_500
# Char cap on the reconcile prompt's concatenated candidates, so a very long conversation cannot
# overflow the model's context.
RECONCILE_BUDGET = 60_000

# Upper bound on conversation-level parallelism. Parallelism comes only from running whole
# conversations at once; each conversation runs the full pipeline sequentially, so output is
# identical at any worker count.
CONV_MAX_WORKERS = 6

# Single source of the vocabulary: contract.GENERALITY_VALUES. Kept as a set here for membership
# tests; importing it (rather than re-listing the literals) stops the two from drifting apart.
VALID_GENERALITY = frozenset(mc.GENERALITY_VALUES)


def load_prompt(fname: str) -> str:
    """The system prompt between the BEGIN/END SYSTEM PROMPT markers of a packaged prompt file."""
    text = resources.files("agent_memory_engine.prompts").joinpath(fname).read_text(encoding="utf-8")
    begin, end = "===BEGIN SYSTEM PROMPT===", "===END SYSTEM PROMPT==="
    b, e = text.find(begin), text.find(end)
    if b < 0 or e < 0 or e < b:
        raise ValueError(f"{fname} must contain '{begin}' before '{end}'")
    return text[b + len(begin) : e].strip()


def load_reconcile_prompt() -> str:
    return load_prompt(RECONCILE_PROMPT_FILE)


# --------------------------------------------------------------------------- #
# Stage 1: windowing (deterministic, no model)
# --------------------------------------------------------------------------- #
def build_windows(turns: list[transcript.Turn], budget: int = WINDOW_BUDGET) -> list[list[transcript.Turn]]:
    """Greedy turn-aware packing. Never splits a turn across windows; a single turn larger than the
    budget becomes its own window (truncated by the per-turn helper). Prefers to start a new window
    at a user turn once the current window is near budget."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"window budget must be a positive int, got {budget!r}")
    near = budget * NEAR_BUDGET_FRAC
    windows: list[list[transcript.Turn]] = []
    cur: list[transcript.Turn] = []
    cur_size = 0
    for t in turns:
        tlen = len(t.content)
        if tlen > budget:
            if cur:
                windows.append(cur)
                cur, cur_size = [], 0
            windows.append([dataclasses.replace(t, content=transcript._truncate_turn(t.content))])
            continue
        if cur and (cur_size + tlen > budget or (t.role == "user" and cur_size >= near)):
            windows.append(cur)
            cur, cur_size = [], 0
        cur.append(t)
        cur_size += tlen
    if cur:
        windows.append(cur)
    return windows


def plan_windows(conv: transcript.Conversation) -> list[list[transcript.Turn]]:
    """The windows a conversation is sent as, after the whole-transcript budget. No model call."""
    return build_windows(transcript.render_transcript(conv).turns)


def _window_conv(
    conv: transcript.Conversation, window_turns: list[transcript.Turn]
) -> transcript.Conversation:
    """A Conversation view over one window's turns (real indices kept). Used both to render the
    window for the model and to evidence-gate the output, so the gate checks the text the model read."""
    return transcript.Conversation(
        conv.conversation_id, conv.tool, conv.source, conv.date, window_turns, truncated=False
    )


def _context_block(statements: list[str], budget: int = CONTEXT_BUDGET) -> str:
    """Statements-only 'context so far' for continuity across windows. Keeps the most recent
    statements that fit the budget."""
    if not statements:
        return ""
    lines: list[str] = []
    total = 0
    for s in reversed(statements):
        line = f"- {s}"
        if total + len(line) + 1 > budget:
            break
        lines.append(line)
        total += len(line) + 1
    if not lines:
        return ""
    lines.reverse()
    return (
        "--- MEMORIES ALREADY EXTRACTED FROM EARLIER WINDOWS OF THIS CONVERSATION "
        "(context only; do not re-emit duplicates) ---\n" + "\n".join(lines)
    )


def build_window_text(
    conv: transcript.Conversation,
    window_turns: list[transcript.Turn],
    window_index: int,
    n_windows: int,
    context_block: str,
) -> str:
    """One window's prompt payload: session metadata, the context block, the window transcript."""
    fields = {
        "tool": conv.tool,
        "session_date": conv.date,
        "window": window_index + 1,
        "of": n_windows,
        "window_turns": [t.i for t in window_turns],
    }
    header = json.dumps(fields, ensure_ascii=False)
    parts = ["--- SESSION METADATA ---", header]
    if context_block:
        parts += ["", context_block]
    parts += ["", transcript.transcript_text(_window_conv(conv, window_turns))]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 3: reconcile text + assembly
# --------------------------------------------------------------------------- #
def build_reconcile_text(
    conv: transcript.Conversation, candidates: list[dict], budget: int = RECONCILE_BUDGET
) -> tuple[str, list[dict]]:
    """The reconcile prompt payload and the candidates it holds. The most recent candidates that fit
    `budget` are sent; older ones past it are left out, and the caller passes them through
    unreconciled rather than letting the model's silence delete them."""
    blocks = []
    for c in candidates:
        b = [
            f"[{c['label']}] (window {c['window_index'] + 1}) type={c['type']} "
            f"scope_level={c['scope_level']} scope_ref={c.get('scope_ref')} "
            f"generality={c['generality']} disposition={c['disposition']}",
            f"    statement: {c['statement']}",
            f"    tags: {c['tags_line']}",
        ]
        if c.get("secondary_type"):
            b.append(f"    secondary_type: {c['secondary_type']}")
        if c.get("scope_applicable_level"):
            b.append(f"    scope_applicable: {c['scope_applicable_level']} {c.get('scope_applicable_ref')}")
        if c.get("decision_outcome"):
            b.append(f"    decision_outcome: {c['decision_outcome']}")
        if c.get("intention_meta"):
            im = c["intention_meta"]
            b.append(f"    intention: what={im.get('what')} done_check={im.get('done_check')}")
        blocks.append("\n".join(b))
    # Keep the most recent candidates that fit: recency wins in reconcile.
    kept: list[str] = []
    total = 0
    for block in reversed(blocks):
        if kept and total + len(block) + 1 > budget:
            break
        kept.append(block)
        total += len(block) + 1
    kept.reverse()
    sent = candidates[len(candidates) - len(kept) :]
    header = json.dumps(
        {
            "tool": conv.tool,
            "session_date": conv.date,
            "conversation_id": conv.conversation_id,
            "n_candidates": len(kept),
        },
        ensure_ascii=False,
    )
    body = "--- CANDIDATE MEMORIES (this conversation, oldest window first) ---\n" + "\n".join(kept)
    return f"--- METADATA ---\n{header}\n\n" + body, sent


def _dedupe_evidence(evs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in evs:
        key = (e.get("quote"), e.get("turn"), e.get("speaker"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _union_member_tags(members: list[dict]) -> dict:
    """Union the members' typed tag objects; the first value wins on domain and activity."""
    return mc.union_tags([m.get("tags") for m in members], single_value="first")


def _assemble_reconciled(
    conv: transcript.Conversation, candidates: list[dict], recon_obj: dict, arm: str
) -> tuple[list[dict], dict]:
    """Turn a reconcile response into final memory dicts. The model decides grouping, deletion and
    phrasing; code owns evidence (the union of the members' verified evidence) and provenance. A
    candidate the model neither covered nor deleted is treated as deleted."""
    by_label = {c["label"]: c for c in candidates}
    covered: set[str] = set()
    finals: list[dict] = []
    out_mems = recon_obj.get("memories") if isinstance(recon_obj, dict) else None
    if not isinstance(out_mems, list):
        out_mems = []
    n_merged = 0
    for om in out_mems:
        if not isinstance(om, dict):
            continue
        members = [m for m in (om.get("members") or []) if m in by_label and m not in covered]
        if not members:
            continue
        covered.update(members)
        # Order by recency ourselves: window, then the candidate's emission order within it.
        mem_cands = sorted(
            (by_label[m] for m in members), key=lambda c: (c["window_index"], int(c["label"][1:]))
        )
        latest = mem_cands[-1]  # the ending state is the truth
        mtype = (om.get("type") or "").lower()
        if mtype not in mc.VALID_TYPES:
            mtype = latest["type"]
        # An intention member keeps the group an intention, so it stays held for review with its meta.
        if any(c["type"] == "intention" for c in mem_cands):
            mtype = "intention"
        gen = (om.get("generality") or "").lower()
        if gen not in VALID_GENERALITY:
            gen = latest.get("generality") or "instance-specific"
        stmt = om.get("statement")
        if not isinstance(stmt, str) or not stmt.strip():
            stmt = latest["statement"]
        scope_level, scope_ref = mc.derive_scope(om.get("scope_level"), om.get("scope_ref"), om.get("scope"))
        if scope_level is None:
            scope_level, scope_ref = latest["scope_level"], latest.get("scope_ref")
        # Tags never come from the reconcile model, which would flatten the typed scheme.
        tags = _union_member_tags(mem_cands)
        decision_outcome = (
            next(
                (
                    c.get("decision_outcome")
                    for c in reversed(mem_cands)
                    if c["type"] == "decision" and c.get("decision_outcome")
                ),
                None,
            )
            if mtype == "decision"
            else None
        )
        secondary_type = next(
            (
                c.get("secondary_type")
                for c in reversed(mem_cands)
                if c.get("secondary_type") and c.get("secondary_type") != mtype
            ),
            None,
        )
        app_member = next((c for c in reversed(mem_cands) if c.get("scope_applicable_level")), None)
        scope_app_level = app_member.get("scope_applicable_level") if app_member else None
        scope_app_ref = app_member.get("scope_applicable_ref") if app_member else None
        # A non-null applicable scope always means "applies somewhere other than where observed".
        if scope_app_level == scope_level and scope_app_ref == scope_ref:
            scope_app_level, scope_app_ref = None, None
        intention_meta = (
            next(
                (
                    c.get("intention_meta")
                    for c in reversed(mem_cands)
                    if c["type"] == "intention" and c.get("intention_meta")
                ),
                None,
            )
            if mtype == "intention"
            else None
        )
        review = mc._review_for(mtype)
        disposition = "confident" if all(c["disposition"] == "confident" for c in mem_cands) else "borderline"
        # The reason travels with a borderline memory: the first borderline member's note.
        disposition_note = next(
            (
                c.get("disposition_note")
                for c in mem_cands
                if c["disposition"] == "borderline" and c.get("disposition_note")
            ),
            None,
        )
        evidence = _dedupe_evidence([e for c in mem_cands for e in c["evidence"]])
        windows = sorted({c["window_index"] + 1 for c in mem_cands})
        turns = sorted({e["turn"] for e in evidence})
        if len(members) > 1:
            n_merged += 1
        finals.append(
            _final_memory(
                conv,
                arm,
                stmt.strip(),
                mtype,
                scope_level,
                scope_ref,
                gen,
                tags,
                decision_outcome,
                disposition,
                evidence,
                windows,
                turns,
                intention_meta,
                review,
                latest.get("valid_as_of"),
                latest.get("temporal_kind"),
                latest.get("volatility"),
                latest.get("description"),
                secondary_type,
                scope_app_level,
                scope_app_ref,
                disposition_note=disposition_note,
            )
        )
    explicit_deleted = [d for d in (recon_obj.get("deleted") or []) if d in by_label]
    deleted = set(explicit_deleted) | (set(by_label) - covered)
    stats = {"n_merged": n_merged, "n_deleted": len(deleted), "deleted_labels": sorted(deleted)}
    return finals, stats


def _final_memory(
    conv: transcript.Conversation,
    arm: str,
    statement: str,
    mtype: str,
    scope_level: str,
    scope_ref: str | None,
    generality: str,
    tags: dict,
    decision_outcome,
    disposition: str,
    evidence: list[dict],
    windows: list[int],
    turns: list[int],
    intention_meta: dict | None = None,
    review: dict | None = None,
    valid_as_of: str | None = None,
    temporal_kind: str | None = None,
    volatility: str | None = None,
    description: str | None = None,
    secondary_type: str | None = None,
    scope_applicable_level: str | None = None,
    scope_applicable_ref: str | None = None,
    disposition_note: str | None = None,
) -> dict:
    # Each evidence item carries its conversation and the quote's effective authorship, so the merge
    # can recompute provenance confidence from the final evidence set.
    ev_out = [
        {
            "quote": e["quote"],
            "turn": e["turn"],
            "speaker": e["speaker"],
            "conversation_id": conv.conversation_id,
            "authorship": e.get("authorship"),
        }
        for e in evidence
    ]
    scope_disp = scope_level if (scope_level == "personal" or not scope_ref) else f"{scope_level}/{scope_ref}"
    return {
        "memory_id": mc.memory_id(statement),
        "statement": statement,
        "type": mtype,
        "secondary_type": secondary_type,
        "scope_level": scope_level,
        "scope_ref": scope_ref,
        "scope": scope_disp,
        "scope_applicable_level": scope_applicable_level,
        "scope_applicable_ref": scope_applicable_ref,
        "generality": generality,
        "origin": {
            "conversation_id": conv.conversation_id,
            "tool": conv.tool,
            "arm": arm,
            "windows": windows,
            "source_turns": turns,
        },
        "when": conv.date,
        "evidence": ev_out,
        "decision_outcome": decision_outcome,
        "tags": tags,
        "disposition": disposition,
        "disposition_note": disposition_note if disposition == "borderline" else None,
        "intention_meta": intention_meta,
        "review": review or mc._review_for(mtype),
        "valid_as_of": valid_as_of,
        "temporal_kind": temporal_kind,
        "volatility": volatility,
        "description": description,
    }


def _candidates_to_finals_noreconcile(
    conv: transcript.Conversation, candidates: list[dict], arm: str
) -> list[dict]:
    """For a conversation with fewer than two candidates there is nothing to reconcile: ship the
    candidates, folding only exact restatements (same normalized statement)."""
    seen: dict[str, dict] = {}
    order: list[str] = []
    for c in candidates:
        mid = mc.memory_id(c["statement"])
        if mid in seen:
            seen[mid]["_ev"] = _dedupe_evidence(seen[mid]["_ev"] + c["evidence"])
            seen[mid]["_windows"].add(c["window_index"] + 1)
            seen[mid]["_dupes"].append(c)
            continue
        seen[mid] = {"c": c, "_ev": list(c["evidence"]), "_windows": {c["window_index"] + 1}, "_dupes": [c]}
        order.append(mid)
    finals = []
    for mid in order:
        rec = seen[mid]
        c = rec["c"]
        ev = _dedupe_evidence(rec["_ev"])
        tags = _union_member_tags(rec["_dupes"])
        # An intention restated as another type keeps the folded memory an intention; the latest
        # intention with meta gives it its meta.
        intents = [d for d in rec["_dupes"] if d["type"] == "intention"]
        mtype = "intention" if intents else c["type"]
        intention_meta = next(
            (d["intention_meta"] for d in reversed(intents) if d.get("intention_meta")), None
        )
        finals.append(
            _final_memory(
                conv,
                arm,
                c["statement"],
                mtype,
                c["scope_level"],
                c.get("scope_ref"),
                c["generality"],
                tags,
                c.get("decision_outcome") if mtype == "decision" else None,
                c["disposition"],
                ev,
                sorted(rec["_windows"]),
                sorted({e["turn"] for e in ev}),
                intention_meta,
                mc._review_for(mtype),
                c.get("valid_as_of"),
                c.get("temporal_kind"),
                c.get("volatility"),
                c.get("description"),
                c.get("secondary_type") if c.get("secondary_type") != mtype else None,
                c.get("scope_applicable_level"),
                c.get("scope_applicable_ref"),
                disposition_note=c.get("disposition_note"),
            )
        )
    return finals


# --------------------------------------------------------------------------- #
def _cached_call(cache, client, system_prompt, prompt_version, cache_id, user_text, account=None):
    """One (possibly cached) model call. Returns (res_dict, was_cache_hit) or (None, error_str) on a
    model failure the client already retried.

    `account(res, was_hit)` runs the instant a result exists and before any local work that could
    fail (persisting it, parsing it): the provider has billed by then, so a cache write that fails
    must not erase the spend."""
    route = resolve_call_route(client, system_prompt)
    fp = cache_mod.fingerprint(cache_id, prompt_version, route.cache_model_id, system_prompt, user_text)
    cached = cache.get(fp)
    if cached is not None:
        if account is not None:
            account(cached, True)
        return cached, True
    r = extract_with_route(client, system_prompt, user_text, route)
    if not r.ok:
        # A failed call may still have been billed: account it, and cache nothing.
        if account is not None:
            account({"cost_usd": r.cost_usd, "duration_ms": r.duration_ms, **usage_fields(r)}, False)
        return None, r.error
    # A recovery can re-decide the route mid-call; store under the route actually used.
    store_id = storage_cache_model_id(r, route)
    if store_id != route.cache_model_id:
        fp = cache_mod.fingerprint(cache_id, prompt_version, store_id, system_prompt, user_text)
    res = {
        "raw_text": r.raw_text,
        "cost_usd": r.cost_usd,
        "duration_ms": r.duration_ms,
        "model_id": r.model_id,
        "canonical_model": r.meta.get("canonical_model"),
        **usage_fields(r),
    }
    if account is not None:
        account(res, False)
    cache.put(fp, res)
    return res, False


def _process_window_result(
    conv: transcript.Conversation,
    wi: int,
    wturns: list[transcript.Turn],
    res: dict,
    client: ModelClient,
    metrics: dict,
    candidates: list[dict],
    running_statements: list[str],
) -> None:
    """Validate and evidence-gate one window's output, append the surviving memories as candidates
    (labels in window/call order), and extend running_statements."""
    obj = mc.extract_json(res["raw_text"])
    # An answer with no readable memories list (truncated JSON, prose) yields nothing; count it so a
    # run cannot read as "nothing to remember" when the model's answer was simply unusable.
    if not (isinstance(obj, dict) and isinstance(obj.get("memories"), list)):
        metrics["windows_unparseable"] += 1
    rep = mc.validate_and_gate(obj or {}, _window_conv(conv, wturns), client.model_id)
    metrics["dropped_schema"] += rep.n_dropped_schema
    metrics["dropped_evidence"] += rep.n_dropped_evidence
    # How each kept memory's tags arrived, so tag non-compliance stays visible in the run's metrics.
    metrics["tags_object"] += rep.n_tags_object
    metrics["tags_list"] += rep.n_tags_list
    metrics["tags_empty"] += rep.n_tags_empty
    metrics["tags_malformed"] += rep.n_tags_malformed
    metrics["unknown_tag_keys"] += rep.n_unknown_tag_keys
    # The gate does not carry generality; re-read it from the parsed object by normalized statement.
    gen_by_stmt = _generality_index(obj)
    for gm in rep.memories:
        gen = gen_by_stmt.get(mc.norm_statement(gm.statement), "instance-specific")
        candidates.append(
            {
                "label": f"C{len(candidates) + 1}",
                "window_index": wi,
                "statement": gm.statement,
                "type": gm.type,
                "secondary_type": gm.secondary_type,
                "scope_level": gm.scope_level,
                "scope_ref": gm.scope_ref,
                "scope_applicable_level": gm.scope_applicable_level,
                "scope_applicable_ref": gm.scope_applicable_ref,
                "generality": gen,
                "tags": gm.tags,
                "tags_line": gm.tags_line,
                "decision_outcome": gm.decision_outcome,
                "disposition": gm.disposition,
                "disposition_note": mc._clean_free_str(gm.disposition_note),
                "intention_meta": gm.intention_meta,
                "review": gm.review,
                "valid_as_of": gm.valid_as_of,
                "temporal_kind": gm.temporal_kind,
                "volatility": gm.volatility,
                "description": gm.description,
                "evidence": [dict(e) for e in gm.evidence],
            }
        )
        running_statements.append(gm.statement)


def _fresh_acc() -> dict:
    """A per-conversation accumulator for the additive counters. One factory, so a counter cannot be
    added to the happy path and forgotten on the exception path."""
    acc = {
        "dropped_schema": 0,
        "dropped_evidence": 0,
        "windows_unparseable": 0,
        "reconcile_unparseable": 0,
        "reconcile_overflow": 0,
        "tags_object": 0,
        "tags_list": 0,
        "tags_empty": 0,
        "tags_malformed": 0,
        "unknown_tag_keys": 0,
        "model_duration_ms_total": 0,
        "canonical_model": None,
        "http_retries": 0,
    }
    acc.update({k: 0 for k in USAGE_METRIC_KEYS})
    return acc


@dataclass
class _ConvResult:
    """Everything one conversation produced, from conversation-local accumulators only, so it is safe
    to compute concurrently. The caller folds these in selection order."""

    conv: transcript.Conversation
    windows: int
    candidates_pre: int
    finals: list[dict]
    recon_stats: dict
    acc: dict
    conv_cost: float
    conv_cache_hits: int
    conv_llm_calls: int
    fatal: dict | None = None
    conv_cached_value: float = 0.0  # the part of conv_cost replayed from the cache, not spent


def _process_conversation(
    conv: transcript.Conversation,
    arm: str,
    extract_prompt: str,
    extract_prompt_version: str,
    reconcile_prompt: str,
    cache,
    client: ModelClient,
    reconcile_client: ModelClient | None = None,
) -> _ConvResult:
    """The full per-conversation pipeline: render -> windows -> per-window extraction in order with
    the running context block -> reconcile. Touches nothing shared but the cache (atomic per-key
    writes) and the client, so it is safe under a conversation-level thread pool."""
    conv = transcript.render_transcript(conv)
    windows = build_windows(conv.turns)
    n_windows = len(windows)
    candidates: list[dict] = []
    running_statements: list[str] = []
    acc = _fresh_acc()
    conv_cost = 0.0
    conv_cached_value = 0.0
    conv_cache_hits = 0
    conv_llm_calls = 0
    fatal: dict | None = None

    def _account(res: dict, hit: bool) -> None:
        nonlocal conv_cost, conv_cached_value, conv_cache_hits, conv_llm_calls
        cost = cost_from_cache_record(res)
        if hit:
            conv_cache_hits += 1
            conv_cached_value += cost
        else:
            conv_llm_calls += 1
        # A cache hit replays the saved call's cost and usage: the metric reports what extracting
        # this arm costs, not how warm the cache happened to be. conv_cached_value keeps the replayed
        # part apart, so what this run actually spent is conv_cost minus it.
        conv_cost += cost
        acc["model_duration_ms_total"] += res.get("duration_ms", 0)
        acc["http_retries"] += int(res.get("http_retries") or 0)
        fold_usage(acc, res.get("usage"))
        if res.get("canonical_model"):
            acc["canonical_model"] = res["canonical_model"]

    conv_pre = 0
    recon_stats = {"n_merged": 0, "n_deleted": 0, "deleted_labels": []}
    finals: list[dict] = []
    # Stages 2 and 3 are where money is spent: an unexpected raise inside them becomes a fatal result
    # with the accounting intact, instead of reaching a guard that can only report zero spend.
    try:
        for wi, wturns in enumerate(windows):
            ctx = _context_block(running_statements)
            wtext = build_window_text(conv, wturns, wi, n_windows, ctx)
            cache_id = f"{conv.conversation_id}#seg={SEGMENTATION}#w{wi}"
            res, hit = _cached_call(
                cache, client, extract_prompt, extract_prompt_version, cache_id, wtext, account=_account
            )
            if res is None:
                fatal = {"stage": "extract", "window": wi, "error": hit}
                break
            _process_window_result(conv, wi, wturns, res, client, acc, candidates, running_statements)

        conv_pre = len(candidates)
        if fatal is None:
            if conv_pre >= 2:
                rtext, sent = build_reconcile_text(conv, candidates)
                overflow = candidates[: len(candidates) - len(sent)]
                acc["reconcile_overflow"] += len(overflow)
                cache_id = f"{conv.conversation_id}#seg={SEGMENTATION}#reconcile"
                # Reconcile runs through a schema-less client: the extraction responseSchema does not
                # express reconcile's {members, deleted} shape, and constraining the call would force
                # the extraction shape, yield no members, and silently delete every memory.
                res, hit = _cached_call(
                    cache,
                    reconcile_client or client,
                    reconcile_prompt,
                    RECONCILE_PROMPT_VERSION,
                    cache_id,
                    rtext,
                    account=_account,
                )
                if res is None:
                    fatal = {"stage": "reconcile", "error": hit}
                else:
                    recon_obj = mc.extract_json(res["raw_text"])
                    if isinstance(recon_obj, dict) and isinstance(recon_obj.get("memories"), list):
                        finals, recon_stats = _assemble_reconciled(conv, sent, recon_obj, arm)
                    else:
                        # No readable answer: keep every candidate rather than read silence as deletion.
                        acc["reconcile_unparseable"] += 1
                        finals = _candidates_to_finals_noreconcile(conv, sent, arm)
                    # Candidates past the reconcile budget were never shown to the model; they pass
                    # through unreconciled, ahead of the newer ones.
                    if overflow:
                        finals = _candidates_to_finals_noreconcile(conv, overflow, arm) + finals
            else:
                finals = _candidates_to_finals_noreconcile(conv, candidates, arm)
    except Exception as e:
        fatal = fatal or {"stage": "exception", "error": repr(e)}
        conv_pre = len(candidates)
        finals = []

    return _ConvResult(
        conv=conv,
        windows=n_windows,
        candidates_pre=conv_pre,
        finals=finals,
        recon_stats=recon_stats,
        acc=acc,
        conv_cost=conv_cost,
        conv_cache_hits=conv_cache_hits,
        conv_llm_calls=conv_llm_calls,
        fatal=fatal,
        conv_cached_value=conv_cached_value,
    )


def run(
    rows: list[dict],
    out_dir: str,
    client: ModelClient,
    *,
    arm: str,
    cache_dir: str,
    workers: int = 1,
    reconcile_client: ModelClient | None = None,
    log=None,
) -> tuple[dict, bool]:
    """Run one arm over the selected sessions `rows` and write `memories.jsonl` and `metrics.json`
    to `out_dir`. Returns (metrics, ok); ok is False when a model failure stopped the run, in which
    case `metrics["fatal_error"]` says where.

    `rows` are selection rows (sources.claude_code.list_sessions); a row whose file cannot be parsed
    is counted in conversations_skipped. Output order is the selection order at any worker count."""
    if arm not in ARMS_BY_NAME:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS_BY_NAME)}")
    spec = ARMS_BY_NAME[arm]
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    workers = max(1, min(workers, CONV_MAX_WORKERS))
    extract_prompt = load_prompt(spec.prompt_file)
    reconcile_prompt = load_reconcile_prompt()
    cache = cache_mod.Cache(cache_dir)
    n = len(rows)
    # A selection never lists one conversation twice: with workers > 1 two identical rows would both
    # miss the cache, both call the model, and write two memories sets for one conversation.
    seen_cids: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        cid = r.get("conversation_id")
        if isinstance(cid, str) and cid:
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
        deduped.append(r)
    rows = deduped

    memories_path = os.path.join(out_dir, "memories.jsonl")
    with run_lock(out_dir), open(memories_path, "w", encoding="utf-8") as mem_out:
        metrics: dict = {
            "arm": arm,
            "extract_prompt_version": spec.prompt_version,
            "reconcile_prompt_version": RECONCILE_PROMPT_VERSION,
            "model_id": client.model_id,
            "structured_tags": getattr(client, "response_schema", None) is not None,
            "thinking_level": getattr(client, "thinking_level", None),
            "response_schema_digest": (
                gemini_schema_digest(getattr(client, "response_schema", None) or {})
                if getattr(client, "response_schema", None) is not None
                else None
            ),
            "explicit_cache": bool(getattr(client, "explicit_cache", False)),
            # A list: reconcile runs on its own unconstrained client, and each prompt gets its own cache
            # resource, so an arm can hold (and pay storage on) more than one.
            "explicit_caches": [],
            # One record per prompt: cached (with its resource and token count) or inline (with why).
            "explicit_cache_prompts": [],
            "explicit_cache_tokens": 0,
            "explicit_cache_storage_usd": 0.0,
            "http_retries": 0,
            # The rates cost_usd_total was computed at (Gemini only; the claude CLI reports its own cost).
            "price_card": price_card_for(client),
            # Conversations whose calls were paid for but whose memories were not written: the one that
            # hit the fatal error, and any still in flight when the run stopped.
            "partial_conversations": [],
            "window_budget": WINDOW_BUDGET,
            "conversations_requested": n,
            "conversations_run": 0,
            "conversations_skipped": 0,
            "conversations_empty": 0,
            "windows_total": 0,
            "windows_unparseable": 0,
            "reconcile_unparseable": 0,
            "reconcile_overflow": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "candidates_pre_reconcile": 0,
            "memories_post_reconcile": 0,
            "memories_total": 0,
            "memories_confident": 0,
            "memories_borderline": 0,
            "generality_general": 0,
            "generality_instance": 0,
            "reconcile_deleted_total": 0,
            "reconcile_merged_total": 0,
            "dropped_schema": 0,
            "dropped_evidence": 0,
            "tags_object": 0,
            "tags_list": 0,
            "tags_empty": 0,
            "tags_malformed": 0,
            "unknown_tag_keys": 0,
            "cost_usd_total": 0.0,
            "cached_answers_usd": 0.0,
            "model_duration_ms_total": 0,
            "canonical_model": None,
            # Token anatomy of the spend. `usage_missing_calls` counts calls whose usage could not be
            # read: non-zero means the token totals cover only part of the arm.
            **{k: 0 for k in USAGE_METRIC_KEYS},
            "skipped_sessions": [],
            "per_conversation": [],
        }
        wall0 = time.time()

        def _merge_and_write(res: _ConvResult) -> None:
            """Fold one conversation's result into the run metrics and append its memories. Main thread
            only, so the output order is the selection order."""
            conv = res.conv
            metrics["windows_total"] += res.windows
            metrics["dropped_schema"] += res.acc["dropped_schema"]
            metrics["dropped_evidence"] += res.acc["dropped_evidence"]
            for k in (
                "tags_object",
                "tags_list",
                "tags_empty",
                "tags_malformed",
                "unknown_tag_keys",
                "windows_unparseable",
                "reconcile_unparseable",
                "reconcile_overflow",
                "http_retries",
                *USAGE_METRIC_KEYS,
            ):
                metrics[k] += res.acc[k]
            metrics["model_duration_ms_total"] += res.acc["model_duration_ms_total"]
            if res.acc["canonical_model"]:
                metrics["canonical_model"] = res.acc["canonical_model"]
            metrics["candidates_pre_reconcile"] += res.candidates_pre
            metrics["reconcile_deleted_total"] += res.recon_stats["n_deleted"]
            metrics["reconcile_merged_total"] += res.recon_stats["n_merged"]
            metrics["memories_post_reconcile"] += len(res.finals)
            metrics["cost_usd_total"] += res.conv_cost
            metrics["cached_answers_usd"] += res.conv_cached_value
            metrics["cache_hits"] += res.conv_cache_hits
            metrics["llm_calls"] += res.conv_llm_calls
            metrics["conversations_run"] += 1
            if not res.finals:
                metrics["conversations_empty"] += 1
            for m in res.finals:
                metrics["memories_total"] += 1
                if m["disposition"] == "confident":
                    metrics["memories_confident"] += 1
                else:
                    metrics["memories_borderline"] += 1
                if m["generality"] == "general-principle":
                    metrics["generality_general"] += 1
                else:
                    metrics["generality_instance"] += 1
                mem_out.write(json.dumps(m, ensure_ascii=False) + "\n")
            mem_out.flush()
            spent = metrics["cost_usd_total"] - metrics["cached_answers_usd"]
            log(
                f"  {arm}: [{metrics['conversations_run']}/{n}] {conv.conversation_id[:15]} "
                f"windows={res.windows} candidates={res.candidates_pre} -> memories={len(res.finals)} "
                f"calls={res.conv_llm_calls} cached={res.conv_cache_hits} | "
                f"{format_cost(metrics['cost_usd_total'], spent)}"
            )
            metrics["per_conversation"].append(
                {
                    "conversation_id": conv.conversation_id,
                    "tool": conv.tool,
                    "date": conv.date,
                    "n_turns": len(conv.turns),
                    "truncated": conv.truncated,
                    "windows": res.windows,
                    "windows_unparseable": res.acc["windows_unparseable"],
                    "reconcile_unparseable": res.acc["reconcile_unparseable"],
                    "reconcile_overflow": res.acc["reconcile_overflow"],
                    "candidates_pre_reconcile": res.candidates_pre,
                    "memories_post_reconcile": len(res.finals),
                    "reconcile_deleted": res.recon_stats["n_deleted"],
                    "reconcile_merged": res.recon_stats["n_merged"],
                    "llm_calls": res.conv_llm_calls,
                    "cache_hits": res.conv_cache_hits,
                    "cost_usd": round(res.conv_cost, 4),
                }
            )

        def _safe_process(conv: transcript.Conversation) -> _ConvResult:
            """Run one conversation, turning any unexpected exception into a fatal result, so it cannot
            unwind past the ordered fold and drop earlier conversations."""
            try:
                return _process_conversation(
                    conv,
                    arm,
                    extract_prompt,
                    spec.prompt_version,
                    reconcile_prompt,
                    cache,
                    client,
                    reconcile_client,
                )
            except Exception as e:
                return _ConvResult(
                    conv=conv,
                    windows=0,
                    candidates_pre=0,
                    finals=[],
                    recon_stats={"n_merged": 0, "n_deleted": 0, "deleted_labels": []},
                    acc=_fresh_acc(),
                    conv_cost=0.0,
                    conv_cache_hits=0,
                    conv_llm_calls=0,
                    fatal={"stage": "exception", "error": repr(e)},
                )

        def _fold_spend(res: _ConvResult, status: str) -> None:
            """Fold the real spend of a conversation whose memories will not be written: a stopped run
            has still paid for every call it made. Quality counters stay with written conversations."""
            metrics["cost_usd_total"] += res.conv_cost
            metrics["cached_answers_usd"] += res.conv_cached_value
            metrics["llm_calls"] += res.conv_llm_calls
            metrics["cache_hits"] += res.conv_cache_hits
            metrics["model_duration_ms_total"] += res.acc["model_duration_ms_total"]
            for k in ("http_retries", *USAGE_METRIC_KEYS):
                metrics[k] += res.acc[k]
            if res.acc["canonical_model"]:
                metrics["canonical_model"] = res.acc["canonical_model"]
            metrics["partial_conversations"].append(
                {
                    "conversation_id": res.conv.conversation_id,
                    "status": status,
                    "windows_done": res.windows,
                    "calls": res.conv_llm_calls + res.conv_cache_hits,
                    "llm_calls": res.conv_llm_calls,
                    "cache_hits": res.conv_cache_hits,
                    "cost_usd": round(res.conv_cost, 6),
                }
            )

        def _fold_fatal(res: _ConvResult) -> None:
            """Fold the failing conversation's non-spend counters before the stop (its spend is folded by
            _fold_spend, so the two never double-count)."""
            metrics["windows_total"] += res.windows
            metrics["dropped_schema"] += res.acc["dropped_schema"]
            metrics["dropped_evidence"] += res.acc["dropped_evidence"]
            for k in (
                "tags_object",
                "tags_list",
                "tags_empty",
                "tags_malformed",
                "unknown_tag_keys",
                "windows_unparseable",
                "reconcile_unparseable",
                "reconcile_overflow",
            ):
                metrics[k] += res.acc[k]
            if res.fatal and res.fatal.get("stage") == "reconcile":
                metrics["candidates_pre_reconcile"] += res.candidates_pre

        def _resolve(row: dict) -> transcript.Conversation | None:
            conv, reason = transcript.resolve_with_reason(row)
            if conv is None:
                metrics["conversations_skipped"] += 1
                metrics["skipped_sessions"].append(
                    {"conversation_id": row.get("conversation_id"), "reason": reason}
                )
            return conv

        # Sessions are resolved lazily, in selection order, as a rolling stream of at most `workers`
        # conversations in flight. Folding and writing happen only on the main thread and strictly by
        # submission index, so memories.jsonl and its order do not depend on which conversation finishes
        # first. A fatal is acted on only when it is next in order: conversations before it are written,
        # its partial counters are folded, and everything after it is dropped.
        stopped = False
        try:
            if workers == 1:
                for row in rows:
                    conv = _resolve(row)
                    if conv is None:
                        continue
                    res = _safe_process(conv)
                    if res.fatal is not None:
                        _fold_fatal(res)
                        _fold_spend(res, "fatal")
                        metrics["fatal_error"] = {"conversation_id": res.conv.conversation_id, **res.fatal}
                        stopped = True
                        break
                    _merge_and_write(res)
            else:
                rows_iter = iter(rows)
                pending: dict[int, Future] = {}
                buffered: dict[int, _ConvResult] = {}
                next_submit = 0
                next_merge = 0
                stop_submitting = False
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    while next_merge < next_submit or not stop_submitting:
                        # Row resolution stays on the main thread.
                        while len(pending) < workers and not stop_submitting:
                            nxt = next(rows_iter, None)
                            if nxt is None:
                                stop_submitting = True
                                break
                            conv = _resolve(nxt)
                            if conv is None:
                                continue
                            pending[next_submit] = ex.submit(_safe_process, conv)
                            next_submit += 1
                        if not pending:
                            break
                        done, _ = wait(pending.values(), return_when=FIRST_COMPLETED)
                        for idx in [i for i, f in pending.items() if f in done]:
                            buffered[idx] = pending.pop(idx).result()
                        # Bound wasted work once a fatal has completed; correctness does not depend on it.
                        if any(r.fatal is not None for r in buffered.values()):
                            stop_submitting = True
                        while next_merge in buffered:
                            res = buffered.pop(next_merge)
                            if res.fatal is not None:
                                _fold_fatal(res)
                                _fold_spend(res, "fatal")
                                metrics["fatal_error"] = {
                                    "conversation_id": res.conv.conversation_id,
                                    **res.fatal,
                                }
                                stopped = True
                                break
                            _merge_and_write(res)
                            next_merge += 1
                        if stopped:
                            # Work behind the fatal was already paid for: fold its spend, in order.
                            for idx in sorted(buffered):
                                _fold_spend(buffered.pop(idx), "abandoned")
                            for idx in sorted(pending):
                                fut = pending.pop(idx)
                                # a worker that raised paid for nothing this can attribute
                                with contextlib.suppress(Exception):
                                    _fold_spend(fut.result(), "abandoned")
                            break
        finally:
            mem_out.close()
            _record_explicit_caches(
                metrics, (client, reconcile_client), roles={f"extract-{arm}", "reconcile"}
            )
            _write_metrics(out_dir, metrics, wall0)
    return metrics, not stopped


def format_cost(cost: float, spent: float | None) -> str:
    """A cost for the summary and the progress log. With nothing replayed from the cache it is just
    the cost. Otherwise it is what this run spent, beside what the replayed answers cost, so a
    cached re-run never reads as money spent."""
    if spent is None or f"{cost:.2f}" == f"{spent:.2f}":
        return f"${cost:.2f}"
    return f"${spent:.2f} spent (cached answers worth ${cost - spent:.2f})"


def price_card_for(client) -> dict | None:
    """The Gemini rates a run's cost is computed at, with the date the provider's price card was
    read. None for a backend that reports its own cost."""
    from .providers.gemini import (
        GEMINI_CACHE_STORAGE_USD_PER_1M_TOKEN_HOUR,
        GEMINI_CACHED_INPUT_USD_PER_1M,
        GEMINI_INPUT_USD_PER_1M,
        GEMINI_OUTPUT_USD_PER_1M,
        GEMINI_PRICE_CARD_DATE,
        GeminiClient,
    )

    if not isinstance(client, GeminiClient):
        return None
    return {
        "source": "ai.google.dev/gemini-api/docs/pricing",
        "read_on": GEMINI_PRICE_CARD_DATE,
        "input_usd_per_1m": GEMINI_INPUT_USD_PER_1M,
        "cached_input_usd_per_1m": GEMINI_CACHED_INPUT_USD_PER_1M,
        "output_usd_per_1m": GEMINI_OUTPUT_USD_PER_1M,
        "cache_storage_usd_per_1m_token_hour": GEMINI_CACHE_STORAGE_USD_PER_1M_TOKEN_HOUR,
    }


def _generality_index(obj) -> dict:
    """Map normalized statement -> generality from the parsed extraction object."""
    out: dict[str, str] = {}
    if not isinstance(obj, dict):
        return out
    for c in obj.get("memories") or []:
        if not isinstance(c, dict):
            continue
        stmt = c.get("statement")
        gen = (c.get("generality") or "").lower()
        if isinstance(stmt, str) and gen in VALID_GENERALITY:
            out[mc.norm_statement(stmt)] = gen
    return out


def _record_explicit_caches(metrics: dict, clients, roles: set[str] | None = None) -> None:
    """Fold the explicit-cache state of the arm's clients into its metrics, deduplicated by resource
    name (the extract and reconcile clients can be the same object). Only resources whose prompt
    role is in `roles` count: in `run` the clients also hold the other stages' caches. A resource the
    client knows no role for counts. Storage is recorded apart from cost_usd_total, priced per
    resource on its own lifetime, capped at the TTL."""
    seen: dict[str, dict] = {}
    prompts: list[dict] = []
    for client in clients:
        if client is None or not getattr(client, "explicit_cache", False):
            continue
        role_of = getattr(client, "explicit_cache_roles", {})
        for entry in getattr(client, "explicit_caches", []):
            role = role_of.get(entry["name"])
            if roles is None or role is None or role in roles:
                seen.setdefault(entry["name"], entry)
        for record in getattr(client, "explicit_cache_prompts", []):
            if record not in prompts:
                prompts.append(record)
    if prompts:
        metrics["explicit_cache_prompts"] = prompts
    if not seen:
        return
    caches = list(seen.values())
    metrics["explicit_caches"] = caches
    metrics["explicit_cache_tokens"] = sum(c["tokens"] for c in caches)
    metrics["explicit_cache_storage_usd"] = round(
        sum(gemini_cache_storage_cost(c["tokens"], c["lifetime_s"]) for c in caches), 6
    )


def _write_metrics(out_dir: str, metrics: dict, wall0: float) -> None:
    m = dict(metrics)
    m["wall_clock_s"] = round(time.time() - wall0, 1)
    kept = m["memories_total"]
    m["cost_per_memory_usd"] = round(m["cost_usd_total"] / kept, 4) if kept else None
    m["empty_conversation_rate"] = (
        round(m["conversations_empty"] / m["conversations_run"], 3) if m["conversations_run"] else None
    )
    m["avg_windows_per_conversation"] = (
        round(m["windows_total"] / m["conversations_run"], 2) if m["conversations_run"] else None
    )
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def preflight_prompt_caches(client_roles: list[tuple], log=None) -> list[dict]:
    """Resolve the explicit-cache decision for every (client, prompts) pair before the first call,
    and log one line per prompt. Clients that are the same object are merged."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    merged: dict[int, tuple] = {}
    for c, roles in client_roles:
        if c is None or not getattr(c, "explicit_cache", False):
            continue
        cur = merged.setdefault(id(c), (c, []))
        cur[1].extend(roles)
    records: list[dict] = []
    for c, roles in merged.values():
        for rec in c.preflight_explicit_caches(roles):
            records.append(rec)
            if rec["status"] == CACHE_STATUS_CACHED:
                log(f"explicit cache: {rec['role']} prompt -> cached ({rec['tokens']} tokens)")
            else:
                log(f"explicit cache: {rec['role']} prompt -> inline ({rec['reason']})")
    return records
