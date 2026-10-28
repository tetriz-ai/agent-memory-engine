# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The cross-conversation merge: fold the arms' memories into one de-duplicated, ranked set.

The two arms read the same sessions through different lenses, and the same lesson recurs across
conversations in different words. This stage folds every arm's output into ONE set and ranks each
merged memory by how much independent support it has: distinct conversations first, then arms.

  1. LOAD + TAG   read each arm's memories.jsonl and tag every memory with its arm.
  2. PRE-CLUSTER  TF-IDF / cosine / average-linkage clustering of the statements at a loose
                  threshold (favouring recall; the model refines next). Yields candidate groups
                  and singletons.
  3. SYNTHESIZE   one model call per candidate group (singletons batched): unify members that are
                  the same lesson, split ones that are not, assign generality. Code, never the model,
                  unions the members' verified evidence, arms and conversations.
  4. GLOBAL DEDUP one model call over all synthesized statements to catch duplicates the grouping
                  missed.
  5. CORROBORATE  per final memory, distinct arms and distinct conversations; rank by
                  (distinct conversations desc, distinct arms desc).
  6. WRITE        memories.jsonl + metrics.json.

Every model call is cached on its exact inputs, so a re-run is free.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources

from . import cache as cache_mod
from . import cluster as rc
from . import contract as mc
from . import transcript
from .extract import price_card_for
from .paths import run_lock
from .providers import (
    USAGE_METRIC_KEYS,
    cost_from_cache_record,
    extract_with_route,
    fold_usage,
    gemini_cache_storage_cost,
    gemini_schema_digest,
    resolve_call_route,
    storage_cache_model_id,
    usage_fields,
)

PROMPT_ID = "merge-v1"
PROMPT_VERSION = f"{PROMPT_ID}+fmt-{mc.MEMORY_FORMAT_VERSION}"

# The arms that feed the merge, in the fixed order used everywhere (source_arms, output).
ARM_ORDER = ["facts", "preferences"]
DEFAULT_THRESHOLD = 0.25
# Single source of the vocabulary: contract.GENERALITY_VALUES (see extract.VALID_GENERALITY).
VALID_GENERALITY = frozenset(mc.GENERALITY_VALUES)

# How much of each member to SHOW the model (the full evidence is always preserved in output).
EVIDENCE_SHOWN_PER_MEMBER = 3
QUOTE_SHOWN_CHARS = 240
SINGLETON_BATCH = 12  # singletons are batched through the same SYNTHESIS prompt to bound calls
# Upper bound on group-synthesis parallelism. Each group's synthesis is one model call reading only
# its own members; output is identical at any worker count because the fold happens strictly in
# group order on the main thread.
MERGE_MAX_WORKERS = 6


# --------------------------------------------------------------------------- #
# 1. LOAD + TAG
# --------------------------------------------------------------------------- #
def load_arms(arm_paths: dict[str, str], log=None) -> list[dict]:
    """Read every arm's memories.jsonl and tag each memory with its arm, in ARM_ORDER. A blank line
    is skipped; a malformed line, or one without a statement, is skipped and counted."""
    return _load_arms(arm_paths, log)[0]


def _load_arms(arm_paths: dict[str, str], log=None) -> tuple[list[dict], int]:
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    mems: list[dict] = []
    skipped = 0
    for arm in ARM_ORDER:
        path = arm_paths[arm]
        if not os.path.exists(path):
            raise FileNotFoundError(f"arm {arm}: {path} not found")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                # A line without a non-empty statement would fail in clustering; drop and count it.
                if not isinstance(o, dict) or not (
                    isinstance(o.get("statement"), str) and o["statement"].strip()
                ):
                    skipped += 1
                    continue
                o["_arm"] = arm
                mems.append(o)
    if skipped:
        log(f"merge: skipped {skipped} malformed arm line(s) lacking a statement")
    return mems, skipped


def member_conversation_ids(m: dict) -> list[str]:
    """Every distinct conversation_id backing one member memory: its origin's `conversation_id` and
    every evidence item that names its own conversation_id. Sorted, de-duplicated."""
    ids: set[str] = set()
    origin = m.get("origin") or {}
    if isinstance(origin.get("conversation_id"), str):
        ids.add(origin["conversation_id"])
    for ev in m.get("evidence") or []:
        if isinstance(ev, dict) and isinstance(ev.get("conversation_id"), str):
            ids.add(ev["conversation_id"])
    return sorted(ids)


def member_evidence(m: dict) -> list[dict]:
    """All of a member's evidence items, each annotated with its arm and (best-effort) its source
    conversation_id so every quote in the merged output stays traceable. The full set is preserved
    — nothing is dropped here (unlike the SHOWN subset fed to the model)."""
    arm = m["_arm"]
    origin = m.get("origin") or {}
    fallback_cid = origin.get("conversation_id")
    out = []
    for ev in m.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        item = {
            "quote": ev.get("quote"),
            "turn": ev.get("turn"),
            "speaker": ev.get("speaker"),
            "conversation_id": ev.get("conversation_id") or fallback_cid,
            "arm": arm,
            # per-quote effective authorship, so build_final can recompute the merged memory's
            # provenance.confidence from the final evidence set; null reads as "no signal".
            "authorship": ev.get("authorship"),
        }
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# 2. PRE-CLUSTER (TF-IDF clustering on the STATEMENTS)
# --------------------------------------------------------------------------- #
def _statement_conv(statement: str) -> transcript.Conversation:
    """Wrap one statement as a single-turn Conversation so cluster.conversation_tokens tokenizes it."""
    return transcript.Conversation(
        conversation_id="stmt",
        tool="",
        source="",
        date=None,
        turns=[transcript.Turn(1, "user", statement)],
        truncated=False,
    )


def cluster_memories(mems: list[dict], threshold: float):
    """Cluster the memory STATEMENTS by TF-IDF average-linkage.
    Returns (clusters, singletons) as lists of index-lists / indices into `mems`."""
    docs = [rc.conversation_tokens(_statement_conv(m["statement"])) for m in mems]
    ids = list(range(len(mems)))
    clusters, singletons, _sims = rc.cluster_documents(ids, docs, threshold)
    return clusters, singletons


# --------------------------------------------------------------------------- #
# 3. SYNTHESIZE per candidate group
# --------------------------------------------------------------------------- #
def load_prompt(section: str) -> str:
    """Pull one system prompt out of prompts/merge.md by its BEGIN/END markers ('SYNTHESIS' or
    'DEDUP')."""
    path = "merge.md"
    text = resources.files("agent_memory_engine.prompts").joinpath(path).read_text(encoding="utf-8")
    begin, end = f"===BEGIN {section} PROMPT===", f"===END {section} PROMPT==="
    b, e = text.find(begin), text.find(end)
    if b < 0 or e < 0 or e < b:
        raise ValueError(f"{path} must contain '{begin}' before '{end}'")
    return text[b + len(begin) : e].strip()


def _render_member(local_id: str, m: dict) -> str:
    cids = member_conversation_ids(m)
    ev_lines = []
    for ev in (m.get("evidence") or [])[:EVIDENCE_SHOWN_PER_MEMBER]:
        if not isinstance(ev, dict):
            continue
        q = (ev.get("quote") or "").replace("\n", " ")
        if len(q) > QUOTE_SHOWN_CHARS:
            q = q[:QUOTE_SHOWN_CHARS] + "…"
        ev_lines.append(f'      - "{q}" (turn {ev.get("turn")}, {ev.get("speaker")})')
    head = (
        f"[{local_id}] arm={m['_arm']} | type={m.get('type')} | scope={m.get('scope')} | "
        f"conversations={','.join(cids) or '?'}"
    )
    stmt = f"    statement: {m['statement']}"
    desc = mc._clean_free_str(m.get("description"))
    desc_line = f"\n    description: {desc}" if desc else ""
    ev = ("    evidence:\n" + "\n".join(ev_lines)) if ev_lines else "    evidence: (none)"
    return f"{head}\n{stmt}{desc_line}\n{ev}"


def _group_key(mems: list[dict], idxs: list[int]) -> str:
    """Stable cache key for a group: hash of its members' memory_ids (order-independent)."""
    mids = sorted(str(mems[i].get("memory_id") or mems[i]["statement"]) for i in idxs)
    return "group:" + hashlib.sha256("+".join(mids).encode()).hexdigest()[:16]


def _scope_of(m: dict) -> tuple[str, str | None]:
    """(scope_level, scope_ref) for a member/synth dict, from its fixed-enum fields or, as a fallback,
    one free `scope` string. Falls back to personal when nothing usable is present."""
    lvl, ref = mc.derive_scope(m.get("scope_level"), m.get("scope_ref"), m.get("scope"))
    return (lvl, ref) if lvl is not None else ("personal", None)


def _norm_generality(g) -> str:
    g = (g or "").strip().lower()
    if g in VALID_GENERALITY:
        return g
    if "general" in g or "principle" in g:
        return "general-principle"
    return "instance-specific"


def _fresh_group_metrics() -> dict:
    """A per-group metrics dict for one synthesize_group call. Single-writer (only its own group's
    thread touches it), so it is safe to mutate under the pool; the main thread folds it into the
    shared `metrics` in groups_idx order afterward. Same key set synthesize_group mutates."""
    return {
        "llm_calls": 0,
        "cache_hits": 0,
        "members_uncovered": 0,
        "cost_usd_total": 0.0,
        "cached_answers_usd": 0.0,
        "model_duration_ms_total": 0,
        "canonical_model": None,
        "http_retries": 0,
        **{k: 0 for k in USAGE_METRIC_KEYS},
    }


def _fold_call_spend(metrics: dict, res: dict, cached: bool = False) -> None:
    """One call's cost, duration, tokens and retries. Called the moment a result EXISTS — for a
    fresh call that is before it is persisted, because the provider has already billed and a local
    write failure must not be able to report the money as never spent. A cache hit replays the
    saved call's numbers: the metric says what the stage costs, not how warm the cache was.
    `cached_answers_usd` keeps the replayed part apart, so what the run spent is the difference."""
    cost = cost_from_cache_record(res)
    metrics["cost_usd_total"] += cost
    if cached:
        metrics["cached_answers_usd"] = metrics.get("cached_answers_usd", 0.0) + cost
    metrics["model_duration_ms_total"] += res.get("duration_ms", 0)
    metrics["http_retries"] += int(res.get("http_retries") or 0)
    fold_usage(metrics, res.get("usage"))


def _fold_group_spend(metrics: dict, gm: dict, group_key: str, status: str) -> None:
    """Fold the REAL SPEND of a group whose output will not be used.

    A stopped merge has still paid for every call it made. Spend only: the quality counters
    (members_uncovered and the like) stay with the groups that actually contributed."""
    metrics["cost_usd_total"] += gm["cost_usd_total"]
    metrics["cached_answers_usd"] = metrics.get("cached_answers_usd", 0.0) + gm.get("cached_answers_usd", 0.0)
    metrics["llm_calls"] += gm["llm_calls"]
    metrics["cache_hits"] += gm["cache_hits"]
    metrics["model_duration_ms_total"] += gm["model_duration_ms_total"]
    for k in ("http_retries", *USAGE_METRIC_KEYS):
        metrics[k] += gm[k]
    if gm.get("canonical_model"):
        metrics["canonical_model"] = gm["canonical_model"]
    metrics["partial_groups"].append(
        {
            "group": group_key,
            "status": status,
            "calls": gm["llm_calls"] + gm["cache_hits"],
            "llm_calls": gm["llm_calls"],
            "cache_hits": gm["cache_hits"],
            "cost_usd": round(gm["cost_usd_total"], 6),
        }
    )


def _group_ran(gm: dict) -> bool:
    """Whether a group made (or replayed) a call or spent money. A group the pool never reached has
    a pristine _fresh_group_metrics and nothing to account for."""
    return bool(gm["llm_calls"] or gm["cache_hits"] or gm["cost_usd_total"])


def _fold_group_metrics(metrics: dict, gm: dict) -> None:
    """Fold one group's per-group metrics into the shared `metrics`. Called strictly in groups_idx
    order on the main thread, so the float sums land in the same order at any worker count.
    Carries `fatal_error` when the group failed."""
    for k in (
        "llm_calls",
        "cache_hits",
        "members_uncovered",
        "cost_usd_total",
        "model_duration_ms_total",
        "http_retries",
        *USAGE_METRIC_KEYS,
    ):
        metrics[k] += gm[k]
    metrics["cached_answers_usd"] = metrics.get("cached_answers_usd", 0.0) + gm.get("cached_answers_usd", 0.0)
    if gm.get("canonical_model"):
        metrics["canonical_model"] = gm["canonical_model"]
    if gm.get("fatal_error"):
        metrics["fatal_error"] = gm["fatal_error"]


def _synthesize_group_safely(mems, idxs, system_prompt, client, cache, metrics):
    """synthesize_group, with any raise (a cache write that fails, a malformed answer the code did
    not expect) turned into this group's fatal. The spend already folded into `metrics` stays."""
    try:
        return synthesize_group(mems, idxs, system_prompt, client, cache, metrics)
    except Exception as e:
        metrics["fatal_error"] = {
            "group": _group_key(mems, idxs),
            "stage": "exception",
            "error": f"{type(e).__name__}: {e}",
        }
        return None


def synthesize_group(mems, idxs, system_prompt, client, cache, metrics):
    """Run the SYNTHESIS prompt on one candidate group. Returns a list of synthesized-memory dicts,
    each {statement,type,scope,generality, _member_idxs:[global indices]}. Code — not the
    model — assigns each input member to exactly one output (uncovered members become passthroughs),
    so no member's evidence is ever lost."""
    local = {f"M{k + 1}": idxs[k] for k in range(len(idxs))}  # local id -> global index
    body = "\n\n".join(_render_member(f"M{k + 1}", mems[idxs[k]]) for k in range(len(idxs)))
    group_input = (
        "Unify the following candidate memories per the rules. Member ids are the [M#] labels.\n\n" + body
    )
    group_key = _group_key(mems, idxs)

    # Per-prompt identity (see extract._cached_call).
    route = resolve_call_route(client, system_prompt)
    fp = cache_mod.fingerprint(group_key, PROMPT_VERSION, route.cache_model_id, system_prompt, group_input)
    cached = cache.get(fp)
    if cached is not None:
        res = cached
        metrics["cache_hits"] += 1
        # A hit replays the saved call's cost and tokens.
        _fold_call_spend(metrics, res, cached=True)
    else:
        r = extract_with_route(client, system_prompt, group_input, route)
        if not r.ok:
            # A failed call is a call and may still have been billed: it counts, and nothing is cached.
            metrics["llm_calls"] += 1
            _fold_call_spend(
                metrics, {"cost_usd": r.cost_usd, "duration_ms": r.duration_ms, **usage_fields(r)}
            )
            metrics["fatal_error"] = {"group": group_key, "error": r.error}
            return None  # signal fatal
        store_id = storage_cache_model_id(r, route)
        if store_id != route.cache_model_id:
            fp = cache_mod.fingerprint(group_key, PROMPT_VERSION, store_id, system_prompt, group_input)
        # `usage` + `cost_formula` ride beside the fingerprint, never inside it.
        res = {
            "raw_text": r.raw_text,
            "cost_usd": r.cost_usd,
            "duration_ms": r.duration_ms,
            "canonical_model": r.meta.get("canonical_model"),
            **usage_fields(r),
        }
        # Billed already — count and fold BEFORE persisting. A cache.put that cannot write (full disk,
        # bad permissions) then stops the merge as a fatal, with this call's spend in the metrics.
        metrics["llm_calls"] += 1  # count only real model invocations; a hit is free
        _fold_call_spend(metrics, res)
        cache.put(fp, res)
    if res.get("canonical_model"):
        metrics["canonical_model"] = res["canonical_model"]

    obj = mc.extract_json(res["raw_text"]) or {}
    out_mems = obj.get("memories") if isinstance(obj, dict) else None
    synthesized = []
    covered: set[int] = set()
    if isinstance(out_mems, list):
        for om in out_mems:
            if not isinstance(om, dict):
                continue
            refs = om.get("members") or []
            gidxs = []
            for rid in refs:
                gi = local.get(str(rid))
                if gi is not None and gi not in covered:  # first assignment wins (partition)
                    gidxs.append(gi)
                    covered.add(gi)
            if not gidxs:
                continue  # an output referencing only already-covered / unknown members is dropped
            stmt = om.get("statement")
            mtype = (om.get("type") or "").lower()
            scope = om.get("scope")
            if not (isinstance(stmt, str) and stmt.strip()):
                # unusable phrasing: fall back to the most-corroborated member's own fields
                covered -= set(gidxs)
                for gi in gidxs:
                    synthesized.append(_passthrough(mems, gi))
                    covered.add(gi)
                continue
            if mtype not in mc.VALID_TYPES:
                mtype = mems[gidxs[0]].get("type") or "fact"
            # An intention member keeps the group an intention, so it stays held for review with its
            # meta, whatever type the model names.
            if any(mems[g].get("type") == "intention" for g in gidxs):
                mtype = "intention"
            # Scope: honor the model's fixed-enum scope (or a free scope string), else the first member's.
            scope_level, scope_ref = mc.derive_scope(om.get("scope_level"), om.get("scope_ref"), scope)
            if scope_level is None:
                scope_level, scope_ref = _scope_of(mems[gidxs[0]])
            scope_disp = (
                scope_level if (scope_level == "personal" or not scope_ref) else f"{scope_level}/{scope_ref}"
            )
            synthesized.append(
                {
                    "statement": stmt.strip(),
                    "type": mtype,
                    "scope_level": scope_level,
                    "scope_ref": scope_ref,
                    "scope": scope_disp,
                    "generality": _norm_generality(om.get("generality")),
                    # description travels with the statement: since synthesis REWRITES the statement into
                    # a unified phrasing, the model writes the matching unified description; fall back to
                    # the first covered member's description (as type/scope do) when it omits one.
                    "description": (
                        mc._clean_free_str(om.get("description"))
                        or mc._clean_free_str(mems[gidxs[0]].get("description"))
                    ),
                    "review": mc._review_for(mtype),
                    "intention_meta": _intention_meta_from_members(mems, gidxs, mtype),
                    "_member_idxs": sorted(gidxs),
                }
            )
    # Any member the model failed to cover survives as its own passthrough memory (never dropped).
    for gi in idxs:
        if gi not in covered:
            metrics["members_uncovered"] += 1
            synthesized.append(_passthrough(mems, gi))
    return synthesized


def _passthrough(mems, gi) -> dict:
    """A member the model neither unified nor phrased usably becomes its own memory, keeping its
    original fields. Generality is inferred cheaply (rule/preference/convention -> principle)."""
    m = mems[gi]
    mtype = m.get("type") or "fact"
    gen = "general-principle" if mtype in {"preference", "convention", "caution"} else "instance-specific"
    scope_level, scope_ref = _scope_of(m)
    scope_disp = scope_level if (scope_level == "personal" or not scope_ref) else f"{scope_level}/{scope_ref}"
    return {
        "statement": (m.get("statement") or "").strip(),
        "type": mtype,
        "scope_level": scope_level,
        "scope_ref": scope_ref,
        "scope": scope_disp,
        "generality": gen,
        "review": mc._review_for(mtype),
        "intention_meta": _intention_meta_from_members(mems, [gi], mtype),
        "description": mc._clean_free_str(m.get("description")),  # kept as-is (no restatement here)
        "_member_idxs": [gi],
    }


# --------------------------------------------------------------------------- #
# 4. GLOBAL DEDUP across synthesized memories
# --------------------------------------------------------------------------- #
def global_dedup(synth: list[dict], system_prompt, client, cache, metrics) -> list[list[int]]:
    """One LLM call: given all synthesized statements, return groups of indices (into `synth`) that
    are duplicates. Returns a list of index-groups (each >= 2). Empty on any parse trouble."""
    if len(synth) < 2:
        return []
    lines = []
    for i, s in enumerate(synth):
        st = s["statement"].replace("\n", " ")
        lines.append(f"[S{i + 1}] (type={s['type']}, scope={s['scope']}) {st}")
    dedup_input = (
        "Identify duplicate/near-duplicate memories across this list. Ids are the [S#] "
        "labels.\n\n" + "\n".join(lines)
    )
    key = "dedup:" + hashlib.sha256("\n".join(s["statement"] for s in synth).encode()).hexdigest()[:16]

    route = resolve_call_route(client, system_prompt)
    fp = cache_mod.fingerprint(key, PROMPT_VERSION, route.cache_model_id, system_prompt, dedup_input)
    cached = cache.get(fp)
    if cached is not None:
        res = cached
        metrics["cache_hits"] += 1
        _fold_call_spend(metrics, res, cached=True)  # a hit replays the saved call's cost and tokens
    else:
        r = extract_with_route(client, system_prompt, dedup_input, route)
        if not r.ok:
            metrics["llm_calls"] += 1
            _fold_call_spend(
                metrics, {"cost_usd": r.cost_usd, "duration_ms": r.duration_ms, **usage_fields(r)}
            )
            metrics["dedup_error"] = r.error
            return []  # non-fatal: fall back to no cross-group dedup
        store_id = storage_cache_model_id(r, route)
        if store_id != route.cache_model_id:
            fp = cache_mod.fingerprint(key, PROMPT_VERSION, store_id, system_prompt, dedup_input)
        # `usage` + `cost_formula` ride beside the fingerprint, never inside it.
        res = {
            "raw_text": r.raw_text,
            "cost_usd": r.cost_usd,
            "duration_ms": r.duration_ms,
            "canonical_model": r.meta.get("canonical_model"),
            **usage_fields(r),
        }
        # Billed already — count and fold BEFORE persisting, so a cache.put that cannot write
        # cannot un-spend a call the provider has answered.
        metrics["llm_calls"] += 1  # count only real model invocations; a hit is free
        _fold_call_spend(metrics, res)
        cache.put(fp, res)

    obj = mc.extract_json(res["raw_text"]) or {}
    groups_raw = obj.get("duplicate_groups") if isinstance(obj, dict) else None
    groups: list[list[int]] = []
    seen: set[int] = set()
    if isinstance(groups_raw, list):
        for g in groups_raw:
            if not isinstance(g, list):
                continue
            idxs = []
            for rid in g:
                label = str(rid).strip().lstrip("Ss")
                if label.isdigit():
                    j = int(label) - 1
                    if 0 <= j < len(synth) and j not in seen:
                        idxs.append(j)
                        seen.add(j)
            if len(idxs) >= 2:
                groups.append(sorted(idxs))
    return groups


# --------------------------------------------------------------------------- #
# 5. CORROBORATE + build final merged memories
# --------------------------------------------------------------------------- #
def _dedupe_evidence(evs: list[dict]) -> list[dict]:
    """Drop exactly-identical evidence rows unioned across members. Keyed on the full identity
    including `arm`: two arms citing the same quote stay traceable; only a duplicated row goes."""
    seen, out = set(), []
    for e in evs:
        key = (e.get("quote"), e.get("turn"), e.get("speaker"), e.get("conversation_id"), e.get("arm"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _temporal_from_members(mems: list[dict], member_idxs: list[int]) -> tuple:
    """Carry (valid_as_of, temporal_kind, volatility) onto a merged memory: code picks them, the
    synthesis model never invents them. Latest wins: all three, as one unit, from the member with the
    most recent conversation date (`when`); ties and undated members fall back to the first member."""
    best = member_idxs[0]
    best_when = mems[best].get("when") or ""
    for gi in member_idxs[1:]:
        w = mems[gi].get("when") or ""
        if w > best_when:  # strict > keeps the first member on a tie (deterministic)
            best, best_when = gi, w
    m = mems[best]
    return m.get("valid_as_of"), m.get("temporal_kind"), m.get("volatility")


def _disposition_from_members(mems: list[dict], member_idxs: list[int], distinct_convs: int) -> dict:
    """Carry disposition onto a merged memory (the synthesis model never sees it). Borderline if ANY
    covered member is borderline, UNLESS >=2 distinct CONVERSATIONS corroborate it — only independent
    conversations resolve a single-source doubt. Arm count is NOT corroboration: the arms read the same
    conversations, so two arms agreeing on one conversation is one source seen twice. Note carried from
    the first borderline member."""
    any_borderline = False
    note = ""
    for gi in member_idxs:
        d = mems[gi].get("disposition")
        lvl = d.get("level") if isinstance(d, dict) else (d if isinstance(d, str) else None)
        if lvl == "borderline":
            any_borderline = True
            if not note:
                n = d.get("note") if isinstance(d, dict) else mems[gi].get("disposition_note")
                note = n or ""
    if any_borderline and distinct_convs < 2:
        return {"level": "borderline", "note": note}
    return {"level": "confident", "note": ""}


def _member_decision_outcome(m: dict):
    """A member's decision outcome."""
    return m.get("decision_outcome")


def _intention_meta_from_members(mems: list[dict], member_idxs: list[int], mtype: str):
    """An intention's meta comes from its latest-dated intention member that has one (by `when`, then
    by member order); a member with no meta never erases another's. null for non-intentions."""
    if mtype != "intention":
        return None
    cands = [
        ((mems[gi].get("when") or ""), gi, mems[gi]["intention_meta"])
        for gi in member_idxs
        if mems[gi].get("type") == "intention" and mems[gi].get("intention_meta")
    ]
    return max(cands, key=lambda c: (c[0], c[1]))[2] if cands else None


def _decision_outcome_from_members(mems: list[dict], member_idxs: list[int], mtype: str):
    """A decision's outcome is code-carried from the LATEST-dated decision
    member (by `when`), consistent with the temporal latest-wins rule — never the first-by-load-order,
    which could publish a stale/flipped outcome. If members recorded DIFFERENT outcomes, the latest is
    kept and the divergence is flagged rather than silently dropped. null for non-decisions."""
    if mtype != "decision":
        return None
    # Only DECISION members with a non-empty STRING outcome are candidates: a stray outcome on a
    # non-decision member must not win, and a non-string (malformed model output) must not reach the
    # string concat below (which would TypeError and abort the whole merge).
    cands = []
    for gi in member_idxs:
        if mems[gi].get("type") != "decision":
            continue
        r = _member_decision_outcome(mems[gi])
        if isinstance(r, str) and r.strip():
            cands.append((mems[gi].get("when") or "", r.strip()))
    if not cands:
        return None
    cands.sort(key=lambda wr: wr[0])  # ascending by conversation date; last = latest
    latest = cands[-1][1]
    if len({r for _, r in cands}) > 1:
        return latest + "  [merged: members recorded differing outcomes; latest by date kept]"
    return latest


def _secondary_type_from_members(mems: list[dict], member_idxs: list[int], primary_type: str):
    """Code-carry the optional SECONDARY type: the latest-dated member's valid secondary type that
    differs from the final primary type. A single field, never a list. null when no member carries one."""
    cands = []
    for gi in member_idxs:
        st = mems[gi].get("secondary_type")
        if (
            isinstance(st, str)
            and st.strip().lower() in mc.VALID_TYPES
            and st.strip().lower() != primary_type
        ):
            cands.append((mems[gi].get("when") or "", st.strip().lower()))
    if not cands:
        return None
    cands.sort(key=lambda wr: wr[0])
    return cands[-1][1]


def _applicable_scope_from_members(
    mems: list[dict], member_idxs: list[int], identified_level, identified_ref
):
    """Code-carry the optional APPLICABLE scope: the latest-dated member that named one, kept only when
    it genuinely differs from the final identified scope. Returns (level, ref) or (None, None)."""
    cands = []
    for gi in member_idxs:
        lvl = mems[gi].get("scope_applicable_level")
        if isinstance(lvl, str) and lvl.strip().lower() in mc.VALID_SCOPE_LEVELS:
            ref = mems[gi].get("scope_applicable_ref")
            ref = ref.strip() if isinstance(ref, str) and ref.strip() else None
            cands.append((mems[gi].get("when") or "", lvl.strip().lower(), ref))
    if not cands:
        return None, None
    cands.sort(key=lambda c: c[0])
    _, lvl, ref = cands[-1]
    if lvl == "personal":
        ref = None
    if lvl == identified_level and ref == identified_ref:  # no real divergence -> drop to null
        return None, None
    return lvl, ref


def build_final(synth: dict, mems: list[dict]) -> dict:
    """Assemble one final merged memory from a synthesized entry: union its members' evidence,
    arms, and conversation_ids, then compute corroboration counts. Temporal rich-meta, the dimensioned
    tags, disposition, a decision's outcome, and the optional secondary_type / applicable-scope fields are
    all code-carried here from the final member set (`_member_idxs`) — never taken from the synthesis model,
    which flattens tags and emits neither disposition nor those carried fields — so they survive uniformly
    whether the entry came from a passthrough, a synthesis, or a dedup fold."""
    member_idxs = synth["_member_idxs"]
    evidence: list[dict] = []
    arms: set[str] = set()
    conv_ids: set[str] = set()
    for gi in member_idxs:
        m = mems[gi]
        arms.add(m["_arm"])
        conv_ids.update(member_conversation_ids(m))  # already spans origin + evidence conv ids
        evidence.extend(member_evidence(m))
    source_arms = [a for a in ARM_ORDER if a in arms]
    source_cids = sorted(conv_ids)
    s_level, s_ref = _scope_of(synth)
    # Tags: union the members' typed objects, single domain/activity by majority with the losers
    # demoted to keyword, open axes unioned. The synthesis model's own `tags` answer is not read.
    tags = mc.union_tags([mems[gi].get("tags") for gi in member_idxs], single_value="majority")
    valid_as_of, temporal_kind, volatility = _temporal_from_members(mems, member_idxs)
    app_level, app_ref = _applicable_scope_from_members(mems, member_idxs, s_level, s_ref)
    final_evidence = _dedupe_evidence(evidence)
    # Provenance confidence is RECOMPUTED from the final merged evidence set against the FINAL type
    # (not taken as the weakest member's stored value) — the cleaner of the two aggregation options:
    # synthesis/dedup can rephrase a member into a different type, and admissibility depends on that
    # final type, so a stored member-level confidence computed under the member's own type could be
    # wrong. Per-quote authorship is the single source of truth; a quote with null authorship reads
    # as "no signal" (uncertain, not low).
    prov_conf = mc.provenance_confidence(synth["type"], [e.get("authorship") for e in final_evidence])
    return {
        "memory_id": mc.memory_id(synth["statement"]),
        "statement": synth["statement"],
        # description rides with the (model-rewritten) statement on the synth entry — NOT code-carried
        # from a member like the temporal fields, which are phrasing-independent metadata.
        "description": synth.get("description"),
        "type": synth["type"],
        # optional second type, code-carried from the members (the synth model does not emit it)
        "secondary_type": _secondary_type_from_members(mems, member_idxs, synth["type"]),
        # nested scope: identified (where observed) + applicable (where it applies, when that differs;
        # null in the common case). Replaces the flat scope_* fields and the redundant `scope` string.
        "scope": {
            "identified": {"level": s_level, "ref": s_ref},
            "applicable": ({"level": app_level, "ref": app_ref} if app_level is not None else None),
        },
        "tags": tags,
        "generality": synth["generality"],
        # review is a required non-nullable field (schema): read it directly so a synth entry that
        # forgot to populate it fails fast here rather than emitting a schema-violating review=null.
        "review": synth["review"],
        "intention_meta": _intention_meta_from_members(mems, member_idxs, synth["type"]),
        "decision_outcome": _decision_outcome_from_members(mems, member_idxs, synth["type"]),
        # grouped temporal rich-meta (kind/volatility/valid_as_of), code-carried latest-wins as a unit.
        "temporal": {"kind": temporal_kind, "volatility": volatility, "valid_as_of": valid_as_of},
        # disposition (human-review trust) stays its own object — a DIFFERENT axis from provenance.confidence.
        "disposition": _disposition_from_members(mems, member_idxs, len(source_cids)),
        # grouped provenance: the confidence annotation (recomputed above from the final evidence and
        # final type; it annotates, it does not gate) plus the merge-only corroboration counts.
        "provenance": {
            "confidence": prov_conf,
            "corroboration": {
                "distinct_arms": len(source_arms),
                "distinct_conversations": len(source_cids),
                "source_arms": source_arms,
                "source_conversation_ids": source_cids,
            },
        },
        "evidence": final_evidence,
    }


def merge_dedup_groups(synth: list[dict], groups: list[list[int]]) -> list[dict]:
    """Fold each dedup group into one synthesized entry (union of member indices), keeping the
    statement/type/scope/generality of the most-corroborated member of the group (by distinct
    conversations, then longest statement, then id). Entries in no group pass through unchanged."""
    in_group = {j for g in groups for j in g}
    out = []
    for g in groups:
        merged_idxs = sorted({gi for j in g for gi in synth[j]["_member_idxs"]})
        # Pick the representative entry. An `intention` MUST win the tie whenever any member of the
        # group is an intention: the merged entry inherits the rep's type + review, and an
        # intention carries review=hold-for-review (the safety gate: never loaded without a human check). If a
        # non-intention were picked over an intention member, the merged memory would silently lose
        # that gate. So intention-ness is the first key, then richness (member count), then longest
        # statement.
        rep = max(
            g,
            key=lambda j: (
                synth[j].get("type") == "intention",
                len(synth[j]["_member_idxs"]),
                len(synth[j]["statement"]),
                synth[j]["statement"],
            ),
        )
        base = dict(synth[rep])
        base["_member_idxs"] = merged_idxs
        out.append(base)
    for j, s in enumerate(synth):
        if j not in in_group:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
def run(
    arm_paths: dict[str, str],
    out_dir: str,
    client,
    *,
    cache_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
    workers: int = 1,
    log=None,
) -> tuple[dict, bool]:
    """Merge the arms' memories.jsonl files (`arm_paths`, keyed by arm name) into one ranked
    `memories.jsonl` plus `metrics.json` in `out_dir`. Returns (metrics, ok); ok is False when a
    model failure stopped the run, and `metrics["fatal_error"]` says where."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    workers = max(1, min(workers, MERGE_MAX_WORKERS))
    mems, arm_lines_skipped = _load_arms(arm_paths, log=log)
    clusters, singletons = cluster_memories(mems, threshold)

    synth_prompt = load_prompt("SYNTHESIS")
    dedup_prompt = load_prompt("DEDUP")
    cache = cache_mod.Cache(cache_dir)
    with run_lock(out_dir):
        return _run_locked(
            mems,
            arm_lines_skipped,
            clusters,
            singletons,
            synth_prompt,
            dedup_prompt,
            cache,
            out_dir,
            client,
            threshold,
            workers,
        )


def _run_locked(
    mems,
    arm_lines_skipped,
    clusters,
    singletons,
    synth_prompt,
    dedup_prompt,
    cache,
    out_dir,
    client,
    threshold,
    workers,
):

    # Candidate groups: every cluster (>=2), plus singletons batched so a batch runs through the
    # SAME synthesis prompt (the prompt splits genuinely-unrelated members back apart).
    groups_idx: list[list[int]] = [list(c) for c in clusters]
    for k in range(0, len(singletons), SINGLETON_BATCH):
        groups_idx.append(list(singletons[k : k + SINGLETON_BATCH]))

    metrics = {
        "arm": "merge",
        "prompt_version": PROMPT_VERSION,
        "model_id": client.model_id,
        "thinking_level": getattr(client, "thinking_level", None),
        "response_schema_digest": (
            gemini_schema_digest(client.response_schema)
            if getattr(client, "response_schema", None) is not None
            else None
        ),
        "explicit_cache": bool(getattr(client, "explicit_cache", False)),
        "price_card": price_card_for(client),
        # A LIST, not one name: this stage makes calls under TWO prompts — per-group synthesis and
        # the cross-group dedup — so each gets its own cache resource and each bills storage.
        "explicit_caches": [],
        # One record per prompt this stage uses (synthesis, dedup): cached with its resource, or
        # inline with the reason. Both merge prompts sit under the provider's caching floor, so in
        # practice this stage sends them inline.
        "explicit_cache_prompts": [],
        "explicit_cache_tokens": 0,
        "explicit_cache_storage_usd": 0.0,
        "http_retries": 0,
        # Synthesis groups that were paid for but never folded into the output: the one that hit the
        # fatal, and any that had already run in parallel when the stop landed.
        "partial_groups": [],
        "candidates_in": len(mems),
        # Arm lines that were not JSON or had no statement; they never reach clustering.
        "arm_lines_skipped": arm_lines_skipped,
        "threshold": threshold,
        "clusters": len(clusters),
        "singletons": len(singletons),
        "candidate_groups": len(groups_idx),
        "singleton_batches": len(groups_idx) - len(clusters),
        "llm_calls": 0,
        "cache_hits": 0,
        "members_uncovered": 0,
        "synthesized_pre_dedup": 0,
        "dedup_groups": 0,
        "final_merged": 0,
        "cost_usd_total": 0.0,
        "cached_answers_usd": 0.0,
        "model_duration_ms_total": 0,
        "canonical_model": None,
        # Token anatomy of the merge stage's spend (prompt / answer / thinking / provider-cached
        # prompt). `usage_missing_calls` > 0 means some call came back without readable token
        # usage, so the token totals cover only part of the stage.
        **{k: 0 for k in USAGE_METRIC_KEYS},
        "per_arm_in": {a: sum(1 for m in mems if m["_arm"] == a) for a in ARM_ORDER},
    }
    wall0 = time.time()

    # Per-group synthesis. Each call reads only its own members (read-only `mems`) and writes only
    # its own per-group metrics dict, so the groups are independent and safe to run concurrently. The
    # pool size bounds in-flight LLM calls at <= workers <= MERGE_MAX_WORKERS. Results and metrics are
    # then folded/assembled STRICTLY in groups_idx order (never completion order): that order drives
    # both the [S#] labels and the global_dedup cache fingerprint, so assembling it in a fixed order
    # keeps the merged output — and the $0 warm-cache replay — byte-identical at any worker count.
    # A group that fails, by a model error or by a raise, returns None with its own fatal_error and
    # spend in its per-group metrics, so every failure is acted on in index order below.
    results: list[list[dict] | None] = [None] * len(groups_idx)
    gms: list[dict] = [_fresh_group_metrics() for _ in groups_idx]
    if workers == 1:
        # Inline, thread-free path: the sequential semantics, including the early stop —
        # groups AFTER a fatal never run.
        for i, idxs in enumerate(groups_idx):
            results[i] = _synthesize_group_safely(mems, idxs, synth_prompt, client, cache, gms[i])
            if results[i] is None:
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _synthesize_group_safely, mems, groups_idx[i], synth_prompt, client, cache, gms[i]
                ): i
                for i in range(len(groups_idx))
            }
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()  # None = this group's fatal; a list = success
    # Fold metrics and assemble `synthesized` STRICTLY in groups_idx order (never completion order).
    # The earliest-index failure wins, exactly as the sequential runner orders it. Only groups
    # strictly before that one contribute their OUTPUT; the groups after it, which under >1 worker
    # have already run, contribute their SPEND — they were billed, and a stopped run must report
    # what it spent.
    synthesized: list[dict] = []
    for i in range(len(groups_idx)):
        res = results[i]
        if res is None:  # this group's fatal: a model error or a raise
            _fold_group_metrics(metrics, gms[i])  # carries fatal_error AND its own spend
            metrics["partial_groups"].append(
                {
                    "group": _group_key(mems, groups_idx[i]),
                    "status": "fatal",
                    "calls": gms[i]["llm_calls"] + gms[i]["cache_hits"],
                    "llm_calls": gms[i]["llm_calls"],
                    "cache_hits": gms[i]["cache_hits"],
                    "cost_usd": round(gms[i]["cost_usd_total"], 6),
                }
            )
            for j in range(i + 1, len(groups_idx)):
                if _group_ran(gms[j]):
                    _fold_group_spend(metrics, gms[j], _group_key(mems, groups_idx[j]), "abandoned")
            _record_explicit_caches(metrics, (client,), roles=MERGE_ROLES)
            _write_metrics(out_dir, metrics, wall0)
            return metrics, False
        _fold_group_metrics(metrics, gms[i])
        synthesized.extend(res)
    metrics["synthesized_pre_dedup"] = len(synthesized)

    try:
        dedup_groups = global_dedup(synthesized, dedup_prompt, client, cache, metrics)
    except Exception as e:  # a raise after the billed call (a cache write that fails): stop, report
        metrics["fatal_error"] = {"stage": "dedup", "error": f"{type(e).__name__}: {e}"}
        _record_explicit_caches(metrics, (client,), roles=MERGE_ROLES)
        _write_metrics(out_dir, metrics, wall0)
        return metrics, False
    metrics["dedup_groups"] = len(dedup_groups)
    merged_synth = merge_dedup_groups(synthesized, dedup_groups)

    finals = [build_final(s, mems) for s in merged_synth]
    # Rank: most independent support first — distinct conversations, then distinct arms; stable id
    # tie-break keeps the ordering deterministic across runs.
    finals.sort(
        key=lambda f: (
            -f["provenance"]["corroboration"]["distinct_conversations"],
            -f["provenance"]["corroboration"]["distinct_arms"],
            f["memory_id"],
        )
    )
    metrics["final_merged"] = len(finals)

    with open(os.path.join(out_dir, "memories.jsonl"), "w", encoding="utf-8") as f:
        for fm in finals:
            f.write(json.dumps(fm, ensure_ascii=False) + "\n")

    _add_corroboration_metrics(metrics, finals)
    _record_explicit_caches(metrics, (client,), roles=MERGE_ROLES)
    _write_metrics(out_dir, metrics, wall0)
    return metrics, True


def _add_corroboration_metrics(metrics, finals):
    def convs(f):
        return f["provenance"]["corroboration"]["distinct_conversations"]

    def arms(f):
        return f["provenance"]["corroboration"]["distinct_arms"]

    metrics["corroboration"] = {
        "distinct_conversations_ge2": sum(1 for f in finals if convs(f) >= 2),
        "distinct_conversations_ge3": sum(1 for f in finals if convs(f) >= 3),
        "distinct_conversations_eq1": sum(1 for f in finals if convs(f) == 1),
        "distinct_arms_ge2": sum(1 for f in finals if arms(f) >= 2),
        "distinct_arms_ge3": sum(1 for f in finals if arms(f) >= 3),
        "distinct_arms_eq1": sum(1 for f in finals if arms(f) == 1),
    }
    metrics["generality"] = {
        "instance-specific": sum(1 for f in finals if f["generality"] == "instance-specific"),
        "general-principle": sum(1 for f in finals if f["generality"] == "general-principle"),
    }


MERGE_ROLES = {"merge-synthesis", "merge-dedup"}


def _record_explicit_caches(metrics: dict, clients, roles: set[str] | None = None) -> None:
    """Fold the explicit-cache state of the merge's client into `metrics` IN PLACE. Only resources
    whose prompt role is in `roles` count: in `run` the client also holds the arms' caches. A
    resource the client knows no role for counts.

    In place, not into a copy: this stage calls under two prompts (per-group synthesis, then the
    cross-group dedup) and each gets its own decision, so the fields matter — and writing them only
    into the dict that reaches disk left the returned/printed metrics reporting zeros for the same
    run, which is exactly the kind of split that gets read as "the cache did nothing".

    STORAGE is priced PER RESOURCE on that resource's own lifetime (capped at the TTL), not on one
    run-wide elapsed; it stays beside cost_usd_total rather than inside it, since no per-call usage
    block reports it."""
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


def _write_metrics(out_dir, metrics, wall0):
    m = dict(metrics)
    m["wall_clock_s"] = round(time.time() - wall0, 1)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
