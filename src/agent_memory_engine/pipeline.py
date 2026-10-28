# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The whole pipeline over a set of sessions: the `facts` arm, the `preferences` arm, then the merge.

Layout under the data dir (default ~/.agent-memory-engine):

    cache/extractions/         per-window and reconcile results, keyed on their exact inputs
    cache/merge/               merge call results
    runs/<run-id>/
      sessions.jsonl           the selected sessions
      memories.jsonl           the merged memories; each line validates against memory.schema.json
      metrics.json             counts and cost per stage, skipped sessions, any stop
      arms/facts/              memories.jsonl + metrics.json from the facts arm
      arms/preferences/        the same for the preferences arm

Outputs carry verbatim quotes from the sessions, so the data dir is refused inside a git repo.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

from . import contract as mc
from . import extract, merge
from .paths import assert_outside_git_repo, run_lock, sanitize_run_id
from .providers import (
    DEFAULT_THINKING_LEVEL,
    ClaudeNotFoundError,
    ModelClient,
    build_client,
    gemini_cache_storage_cost,
)
from .sources.claude_code import SessionRef

DEFAULT_DATA_DIR = "~/.agent-memory-engine"
ARM_NAMES = tuple(a.name for a in extract.ARMS)
# Conversation-level parallelism per provider: the Gemini REST backend takes concurrent calls; the
# local `claude` CLI runs one at a time.
DEFAULT_WORKERS = {"gemini": 6, "claude": 1}
# Held on runs/<run-id>/ for a whole run or extract, so a second command with the same run id fails
# before it writes anything.
RUN_LOCK = ".run-lock"


@dataclass
class RunResult:
    run_id: str
    run_dir: str
    ok: bool
    metrics: dict = field(default_factory=dict)

    @property
    def memories_path(self) -> str:
        return os.path.join(self.run_dir, "memories.jsonl")


@dataclass
class Clients:
    """The clients one run uses. `extract` may constrain its output with a responseSchema;
    `reconcile` and `merge` must not (their answers have a different shape)."""

    extract: ModelClient
    reconcile: ModelClient
    merge: ModelClient

    def close(self) -> None:
        seen: set[int] = set()
        for c in (self.extract, self.reconcile, self.merge):
            if id(c) in seen:
                continue
            seen.add(id(c))
            c.close()


def make_clients(
    provider: str, *, model: str | None = None, api_key: str | None = None, thinking_level: str | None = None
) -> Clients:
    """The default configuration for a provider. Gemini: constrained decoding of the extraction
    output (structured tags), explicit prompt caching, the given thinking level on every stage.
    Claude: one CLI client for every stage."""
    if provider == "gemini":
        level = thinking_level if thinking_level is not None else DEFAULT_THINKING_LEVEL
        schema = mc.structured_output_schema(sorted(mc.VALID_TYPES))
        extract_client = build_client(
            "gemini",
            model,
            api_key=api_key,
            thinking_level=level,
            response_schema=schema,
            explicit_cache=True,
        )
        plain = build_client("gemini", model, api_key=api_key, thinking_level=level, explicit_cache=True)
        return Clients(extract_client, plain, plain)
    if provider == "claude" and shutil.which("claude") is None:
        raise ClaudeNotFoundError(
            "no claude CLI on PATH: install Claude Code and log in, or use --provider gemini"
        )
    client = build_client(provider, model)
    return Clients(client, client, client)


def new_run_id() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")


def prepare_data_dir(data_dir: str) -> str:
    """Expand the path, refuse a location inside a git repo, and create the data dir owner-only."""
    path = os.path.abspath(os.path.expanduser(data_dir))
    assert_outside_git_repo(path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    # Owner-only whatever the caller's umask: every file below holds verbatim session quotes.
    os.chmod(path, 0o700)
    return path


class RunConflictError(ValueError):
    """The run id already holds output that this command would mix with or overwrite."""


def _arm_paths(run_dir: str) -> dict[str, str]:
    return {arm: os.path.join(run_dir, "arms", arm, "memories.jsonl") for arm in ARM_NAMES}


def _selection_key(row: dict) -> tuple[str, str]:
    """One session of a selection: its id and the file it is read from. Raises ValueError for a
    row without both."""
    cid, path = row["conversation_id"], row["path"]
    if not (isinstance(cid, str) and cid and isinstance(path, str) and path):
        raise ValueError("a session row without an id or a path")
    return cid, path


def run_arms(
    sessions: list[SessionRef],
    clients: Clients,
    *,
    data_dir: str,
    run_id: str,
    arms: tuple[str, ...] = ARM_NAMES,
    workers: int = 1,
    log=None,
) -> dict:
    """Run the given arms over `sessions` into runs/<run_id>/arms/<arm>/. Stops at the first arm
    that stops; returns {arm: metrics} and an `ok` flag under the key "_ok"."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    run_dir = os.path.join(data_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    rows = [s.as_row() for s in sessions]
    sessions_path = os.path.join(run_dir, "sessions.jsonl")
    if os.path.exists(sessions_path):
        # Every arm of a run reads one selection, so the merge never folds two selections together.
        # A selection is its sessions and the files they are read from, in any order.
        try:
            with open(sessions_path, encoding="utf-8") as f:
                held = sorted(_selection_key(json.loads(line)) for line in f if line.strip())
        except (OSError, ValueError, KeyError, TypeError):
            held = []
        if not held:
            raise RunConflictError(f"run {run_id} has an unreadable {sessions_path}; use a fresh --run-id")
        if held != sorted(_selection_key(r) for r in rows):
            raise RunConflictError(
                f"run {run_id} already holds a different session selection; "
                "select the same sessions, or use a fresh --run-id"
            )
    with open(sessions_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out: dict = {"_ok": True}
    for arm in arms:
        log(f"{arm}: extracting from {len(rows)} session(s)")
        metrics, ok = extract.run(
            rows,
            os.path.join(run_dir, "arms", arm),
            clients.extract,
            arm=arm,
            cache_dir=os.path.join(data_dir, "cache", "extractions"),
            workers=workers,
            reconcile_client=clients.reconcile,
            log=log,
        )
        out[arm] = metrics
        if not ok:
            out["_ok"] = False
            break
    return out


def run_merge(clients: Clients, *, data_dir: str, run_id: str, log=None) -> tuple[dict, bool]:
    """Merge runs/<run_id>/arms/*/memories.jsonl into runs/<run_id>/memories.jsonl."""
    run_dir = os.path.join(data_dir, "runs", run_id)
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    log("merge: folding the arms into one set")
    return merge.run(
        _arm_paths(run_dir),
        run_dir,
        clients.merge,
        cache_dir=os.path.join(data_dir, "cache", "merge"),
        log=log,
    )


def arm_stage(m: dict, storage: float | None = None) -> dict:
    """One arm's entry in the run summary. `storage` replaces the arm's own prompt-cache storage
    figure when the run prices storage once across its stages."""
    return {
        "sessions_run": m["conversations_run"],
        "sessions_skipped": m["conversations_skipped"],
        "windows": m["windows_total"],
        "windows_unparseable": m["windows_unparseable"],
        "reconcile_unparseable": m["reconcile_unparseable"],
        "reconcile_overflow": m["reconcile_overflow"],
        "memories": m["memories_total"],
        **_cost_fields(m, storage),
        "fatal_error": m.get("fatal_error"),
    }


def _cost_fields(m: dict, storage: float | None = None) -> dict:
    """A stage's cost: `cost_usd` counts every answer at its price, replayed ones included, plus
    holding the prompt cache. `cached_answers_usd` is the replayed part, and `spent_usd` the rest:
    what this run actually paid. Both parts are taken from the rounded figures, so they add up to
    `cost_usd` to the microdollar."""
    if storage is None:
        storage = m.get("explicit_cache_storage_usd", 0.0)
    cost = round(m["cost_usd_total"] + storage, 6)
    cached = round(m.get("cached_answers_usd", 0.0), 6)
    return {"cost_usd": cost, "cached_answers_usd": cached, "spent_usd": round(cost - cached, 6)}


# The stages that use each preflighted prompt's cache. Reconcile serves both arms.
_PROMPT_STAGES = {
    **{f"extract-{a}": (a,) for a in ARM_NAMES},
    "reconcile": ARM_NAMES,
    "merge-synthesis": ("merge",),
    "merge-dedup": ("merge",),
}


@dataclass
class _Storage:
    """A run's prompt-cache storage: each stage's share in USD and the resources behind it, and the
    storage of prompts no stage that ran used (a stopped run preflighted them for later stages)."""

    usd: dict[str, float]
    caches: dict[str, list[dict]]
    unused_usd: float = 0.0


def _cache_names(clients: Clients) -> set[str]:
    """Every prompt-cache resource the clients have created so far."""
    return {
        e["name"]
        for c in (clients.extract, clients.reconcile, clients.merge)
        if c is not None and getattr(c, "explicit_cache", False)
        for e in getattr(c, "explicit_caches", [])
    }


def _storage_by_stage(clients: Clients, stage_names: list[str], exclude: set[str] | None = None) -> _Storage:
    """The run's prompt-cache storage, each resource priced once on its own lifetime and split over
    the stages that ran and use its prompt. The clients are shared by every stage of a run, so a
    stage's own record can list resources other stages hold too.

    A resource whose prompt no stage that ran uses goes to `unused_usd`, never to another stage. A
    resource with no known role is split over every stage that ran. `exclude` names resources a
    reused client created before this run, which this run does not pay for."""
    out = _Storage({s: 0.0 for s in stage_names}, {s: [] for s in stage_names})
    resources: dict[str, tuple[dict, str | None]] = {}
    for c in (clients.extract, clients.reconcile, clients.merge):
        if c is None or not getattr(c, "explicit_cache", False):
            continue
        roles = getattr(c, "explicit_cache_roles", {})
        for e in getattr(c, "explicit_caches", []):
            if e["name"] not in (exclude or set()):
                resources.setdefault(e["name"], (e, roles.get(e["name"])))
    for entry, role in resources.values():
        usd = gemini_cache_storage_cost(entry["tokens"], entry["lifetime_s"])
        if role is None:
            owners = list(stage_names)
        else:
            owners = [s for s in _PROMPT_STAGES.get(role, ()) if s in out.usd]
        if not owners:
            out.unused_usd += usd
            continue
        for s in owners:
            out.usd[s] += usd / len(owners)
            out.caches[s].append(entry)
    return out


def merge_stage(m: dict, storage: float | None = None, caches: list[dict] | None = None) -> dict:
    """The merge's entry in the run summary: its full metrics, plus the summary fields. `storage`
    and `caches` replace the merge's own prompt-cache figures, as in arm_stage."""
    return {
        **m,
        **({} if storage is None else {"explicit_cache_storage_usd": round(storage, 6)}),
        **({} if caches is None else _cache_fields(caches)),
        "memories_in": m["candidates_in"],
        "memories_merged": m["final_merged"],
        **_cost_fields(m, storage),
        "fatal_error": m.get("fatal_error"),
        "dedup_error": m.get("dedup_error"),
    }


def _cache_fields(caches: list[dict]) -> dict:
    return {"explicit_caches": caches, "explicit_cache_tokens": sum(c["tokens"] for c in caches)}


def _stage_names(arm_metrics: dict, merge_metrics: dict | None) -> list[str]:
    return [a for a in ARM_NAMES if a in arm_metrics] + (["merge"] if merge_metrics is not None else [])


def _summary(
    selected: list[SessionRef],
    arm_metrics: dict,
    merge_metrics: dict | None,
    ok: bool,
    clients: Clients | None = None,
    storage: _Storage | None = None,
) -> dict:
    """The run summary. With `storage`, or `clients` to compute it from, prompt-cache storage is
    priced once for the run instead of taken from each stage's own record."""
    if storage is None and clients is not None:
        storage = _storage_by_stage(clients, _stage_names(arm_metrics, merge_metrics))
    usd = storage.usd if storage is not None else {}
    unused = storage.unused_usd if storage is not None else 0.0
    stages: dict = {}
    skipped: list[dict] = []
    for arm in ARM_NAMES:
        m = arm_metrics.get(arm)
        if m is None:
            continue
        stages[arm] = arm_stage(m, usd.get(arm))
        for s in m.get("skipped_sessions", []):
            if s not in skipped:
                skipped.append(s)
    if merge_metrics is not None:
        caches = storage.caches.get("merge") if storage is not None else None
        stages["merge"] = merge_stage(merge_metrics, usd.get("merge"), caches)
    # Storage no stage that ran used is still billed: it is in the totals and spent.
    total = round(sum(st["cost_usd"] for st in stages.values()) + unused, 6)
    spent = round(sum(st["spent_usd"] for st in stages.values()) + unused, 6)
    cached = round(sum(st["cached_answers_usd"] for st in stages.values()), 6)
    windows = max((stages[a]["windows"] for a in ARM_NAMES if a in stages), default=0)
    return {
        "ok": ok,
        "sessions_selected": len(selected),
        "skipped_sessions": skipped,
        "windows": windows,
        "stages": stages,
        "unused_prompt_cache_usd": round(unused, 6),
        "cost_usd_total": total,
        "spent_usd_total": spent,
        "cached_answers_usd_total": cached,
    }


def _rewrite_arm_storage(run_dir: str, arm_metrics: dict, storage: _Storage) -> None:
    """Put the run's single-count storage share into each arm's metrics.json. An arm writes its file
    before the caches are deleted, and can only see resources other stages share, so its own
    figure is replaced once the run has priced every resource. A cache two arms share appears in
    both lists, and each arm carries half its storage."""
    for arm, m in arm_metrics.items():
        path = os.path.join(run_dir, "arms", arm, "metrics.json")
        fields = {
            **_cache_fields(storage.caches.get(arm, [])),
            "explicit_cache_storage_usd": round(storage.usd.get(arm, 0.0), 6),
        }
        m.update(fields)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                on_disk = json.load(f)
            on_disk.update(fields)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(on_disk, f, indent=2)


def run(
    sessions: list[SessionRef],
    *,
    client: ModelClient | None = None,
    provider: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
    thinking_level: str | None = None,
    workers: int | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    run_id: str | None = None,
    log=None,
) -> RunResult:
    """Extract memories from `sessions` with both arms, merge them, and write the run directory.

    Pass `client` to use your own ModelClient for every stage; otherwise one is built for
    `provider` ("gemini" needs `api_key`; "claude" uses the local `claude` login)."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    if not sessions:
        raise ValueError("no sessions selected; nothing to extract")
    data_dir = prepare_data_dir(data_dir)
    run_id = sanitize_run_id(run_id or new_run_id())
    run_dir = os.path.join(data_dir, "runs", run_id)
    # Build the clients first: a provider refused at setup (no Gemini key, no claude CLI on PATH)
    # leaves no run directory. A claude CLI that is present but not logged in fails at its first
    # call instead, which stops the run.
    if client is not None:
        clients = Clients(client, client, client)
        provider_workers = 1
    else:
        clients = make_clients(provider, model=model, api_key=api_key, thinking_level=thinking_level)
        provider_workers = DEFAULT_WORKERS.get(provider, 1)
    try:
        with run_lock(run_dir, RUN_LOCK):
            return _run_locked(sessions, clients, workers or provider_workers, data_dir, run_id, run_dir, log)
    finally:
        clients.close()


def _run_locked(
    sessions, clients: Clients, workers: int, data_dir: str, run_id: str, run_dir: str, log
) -> RunResult:
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        raise RunConflictError(f"run {run_id} already exists in {data_dir}; use a fresh --run-id")
    if os.path.exists(os.path.join(run_dir, "arms")):
        raise RunConflictError(
            f"run {run_id} already has arm output in {data_dir}; merge it, or use a fresh --run-id"
        )
    merge_metrics = None
    # A reused client still lists the caches of its earlier runs. This run pays only for its own.
    earlier = _cache_names(clients)
    extract.preflight_prompt_caches(
        [
            (
                clients.extract,
                [(f"extract-{a.name}", extract.load_prompt(a.prompt_file)) for a in extract.ARMS],
            ),
            (clients.reconcile, [("reconcile", extract.load_reconcile_prompt())]),
            (
                clients.merge,
                [
                    ("merge-synthesis", merge.load_prompt("SYNTHESIS")),
                    ("merge-dedup", merge.load_prompt("DEDUP")),
                ],
            ),
        ],
        log=log,
    )
    arm_metrics = run_arms(sessions, clients, data_dir=data_dir, run_id=run_id, workers=workers, log=log)
    ok = arm_metrics.pop("_ok")
    if ok:
        merge_metrics, ok = run_merge(clients, data_dir=data_dir, run_id=run_id, log=log)
    # Delete the prompt caches before the summary prices them, so each is billed to its deletion.
    # run() closes the clients again on every path; a second close is a no-op.
    clients.close()
    storage = _storage_by_stage(clients, _stage_names(arm_metrics, merge_metrics), exclude=earlier)
    summary = _summary(sessions, arm_metrics, merge_metrics, ok, storage=storage)
    _rewrite_arm_storage(run_dir, arm_metrics, storage)
    summary["run_id"] = run_id
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return RunResult(run_id=run_id, run_dir=run_dir, ok=ok, metrics=summary)
