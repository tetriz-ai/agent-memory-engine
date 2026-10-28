# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The `tetriz-agent-memory-engine` command: sessions, run, extract, merge.

Exit codes: 0 success; 1 a model failure stopped the run (partial outputs and metrics.json are
written); 2 a usage or setup error before any model call: a bad flag, no Gemini key, no claude CLI
on PATH, no sessions directory, no matching session, a data dir inside a git repo, a run id another
live run holds, or a run id that already holds other output.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

from . import __version__, extract, pipeline
from .paths import RunLockedError, UnsafePathError, run_lock
from .providers import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PROVIDER,
    PROVIDERS,
    ClaudeCLIClient,
    ClaudeNotFoundError,
    GeminiKeyError,
    load_gemini_key,
)
from .providers.gemini import DEFAULT_THINKING_LEVEL, GEMINI_THINKING_LEVELS
from .sources.claude_code import DEFAULT_CLAUDE_HOME, SessionRef, list_sessions, parse_iso
from .transcript import resolve_with_reason

EXIT_OK, EXIT_STOPPED, EXIT_USAGE = 0, 1, 2


class UsageError(Exception):
    pass


def _date(value: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date (YYYY-MM-DD): {value!r}") from None


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def _add_selection(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("session selection")
    g.add_argument(
        "--since", type=_date, metavar="DATE", help="first day to include, YYYY-MM-DD (local time, inclusive)"
    )
    g.add_argument(
        "--until", type=_date, metavar="DATE", help="day to stop before, YYYY-MM-DD (local time, exclusive)"
    )
    g.add_argument(
        "--project",
        action="append",
        metavar="PATH",
        default=[],
        help="only sessions whose working directory is PATH or below it; repeatable",
    )
    g.add_argument(
        "--claude-home",
        default=DEFAULT_CLAUDE_HOME,
        metavar="DIR",
        help=f"Claude Code's home directory (default {DEFAULT_CLAUDE_HOME})",
    )
    g.add_argument(
        "--limit", type=_positive_int, metavar="N", help="only the N most recent selected sessions"
    )


def _add_provider(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model provider")
    g.add_argument(
        "--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER, help=f"default {DEFAULT_PROVIDER}"
    )
    g.add_argument(
        "--model",
        metavar="ID",
        help="override the provider's pinned model "
        f"({DEFAULT_GEMINI_MODEL} or {ClaudeCLIClient.DEFAULT_MODEL})",
    )
    g.add_argument(
        "--thinking-level",
        choices=sorted(GEMINI_THINKING_LEVELS),
        help=f"Gemini reasoning level (default {DEFAULT_THINKING_LEVEL}); Gemini only",
    )
    g.add_argument(
        "--gemini-key-file",
        metavar="PATH",
        help="read the Gemini API key from PATH instead of GEMINI_API_KEY",
    )


def _add_output(p: argparse.ArgumentParser, run_id_required: bool = False) -> None:
    g = p.add_argument_group("output")
    g.add_argument(
        "--data-dir",
        default=pipeline.DEFAULT_DATA_DIR,
        metavar="DIR",
        help=f"where caches and runs are written (default {pipeline.DEFAULT_DATA_DIR})",
    )
    g.add_argument(
        "--run-id",
        required=run_id_required,
        metavar="ID",
        help="name of the run directory" + ("" if run_id_required else " (default: the current time)"),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tetriz-agent-memory-engine",
        description="Extract typed, tagged, evidence-checked memories from your local Claude Code sessions.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser(
        "sessions",
        help="list the sessions a selection matches (no model call)",
        description="List the Claude Code sessions a selection matches, with the turns and model windows "
        "each one holds. No model is called.",
    )
    _add_selection(p)

    p = sub.add_parser(
        "run",
        help="extract facts and preferences, then merge them into one memories.jsonl",
        description="Extract memories from the selected sessions with both arms, merge them, and write "
        "memories.jsonl and metrics.json to a new run directory.",
    )
    _add_selection(p)
    _add_provider(p)
    p.add_argument(
        "--workers",
        type=_positive_int,
        metavar="N",
        help=f"sessions processed at once (default 6 for gemini, 1 for claude; at most "
        f"{extract.CONV_MAX_WORKERS})",
    )
    _add_output(p)

    p = sub.add_parser(
        "extract",
        help="run one arm only",
        description="Run one extraction arm over the selected sessions into the run directory. Give "
        "both arms the same --run-id and selection, then merge them.",
    )
    p.add_argument(
        "--arm",
        required=True,
        choices=pipeline.ARM_NAMES,
        help="which arm to run; give both extract calls the same --run-id so merge can find them",
    )
    _add_selection(p)
    _add_provider(p)
    p.add_argument(
        "--workers",
        type=_positive_int,
        metavar="N",
        help=f"sessions processed at once (default 6 for gemini, 1 for claude; at most "
        f"{extract.CONV_MAX_WORKERS})",
    )
    _add_output(p)

    p = sub.add_parser(
        "merge",
        help="merge the arms of an existing run",
        description="Merge the two arms of an existing run into one memories.jsonl. This calls the "
        "model, so it needs the same provider and key as the extraction.",
    )
    _add_provider(p)
    _add_output(p, run_id_required=True)
    return ap


def _select(args) -> list[SessionRef]:
    try:
        refs = list_sessions(
            args.claude_home, since=args.since, until=args.until, projects=args.project or None
        )
    except FileNotFoundError as e:
        raise UsageError(f"{e}; check --claude-home") from None
    except ValueError as e:
        raise UsageError(str(e)) from None
    if args.limit:
        refs = refs[-args.limit :]
    return refs


def _check_provider_flags(args) -> None:
    if args.provider != "gemini":
        if args.thinking_level is not None:
            raise UsageError("--thinking-level is a Gemini setting; the claude provider has none")
        if args.gemini_key_file is not None:
            raise UsageError("--gemini-key-file only applies to --provider gemini")


def _api_key(args) -> str | None:
    if args.provider != "gemini":
        return None
    try:
        return load_gemini_key(args.gemini_key_file)
    except GeminiKeyError as e:
        raise UsageError(str(e)) from None


def _tilde(p: str) -> str:
    """`p` with the home directory shown as `~`."""
    home = os.path.expanduser("~")
    for h in dict.fromkeys((home, os.path.realpath(home))):
        if p == h or p.startswith(h + os.sep):
            return "~" + p[len(h) :]
    return p


def _short_project(ref: SessionRef) -> str:
    return _tilde(ref.project) if ref.project_from_cwd else ref.project


def cmd_sessions(args, out) -> int:
    refs = _select(args)
    rows = []
    total_windows = 0
    for ref in refs:
        conv, reason = resolve_with_reason(ref.as_row())
        turns = len(conv.turns) if conv else 0
        windows = len(extract.plan_windows(conv)) if conv else 0
        total_windows += windows
        when = parse_iso(ref.date)
        day = when.astimezone().date().isoformat() if when else "-"  # the local day selection uses
        rows.append((day, _short_project(ref), ref.native_id, turns, windows, reason))
    width = max([len("PROJECT")] + [len(r[1]) for r in rows])
    out.write(f"{'DATE':<10}  {'PROJECT':<{width}}  {'SESSION':<36}  {'TURNS':>5}  {'WINDOWS':>7}\n")
    for d, proj, sid, turns, windows, reason in rows:
        mark = f"  skipped: {reason}" if reason else ""
        out.write(f"{d:<10}  {proj:<{width}}  {sid:<36}  {turns:>5}  {windows:>7}{mark}\n")
    out.write(
        f"{len(rows)} session{'s' if len(rows) != 1 else ''}, {total_windows} "
        f"window{'s' if total_windows != 1 else ''}.\n"
    )
    return EXIT_OK


_cost = extract.format_cost


def _arm_line(arm: str, s: dict) -> str:
    """One arm's summary line: sessions run and skipped, unparseable windows, memories, cost, and
    the reconcile fallbacks and stop reason when there are any."""
    line = (
        f"{arm:<12} {s['sessions_run']} sessions, {s['sessions_skipped']} skipped, "
        f"{s['windows_unparseable']} unparseable windows, {s['memories']} memories, "
        f"{_cost(s['cost_usd'], s.get('spent_usd'))}"
    )
    extras = []
    if s.get("reconcile_unparseable"):
        extras.append(f"{s['reconcile_unparseable']} unparseable reconciles kept unreconciled")
    if s.get("reconcile_overflow"):
        extras.append(f"{s['reconcile_overflow']} candidates past the reconcile budget kept unreconciled")
    if extras:
        line += f" ({'; '.join(extras)})"
    if s.get("fatal_error"):
        line += f"  STOPPED: {s['fatal_error'].get('error')}"
    return line


def _merge_line(m: dict) -> str:
    line = (
        f"{'merge':<12} {m['memories_in']} in, {m['memories_merged']} merged, "
        f"{_cost(m['cost_usd'], m.get('spent_usd'))}"
    )
    if m.get("dedup_error"):
        line += f"  dedup failed: {m['dedup_error']}"
    if m.get("fatal_error"):
        line += f"  STOPPED: {m['fatal_error'].get('error')}"
    return line


def _print_summary(summary: dict, run_dir: str, out) -> None:
    stages = summary["stages"]
    for arm in pipeline.ARM_NAMES:
        s = stages.get(arm)
        if s is not None:
            out.write(_arm_line(arm, s) + "\n")
    m = stages.get("merge")
    if m is not None:
        out.write(_merge_line(m) + "\n")
    if summary["ok"]:
        out.write(
            f"Wrote {_tilde(os.path.join(run_dir, 'memories.jsonl'))} "
            f"({m['memories_merged'] if m else 0} memories), "
            f"total {_cost(summary['cost_usd_total'], summary.get('spent_usd_total'))}\n"
        )
    else:
        out.write(f"Stopped. Partial outputs and metrics.json are in {_tilde(run_dir)}\n")


def cmd_run(args, out) -> int:
    _check_provider_flags(args)
    refs = _select(args)
    if not refs:
        raise UsageError("no sessions match the selection")
    key = _api_key(args)
    try:
        result = pipeline.run(
            refs,
            provider=args.provider,
            api_key=key,
            model=args.model,
            thinking_level=args.thinking_level,
            workers=args.workers,
            data_dir=args.data_dir,
            run_id=args.run_id,
        )
    except ValueError as e:
        raise UsageError(str(e)) from None
    _print_summary(result.metrics, result.run_dir, out)
    return EXIT_OK if result.ok else EXIT_STOPPED


def cmd_extract(args, out) -> int:
    _check_provider_flags(args)
    refs = _select(args)
    if not refs:
        raise UsageError("no sessions match the selection")
    key = _api_key(args)
    data_dir = pipeline.prepare_data_dir(args.data_dir)
    run_id = pipeline.sanitize_run_id(args.run_id or pipeline.new_run_id())
    run_dir = os.path.join(data_dir, "runs", run_id)
    clients = pipeline.make_clients(
        args.provider, model=args.model, api_key=key, thinking_level=args.thinking_level
    )
    try:
        with run_lock(run_dir, pipeline.RUN_LOCK):
            res = pipeline.run_arms(
                refs,
                clients,
                data_dir=data_dir,
                run_id=run_id,
                arms=(args.arm,),
                workers=args.workers or pipeline.DEFAULT_WORKERS[args.provider],
            )
    except pipeline.RunConflictError as e:
        raise UsageError(str(e)) from None
    finally:
        clients.close()
    s = pipeline.arm_stage(res[args.arm])
    out.write(_arm_line(args.arm, s) + "\n")
    if res["_ok"]:
        out.write(f"Wrote {_tilde(os.path.join(run_dir, 'arms', args.arm, 'memories.jsonl'))}\n")
        return EXIT_OK
    arm_dir = _tilde(os.path.join(run_dir, "arms", args.arm))
    out.write(f"Stopped. Partial outputs and metrics.json are in {arm_dir}\n")
    return EXIT_STOPPED


def cmd_merge(args, out) -> int:
    _check_provider_flags(args)
    key = _api_key(args)
    data_dir = pipeline.prepare_data_dir(args.data_dir)
    run_id = pipeline.sanitize_run_id(args.run_id)
    run_dir = os.path.join(data_dir, "runs", run_id)
    missing = [p for p in pipeline._arm_paths(run_dir).values() if not os.path.exists(p)]
    if missing:
        raise UsageError(f"run {run_id} has no arm output at: {', '.join(missing)}")
    clients = pipeline.make_clients(
        args.provider, model=args.model, api_key=key, thinking_level=args.thinking_level
    )
    try:
        with run_lock(run_dir, pipeline.RUN_LOCK):
            m, ok = pipeline.run_merge(clients, data_dir=data_dir, run_id=run_id)
    finally:
        clients.close()
    out.write(_merge_line(pipeline.merge_stage(m)) + "\n")
    if ok:
        out.write(f"Wrote {_tilde(os.path.join(run_dir, 'memories.jsonl'))}\n")
        return EXIT_OK
    out.write(f"Stopped. Partial outputs and metrics.json are in {_tilde(run_dir)}\n")
    return EXIT_STOPPED


COMMANDS = {"sessions": cmd_sessions, "run": cmd_run, "extract": cmd_extract, "merge": cmd_merge}


def main(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    os.umask(0o077)  # outputs carry verbatim quotes: keep them owner-only
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, out)
    except (UsageError, UnsafePathError, RunLockedError, ClaudeNotFoundError) as e:
        print(f"tetriz-agent-memory-engine: error: {e}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
