# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Find the user's local Claude Code sessions and select them by date and project.

Claude Code stores one session per file at `<claude-home>/projects/<encoded-dir>/<session-uuid>.jsonl`,
one JSON event per line. `<encoded-dir>` is the session's working directory with every character
other than a letter or digit replaced by `-`. Sub-agent transcripts live one level deeper
(`<session-uuid>/subagents/agent-*.jsonl`); they belong to their parent session and are not listed.

Selection reads only structural metadata: the first event timestamp and the working directory
(`cwd`) the events carry. Nothing here calls a model.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
from dataclasses import asdict, dataclass

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")

DEFAULT_CLAUDE_HOME = "~/.claude"


@dataclass(frozen=True)
class SessionRef:
    """One selectable session. `date` is the first event's timestamp (ISO 8601), or the file's
    modification time when no event carries one; `project` is the working directory the events
    record, or the encoded folder name when none does (`project_from_cwd` says which)."""

    conversation_id: str
    native_id: str
    path: str
    date: str | None
    project: str
    project_from_cwd: bool
    tool: str = "claude-code"
    source: str = "local"

    def as_row(self) -> dict:
        return asdict(self)


def encode_project_dir(path: str) -> str:
    """The folder name Claude Code gives a working directory."""
    return _NON_ALNUM_RE.sub("-", path)


def parse_iso(s: str | None) -> _dt.datetime | None:
    """An ISO 8601 timestamp as a timezone-aware datetime; a zone-less value is taken as UTC."""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return dt


def _scan_session(path: str) -> tuple[_dt.datetime | None, str | None]:
    """(first event timestamp, first `cwd`) from one session file, falling back to the file's
    modification time for the date. Unreadable lines are skipped."""
    first_ts: _dt.datetime | None = None
    cwd: str | None = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(o, dict):
                    continue
                if first_ts is None:
                    first_ts = parse_iso(o.get("timestamp"))
                if cwd is None and isinstance(o.get("cwd"), str) and o["cwd"]:
                    cwd = o["cwd"]
                if first_ts is not None and cwd is not None:
                    break
    except OSError:
        pass
    if first_ts is None:
        try:
            first_ts = _dt.datetime.fromtimestamp(os.path.getmtime(path), _dt.UTC)
        except OSError:
            first_ts = None
    return first_ts, cwd


def _normalize_project(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    return p.rstrip(os.sep) or os.sep


def project_matches(ref: SessionRef, project: str) -> bool:
    """True when the session ran in `project` or below it. Matched on the recorded working
    directory as a path prefix (whole path components). A session whose events carry no `cwd` is
    matched on its encoded folder name, exactly. The encoding maps every character other than a
    letter or digit to `-`, so a prefix test would also take sibling folders such as `shop-admin`
    for `shop`; such a session is matched only for its own folder, never for a parent. Distinct
    paths can still encode to the same folder name."""
    want = _normalize_project(project)
    if ref.project_from_cwd:
        have = ref.project.rstrip(os.sep) or os.sep
        return have == want or have.startswith(want.rstrip(os.sep) + os.sep)
    enc = encode_project_dir(want)
    return ref.project == enc


def _local_date(dt: _dt.datetime) -> _dt.date:
    return dt.astimezone().date()


def list_sessions(
    claude_home: str = DEFAULT_CLAUDE_HOME,
    since: _dt.date | None = None,
    until: _dt.date | None = None,
    projects: list[str] | None = None,
) -> list[SessionRef]:
    """Every top-level Claude Code session under `claude_home`, oldest first.

    `since` is inclusive and `until` exclusive, compared with the calendar day (local time zone) of
    the session's first event. `projects` keeps sessions that ran in any of the given directories
    or below them; None or empty keeps all. Raises FileNotFoundError when `claude_home` has no
    `projects` directory, so a mistyped home is never read as "no sessions"."""
    if since is not None and until is not None and since >= until:
        raise ValueError(f"--since {since} must be before --until {until}")
    projects_root = os.path.join(os.path.expanduser(claude_home), "projects")
    if not os.path.isdir(projects_root):
        raise FileNotFoundError(f"no Claude Code sessions directory at {projects_root}")
    refs: list[tuple[_dt.datetime | None, SessionRef]] = []
    for proj_dir in sorted(glob.glob(os.path.join(glob.escape(projects_root), "*"))):
        if not os.path.isdir(proj_dir):
            continue
        encoded = os.path.basename(proj_dir)
        for f in sorted(glob.glob(os.path.join(glob.escape(proj_dir), "*.jsonl"))):
            stem = os.path.basename(f)[: -len(".jsonl")]
            m = UUID_RE.fullmatch(stem)
            if not m:
                continue
            uuid = m.group(0)
            ts, cwd = _scan_session(f)
            ref = SessionRef(
                conversation_id=f"cc:{uuid}",
                native_id=uuid,
                path=f,
                date=ts.isoformat() if ts else None,
                project=cwd or encoded,
                project_from_cwd=cwd is not None,
            )
            if since is not None and (ts is None or _local_date(ts) < since):
                continue
            if until is not None and (ts is None or _local_date(ts) >= until):
                continue
            if projects and not any(project_matches(ref, p) for p in projects):
                continue
            refs.append((ts, ref))
    epoch = _dt.datetime.min.replace(tzinfo=_dt.UTC)
    refs.sort(key=lambda tr: (tr[0] or epoch, tr[1].native_id))
    return [r for _, r in refs]
