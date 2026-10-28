# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Guards for one privacy invariant: verbatim transcript content never lands inside version control.

The pipeline writes memories.jsonl files that carry verbatim conversation quotes. Two guards keep
those outputs out of git:

  - assert_outside_git_repo(path) refuses any output dir that resolves (through symlinks)
    to a location inside a git repo — treating a `.git` DIRECTORY *or* a `.git` FILE as the
    marker, because a worktree's `.git` is a regular file pointing at the real gitdir. A
    directory-only check silently misses a worktree and would let a mistyped --data-dir leak
    transcripts into VCS.
  - sanitize_run_id(run_id) rejects a run_id that could escape the validated data dir
    (empty/whitespace, or containing a path separator or '..'), so `--run-id ../../repo`
    cannot redirect the join out of the guarded tree.
"""

from __future__ import annotations

import contextlib
import fcntl
import os


class RunLockedError(RuntimeError):
    """Another live command holds the lock on this run directory."""


class UnsafePathError(ValueError):
    """An output location that could leak transcript content: inside a git repo, or a run id that
    escapes the data dir."""


def assert_outside_git_repo(path: str) -> None:
    """Raise UnsafePathError if `path` resolves to a location inside a git repo (worktree included). Walk the
    real (symlink-resolved) path's ancestors and stop at filesystem root."""
    real = os.path.realpath(path)
    p = real
    while True:
        dotgit = os.path.join(p, ".git")
        # A worktree's `.git` is a FILE (a gitdir pointer), not a directory — check for either,
        # otherwise the guard misses a worktree and lets transcript content into VCS.
        if os.path.isdir(dotgit) or os.path.isfile(dotgit):
            raise UnsafePathError(
                f"{path} resolves inside a git repo ({p}); refusing to write "
                f"conversation content into version control"
            )
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent


def sanitize_run_id(run_id: str) -> str:
    """Return run_id unchanged if it is a safe single path segment; otherwise raise
    UnsafePathError. Rejects an empty/whitespace id or one carrying a path separator or '..', so
    it cannot escape the validated data dir when joined into the output path."""
    if not run_id or not run_id.strip():
        raise UnsafePathError("--run-id is empty; refusing to build an output path from it")
    if run_id.strip() in (".", "..") or os.sep in run_id or "/" in run_id or ".." in run_id:
        raise UnsafePathError(
            f"--run-id {run_id!r} contains a path separator or '..'; refusing to let it escape the data dir"
        )
    return run_id


@contextlib.contextmanager
def run_lock(out_dir: str, name: str = ".lock"):
    """Hold an advisory lock (file `name` in `out_dir`) for the duration of the block, so two live
    runs cannot interleave writes to the same outputs. A second holder fails at once with
    RunLockedError, a RuntimeError."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunLockedError(f"another live run holds {out_dir}; use a fresh --run-id") from None
        yield
