# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Test-wide guard: no test can reach a real key, a real session store, the network or `claude`.

Every test runs with HOME pointed at a fresh temp dir and GEMINI_API_KEY unset, and with the two
ways this code reaches the outside world (urllib and subprocess) replaced by functions that fail.
A test that needs either patches it explicitly, which overrides the guard for that test only.
Every `tempfile.mkdtemp` lands inside the test's own `tmp_path`, which pytest removes once the test
passes, so a run leaves no synthetic transcripts or caches behind.
"""

from __future__ import annotations

import subprocess
import tempfile
import urllib.request

import pytest


class ExternalCallBlocked(AssertionError):
    pass


def _blocked_urlopen(*args, **kwargs):
    raise ExternalCallBlocked("a test tried to reach the network; patch urllib.request.urlopen")


def _blocked_subprocess(*args, **kwargs):
    raise ExternalCallBlocked("a test tried to start a process; patch subprocess.run")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    monkeypatch.setattr(subprocess, "run", _blocked_subprocess)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    real_mkdtemp = tempfile.mkdtemp

    def _mkdtemp(suffix=None, prefix=None, dir=None):
        return real_mkdtemp(suffix, prefix, dir if dir is not None else str(scratch))

    monkeypatch.setattr(tempfile, "mkdtemp", _mkdtemp)
    import time

    if hasattr(time, "tzset"):
        time.tzset()
    yield home
