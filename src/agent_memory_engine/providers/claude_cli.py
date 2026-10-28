# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""ClaudeCLIClient: the local `claude -p` login as a model transport.

The call runs in a fully controlled shape, so the model sees only this tool's prompt:

    claude -p --model <id> --output-format json
              --system-prompt-file <our prompt>         # replaces the agent prompt
              --exclude-dynamic-system-prompt-sections  # drops harness-injected sections
              --disallowedTools "*"                     # text in, text out, no tools
              --no-session-persistence                  # the call is not saved as a session

It runs from a neutral working directory (the system temp dir), so no project CLAUDE.md or settings
are injected into the call. The model id is pinned to a concrete version, never a moving alias, so
two runs never mix underlying models. Cost and duration come from the CLI's own JSON report.

Without --no-session-persistence each call would be saved as a Claude Code session in the user's
session store, where a later run would read it back as input. The flag needs a `claude` CLI that
supports it.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time

from .base import ExtractResult, ModelClient


class ClaudeNotFoundError(RuntimeError):
    """No `claude` CLI on PATH: a setup error, raised before any run starts."""


class ClaudeCLIClient(ModelClient):
    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        claude_bin: str = "claude",
        neutral_cwd: str | None = None,
        timeout_s: int = 900,
        max_attempts: int = 3,
    ):
        self.model_id = model_id
        self.cache_model_id = model_id  # no extra output-shaping knob to fold in
        self.claude_bin = claude_bin
        self.neutral_cwd = neutral_cwd or tempfile.gettempdir()
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts

    def extract(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        """Retry transient failures (spawn errors, non-zero exit) with a short backoff. A timeout is
        terminal: deterministic for a given input, so retrying it only burns the timeout again."""
        last = None
        spent = 0.0  # every attempt the CLI reports a cost for was billed, failed or not
        for attempt in range(1, self.max_attempts + 1):
            last = self._extract_once(system_prompt, conversation_text)
            spent += last.cost_usd
            done = last.ok or last.error == "timeout" or attempt == self.max_attempts
            if done:
                last.cost_usd = spent
                last.meta["http_retries"] = attempt - 1
                return last
            time.sleep(2 * attempt)
        return last or ExtractResult("", 0.0, 0, self.model_id, ok=False, error="no-attempts")

    def build_command(self, system_prompt_file: str) -> list[str]:
        return [
            self.claude_bin,
            "-p",
            "--model",
            self.model_id,
            "--output-format",
            "json",
            "--system-prompt-file",
            system_prompt_file,
            "--exclude-dynamic-system-prompt-sections",
            "--disallowedTools",
            "*",
            "--no-session-persistence",
        ]

    def _extract_once(self, system_prompt: str, conversation_text: str) -> ExtractResult:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as sf:
            sf.write(system_prompt)
            sysfile = sf.name
        cmd = self.build_command(sysfile)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                input=conversation_text,
                capture_output=True,
                text=True,
                cwd=self.neutral_cwd,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExtractResult(
                "", 0.0, int((time.time() - t0) * 1000), self.model_id, ok=False, error="timeout"
            )
        except Exception as e:  # transport failure (binary missing, etc.)
            return ExtractResult(
                "", 0.0, int((time.time() - t0) * 1000), self.model_id, ok=False, error=f"spawn: {e}"
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(sysfile)
        if proc.returncode != 0:
            # The CLI's own error verbatim: auth, a disabled subscription and rate limits land here.
            err = (proc.stderr or proc.stdout or "").strip()
            return ExtractResult(
                "",
                0.0,
                int((time.time() - t0) * 1000),
                self.model_id,
                ok=False,
                error=f"exit {proc.returncode}: {err[:2000]}",
            )
        try:
            d = json.loads(proc.stdout)
        except Exception as e:
            return ExtractResult(
                "",
                0.0,
                int((time.time() - t0) * 1000),
                self.model_id,
                ok=False,
                error=f"cli-json-parse: {e}: {proc.stdout[:500]}",
            )
        if not isinstance(d, dict):
            return ExtractResult(
                "",
                0.0,
                int((time.time() - t0) * 1000),
                self.model_id,
                ok=False,
                error=f"cli-json-shape: expected a JSON object, got {type(d).__name__}",
            )
        if d.get("is_error"):
            return ExtractResult(
                "",
                float(d.get("total_cost_usd") or 0),
                int(d.get("duration_ms") or 0),
                self.model_id,
                ok=False,
                error=f"cli-is_error: {d.get('result')}",
            )
        canonical = next(iter((d.get("modelUsage") or {}).keys()), None)
        return ExtractResult(
            raw_text=d.get("result") or "",
            cost_usd=float(d.get("total_cost_usd") or 0),
            duration_ms=int(d.get("duration_ms") or 0),
            model_id=self.model_id,
            ok=True,
            meta={"canonical_model": canonical, "usage": d.get("usage"), "session_id": d.get("session_id")},
        )
