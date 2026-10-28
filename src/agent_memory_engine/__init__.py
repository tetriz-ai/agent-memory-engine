# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Extract typed, tagged, evidence-checked memories from local Claude Code sessions.

    import datetime
    import os

    from agent_memory_engine import list_sessions, run

    sessions = list_sessions(since=datetime.date(2026, 9, 1))
    result = run(sessions, api_key=os.environ["GEMINI_API_KEY"])
    print(result.memories_path)

`run` takes any `ModelClient` through `client=`.
"""

__version__ = "0.1.0"

from .pipeline import RunResult, run
from .providers import ExtractResult, ModelClient
from .sources import SessionRef, list_sessions

__all__ = ["ExtractResult", "ModelClient", "RunResult", "SessionRef", "__version__", "list_sessions", "run"]
