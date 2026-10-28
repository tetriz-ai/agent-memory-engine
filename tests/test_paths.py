# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Run ids that would escape the data dir, and the run lock."""

from __future__ import annotations

import tempfile
import unittest

from agent_memory_engine.paths import UnsafePathError, run_lock, sanitize_run_id


class RunIds(unittest.TestCase):
    def test_table(self):
        for bad in ("", " ", "a/b", "..", ".", "../x", "a..b"):
            with self.subTest(run_id=bad), self.assertRaises(UnsafePathError):
                sanitize_run_id(bad)
        for good in ("2026-10-13T101502", "shop-sept", "a.b"):
            self.assertEqual(sanitize_run_id(good), good)


class RunLock(unittest.TestCase):
    def test_a_held_lock_refuses_a_second_holder(self):
        d = tempfile.mkdtemp()
        with run_lock(d), self.assertRaises(RuntimeError), run_lock(d):
            pass
        with run_lock(d):  # released on exit
            pass
