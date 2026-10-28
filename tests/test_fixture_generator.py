# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The storefront session generator's --check mode: the committed copies match a fresh generation,
and a tampered or missing copy fails the check."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "make_storefront_session", os.path.join(ROOT, "tests", "fixtures", "make_storefront_session.py")
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _check(root: str) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = gen.main(["--check", "--root", root])
    return code, err.getvalue()


def _copy_of_the_committed_files() -> str:
    root = tempfile.mkdtemp()
    for rel in gen.COPIES:
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(root, rel))
    return root


class Check(unittest.TestCase):
    def test_the_committed_copies_pass(self):
        self.assertEqual(_check(ROOT), (0, ""))

    def test_a_tampered_copy_fails_and_is_named(self):
        root = _copy_of_the_committed_files()
        tampered = os.path.join(root, gen.COPIES[1])
        with open(tampered, "a", encoding="utf-8") as f:
            f.write("\n")
        code, err = _check(root)
        self.assertEqual(code, 1)
        self.assertIn(gen.COPIES[1], err)
        self.assertNotIn(gen.COPIES[0], err)

    def test_a_missing_copy_fails(self):
        root = _copy_of_the_committed_files()
        os.remove(os.path.join(root, gen.COPIES[0]))
        code, err = _check(root)
        self.assertEqual(code, 1)
        self.assertIn("missing: " + gen.COPIES[0], err)

    def test_stdout_mode_writes_the_committed_bytes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(gen.main([]), 0)
        with open(os.path.join(ROOT, gen.COPIES[0]), encoding="utf-8") as f:
            self.assertEqual(out.getvalue(), f.read())
