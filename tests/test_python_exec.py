"""python 工具的路径限制和子进程往返。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.python_exec import execute_python, run_user_code


def _run(code: str, root: Path, **kwargs) -> dict:
    raw = dict(
        series=[1.0, 2, 3],
        horizon=2,
        freq="D",
        actual=None,
        forecast=None,
        baseline=None,
        metrics=None,
        read_roots=("tsfm_server", "shared"),
        write_roots=("tsfm_server", "shared"),
    )
    raw.update(kwargs)
    return run_user_code(code, root=root, **raw)


class PythonExecTest(unittest.TestCase):
    def test_rewrite_series_and_block_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            changed = _run("series = [1.0, 2.0, 4.0]\nresult = sum(series)", root)
            self.assertTrue(changed["ok"])
            self.assertEqual(changed["result"], 7.0)
            self.assertEqual(changed["series"], [1.0, 2.0, 4.0])
            blocked = _run("import os\nresult = 1", root)
            self.assertFalse(blocked["ok"])
            self.assertIn("不允许导入", blocked["error"])

    def test_write_file_stays_inside_tsfm_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = _run("result = write_file('tsfm_server/adapters/probe.py', 'VALUE = 1\\n')", root)
            self.assertTrue(written["ok"], written)
            self.assertEqual((root / "tsfm_server/adapters/probe.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            escaped = _run("write_file('../secrets.txt', 'no')", root)
            self.assertFalse(escaped["ok"])
            self.assertFalse((root.parent / "secrets.txt").exists())
            outside = _run("write_file('/tmp/tsfm-agent-escape.py', 'no')", root)
            self.assertFalse(outside["ok"])

    def test_subprocess_roundtrip_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = dict(
                root=root,
                series=[1.0, 2, 3],
                horizon=2,
                freq="D",
                actual=None,
                forecast=None,
                baseline=None,
                metrics=None,
                read_roots=("tsfm_server",),
                write_roots=("tsfm_server",),
            )
            out = execute_python("result = sum(series)\nseries.append(4.0)", timeout_s=10, **common)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["result"], 6)
            self.assertEqual(out["series"][-1], 4.0)
            timed = execute_python("while True:\n    pass", timeout_s=2, **common)
            self.assertFalse(timed["ok"])
            self.assertIn("超过", timed["error"])


if __name__ == "__main__":
    unittest.main()
