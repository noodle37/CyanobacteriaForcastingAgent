"""重新导入适配器后，CPU 基线仍能预测。"""

from __future__ import annotations

import unittest
from pathlib import Path

from tsfm_server.reload import load_fresh_runtime


class ReloadTest(unittest.TestCase):
    def test_fresh_runtime_forecasts_with_naive(self):
        root = Path(__file__).resolve().parents[1]
        runtime = load_fresh_runtime(root / "configs" / "models.yaml")
        out = runtime.forecast("seasonal_naive", [1, 2, 3, 4, 5, 6, 7, 8], 2, "D")
        self.assertEqual(len(out["point"]), 2)
        self.assertEqual(out["model"], "seasonal_naive")


if __name__ == "__main__":
    unittest.main()
