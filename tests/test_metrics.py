import unittest

from shared.metrics import evaluate_forecast


class MetricsTest(unittest.TestCase):
    def test_perfect_forecast(self):
        scored = evaluate_forecast(
            point=[1, 2, 3],
            actual=[1, 2, 3],
            history=[0, 1, 2, 3, 4],
            q10=[1, 2, 3],
            q90=[1, 2, 3],
            season=1,
        )
        self.assertEqual(scored["mae"], 0)
        self.assertEqual(scored["mase"], 0)
        self.assertEqual(scored["coverage"], 1)
        self.assertEqual(scored["crps_proxy"], 0)

    def test_short_history_has_no_mase(self):
        scored = evaluate_forecast([2], [1], [1], season=7)
        self.assertIsNone(scored["mase"])
        self.assertEqual(scored["mae"], 1)


if __name__ == "__main__":
    unittest.main()
