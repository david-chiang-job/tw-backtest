import unittest

import numpy as np

from twbacktest.indicators import INDICATORS
from twbacktest.registry import Registry, UnknownComponentError


class MaDeviation(unittest.TestCase):
    def test_average_excludes_today_and_short_history_is_nan(self):
        close = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ind = INDICATORS.get("ma_deviation")
        # days=3：第 3 天 = (10+20+30)/3 = 20，第 4 天 = (20+30+40)/3 = 30
        trig = ind.trigger(close, -0.05, days=3)
        self.assertTrue(np.isnan(trig[:3]).all())
        self.assertAlmostEqual(trig[3], 20.0 * 0.95)
        self.assertAlmostEqual(trig[4], 30.0 * 0.95)

    def test_one_base_serves_many_thresholds(self):
        ind = INDICATORS.get("ma_deviation")
        base = ind.base(np.array([10.0, 20.0, 30.0, 40.0]), days=2)
        self.assertAlmostEqual(ind.apply_threshold(base, -0.1)[2], 15.0 * 0.9)
        self.assertAlmostEqual(ind.apply_threshold(base, 0.2)[2], 15.0 * 1.2)


class RegistryChecks(unittest.TestCase):
    def test_unknown_name_lists_available(self):
        with self.assertRaises(UnknownComponentError) as cm:
            INDICATORS.get("nope")
        self.assertIn("nope", str(cm.exception))
        self.assertIn("ma_deviation", str(cm.exception))

    def test_duplicate_name_rejected(self):
        r = Registry("x")
        r.register("a")(1)
        with self.assertRaises(ValueError):
            r.register("a")(2)




class Drawdown(unittest.TestCase):
    def test_peak_excludes_today_and_day_zero_is_nan(self):
        close = np.array([10.0, 30.0, 20.0, 25.0, 40.0])
        trig = INDICATORS.get("drawdown").trigger(close, -0.1)
        self.assertTrue(np.isnan(trig[0]))
        # 第 1 天最高 10；第 2 天最高 30；第 4 天最高仍是 30（40 是當天，不含）
        self.assertAlmostEqual(trig[1], 10.0 * 0.9)
        self.assertAlmostEqual(trig[2], 30.0 * 0.9)
        self.assertAlmostEqual(trig[4], 30.0 * 0.9)


if __name__ == "__main__":
    unittest.main()
