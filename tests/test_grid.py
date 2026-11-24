import functools
import tomllib
import unittest

import numpy as np

from twbacktest.config import ConfigError
from twbacktest.grid import IndicatorCache, expand, tier_trigger
from twbacktest.indicators import INDICATORS, Indicator
from twbacktest.registry import Registry

TOML = """
[[tiers]]
name = "補倉"
target_stock = { from = 0.5, to = 0.9, step = 0.2 }
reset = "new_high"
[tiers.when]
score_at_least = 60
weighted = [
  { points = 50, indicator = "ma_deviation", days = 5, at_most = { from = -0.03, to = -0.07, step = -0.005 } },
  { points = 25, indicator = "ma_deviation", days = 3, at_most = -0.05 },
  { points = 25, indicator = "drawdown", at_most = -0.10 },
]
"""

P5 = ("tiers", 0, "when", "weighted", 0, "at_most")
PTS = ("tiers", 0, "target_stock")


def counting_registry(counts):
    reg = Registry("指標")
    for name in ("ma_deviation", "drawdown"):
        real = INDICATORS.get(name)

        def make(name, real):
            @functools.wraps(real.base)
            def base(close, **kw):
                counts[(name, tuple(sorted(kw.items())))] = counts.get((name, tuple(sorted(kw.items()))), 0) + 1
                return real.base(close, **kw)

            return base

        reg.register(name)(Indicator(name, make(name, real), real.apply_threshold))
    return reg


def leaves(strategy):
    return strategy.config.tiers[0].when["weighted"]


class RangeExpansion(unittest.TestCase):
    def test_two_ranges_multiply_with_exact_values(self):
        s = expand(tomllib.loads(TOML))
        self.assertEqual(len(s), 27)
        thresholds = [-0.03, -0.035, -0.04, -0.045, -0.05, -0.055, -0.06, -0.065, -0.07]
        # 先出現的範圍（target_stock）變得慢，後面的（5 日門檻）變得快
        self.assertEqual(s[0].picks, {PTS: 0.5, P5: -0.03})
        self.assertEqual(s[1].picks, {PTS: 0.5, P5: -0.035})
        self.assertEqual(s[9].picks, {PTS: 0.7, P5: -0.03})
        self.assertEqual(s[26].picks, {PTS: 0.9, P5: -0.07})
        self.assertEqual([x.picks[P5] for x in s[:9]], thresholds)
        self.assertEqual([x.config.tiers[0].target_stock for x in s[::9]], [0.5, 0.7, 0.9])
        self.assertEqual(leaves(s[4])[0]["at_most"], -0.05)
        self.assertEqual(leaves(s[4])[1]["at_most"], -0.05)  # 固定值不被動到

    def test_bad_ranges_rejected(self):
        for rng in ("{ from = 1, to = 3, step = 0 }", "{ from = 1, to = 3, step = -1 }"):
            d = tomllib.loads(TOML.replace("{ from = 0.5, to = 0.9, step = 0.2 }", rng))
            with self.assertRaises(ConfigError):
                expand(d)

    def test_single_value_range_and_int_values(self):
        d = tomllib.loads(TOML.replace("{ from = 0.5, to = 0.9, step = 0.2 }", "0.5"))
        d["tiers"][0]["when"]["weighted"][1]["days"] = {"from": 3, "to": 4, "step": 1}
        s = expand(d)
        self.assertEqual(len(s), 18)
        self.assertIsInstance(leaves(s[0])[1]["days"], int)

    def test_check_runs_per_group_and_names_it(self):
        d = tomllib.loads(TOML)
        d["tiers"][0]["when"]["weighted"][0]["points"] = {"from": 0, "to": 50, "step": 50}
        with self.assertRaises(ConfigError) as cm:
            expand(d)
        self.assertIn("第 ", str(cm.exception))
        self.assertIn("points", str(cm.exception))


class SharedIndicatorCache(unittest.TestCase):
    close = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    def test_each_base_computed_once_across_all_strategies(self):
        counts = {}
        reg = counting_registry(counts)
        strategies = expand(tomllib.loads(TOML))
        cache = IndicatorCache(self.close, reg)
        for s in strategies:
            for leaf in leaves(s):
                cache.trigger(leaf)
        by_name = {}
        for (name, _), n in counts.items():
            by_name[name] = by_name.get(name, 0) + n
        self.assertEqual(by_name, {"ma_deviation": 2, "drawdown": 1})
        self.assertEqual(sorted(counts.values()), [1, 1, 1])  # 5 日、3 日、drawdown 各一次

    def test_cached_trigger_values_are_correct(self):
        s = expand(tomllib.loads(TOML))[0]  # 5 日門檻 -3%
        cache = IndicatorCache(self.close)
        five, three, dd = (cache.trigger(x) for x in leaves(s))
        self.assertTrue(np.isnan(five[4]))
        self.assertAlmostEqual(five[5], 30.0 * 0.97)  # (10+..+50)/5 = 30
        self.assertAlmostEqual(three[3], 20.0 * 0.95)
        self.assertAlmostEqual(dd[3], 30.0 * 0.9)
        # 另一個門檻共用同一份基準，值要用自己的門檻算
        other = expand(tomllib.loads(TOML))[8]
        self.assertAlmostEqual(cache.trigger(leaves(other)[0])[5], 30.0 * 0.93)

    def test_tier_trigger_combines_leaves(self):
        s = expand(tomllib.loads(TOML))[0]
        got = tier_trigger(s.config.tiers[0], IndicatorCache(self.close))
        # 第 5 天：5 日 29.1(50)、3 日 (40+50+... 前3日=30,40,50)->40*0.95=38(25)、
        # drawdown 50*0.9=45(25)；由高到低 45(25)、38(25)、29.1(50) 累積到 100，第一次 >=60 是 29.1
        self.assertAlmostEqual(got[5], 29.1)
        self.assertTrue(np.isnan(got[0]))


if __name__ == "__main__":
    unittest.main()
