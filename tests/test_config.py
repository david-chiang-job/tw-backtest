import copy
import tomllib
import unittest

from twbacktest.config import ConfigError, parse_config

GOOD = """
[[tiers]]
name = "補倉"
target_stock = 0.9
reset = "new_high"
[tiers.when]
score_at_least = 60
weighted = [
  { points = 50, indicator = "ma_deviation", days = 5, at_most = -0.05 },
  { points = 25, indicator = "ma_deviation", days = 3, at_most = -0.04 },
  { points = 25, indicator = "drawdown", at_most = -0.10 },
]

[other]
x = 1
"""


def good():
    return tomllib.loads(GOOD)


def weighted(d):
    return d["tiers"][0]["when"]


class ConfigChecks(unittest.TestCase):
    def assert_rejected(self, d, *words):
        with self.assertRaises(ConfigError) as cm:
            parse_config(d)
        for w in words:
            self.assertIn(w, str(cm.exception))

    def test_good_config_passes(self):
        cfg = parse_config(good())
        self.assertEqual(cfg.tiers[0].name, "補倉")
        self.assertEqual(cfg.tiers[0].side, 1)
        self.assertEqual(cfg.extra, {"other": {"x": 1}})

    def test_all_at_least_is_sell(self):
        d = good()
        for leaf in weighted(d)["weighted"]:
            leaf["at_least"] = -leaf.pop("at_most")
        self.assertEqual(parse_config(d).tiers[0].side, -1)

    def test_mixed_buy_and_sell_rejected(self):
        d = good()
        leaf = weighted(d)["weighted"][2]
        leaf["at_least"] = leaf.pop("at_most")
        self.assert_rejected(d, "補倉", "買賣混用")

    def test_points_not_positive_rejected(self):
        d = good()
        weighted(d)["weighted"][1]["points"] = 0
        self.assert_rejected(d, "補倉", "weighted[1]", "points")

    def test_total_points_below_threshold_rejected(self):
        d = good()
        weighted(d)["score_at_least"] = 101
        self.assert_rejected(d, "補倉", "到不了門檻")

    def test_total_points_equal_threshold_passes(self):
        d = good()
        weighted(d)["score_at_least"] = 100
        parse_config(d)

    def test_unknown_indicator_lists_names(self):
        d = good()
        weighted(d)["weighted"][0]["indicator"] = "nope"
        self.assert_rejected(d, "nope", "ma_deviation")

    def test_side_key_missing_or_doubled(self):
        d = good()
        del weighted(d)["weighted"][2]["at_most"]
        self.assert_rejected(d, "恰好寫一個")
        d = good()
        weighted(d)["weighted"][2]["at_least"] = 0.1
        self.assert_rejected(d, "恰好寫一個")

    def test_params_checked_against_indicator_signature(self):
        d = good()
        del weighted(d)["weighted"][0]["days"]
        self.assert_rejected(d, "缺參數", "days")
        d = good()
        weighted(d)["weighted"][2]["days"] = 3
        self.assert_rejected(d, "不認得的鍵", "days")

    def test_two_combinator_keys_rejected(self):
        d = good()
        weighted(d)["any"] = [copy.deepcopy(weighted(d)["weighted"][0])]
        self.assert_rejected(d, "補倉")


if __name__ == "__main__":
    unittest.main()
