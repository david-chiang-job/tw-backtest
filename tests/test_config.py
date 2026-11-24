import copy
import tomllib
import unittest

from twbacktest.config import Account, ConfigError, Costs, parse_config, shared_account

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

    def test_fill_defaults_and_params(self):
        cfg = parse_config(good())
        self.assertEqual((cfg.fill.model, cfg.fill.same_day, cfg.fill.params), ("touch", "buy_first", {}))
        d = good()
        d["fill"] = {"model": "touch", "same_day": "sell_first", "band": [0.2, 0.8]}
        cfg = parse_config(d)
        self.assertEqual((cfg.fill.same_day, cfg.fill.params), ("sell_first", {"band": [0.2, 0.8]}))
        self.assertNotIn("fill", cfg.extra)
        d["fill"] = {"model": "close_next_open"}
        self.assertEqual(parse_config(d).fill.model, "close_next_open")

    def test_bad_fill_rejected(self):
        d = good()
        d["fill"] = {"model": "nope"}
        self.assert_rejected(d, "nope", "touch")
        d["fill"] = {"same_day": "whenever"}
        self.assert_rejected(d, "same_day", "whenever")
        d["fill"] = {"model": "touch", "bnad": [0.1, 0.9]}
        self.assert_rejected(d, "不認得的鍵", "bnad")
        d["fill"] = {"model": "close_next_open", "band": [0.1, 0.9]}
        self.assert_rejected(d, "不認得的鍵", "band")

    def test_unknown_reset_rejected_known_pass(self):
        d = good()
        d["tiers"][0]["reset"] = "nope"
        self.assert_rejected(d, "補倉", "reset", "nope", "new_high")
        for name in ("recross", "new_high", "none"):
            d["tiers"][0]["reset"] = name
            self.assertEqual(parse_config(d).tiers[0].reset, name)

    def test_target_stock_range(self):
        d = good()
        for bad in (-0.1, 1.5):
            d["tiers"][0]["target_stock"] = bad
            self.assert_rejected(d, "補倉", "target_stock")
        for ok in (0, 1, 0.5):
            d["tiers"][0]["target_stock"] = ok
            self.assertEqual(parse_config(d).tiers[0].target_stock, ok)

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


class AccountAndCosts(unittest.TestCase):
    def parse(self, **extra):
        d = good()
        d.update(extra)
        return parse_config(d)

    def test_defaults_and_given_values(self):
        c = self.parse()
        self.assertEqual(c.account, Account())
        self.assertEqual(c.costs, Costs())
        self.assertNotIn("account", c.extra)
        self.assertIn("other", c.extra)  # 對照：別的區段仍原樣留在 extra
        c = self.parse(account={"cash": 1_000_000, "cash_rate": 0.02}, costs={"sell_tax": 0.003})
        self.assertEqual((c.account.cash, c.account.cash_rate), (1_000_000, 0.02))
        self.assertEqual((c.costs.sell_tax, c.costs.buy_fee), (0.003, Costs().buy_fee))
        self.assertNotIn("costs", c.extra)

    def test_bad_values_rejected_with_section_prefix(self):
        bad = [
            ("account", 5),
            ("account", {"cash": 0}),
            ("account", {"cash": -1}),
            ("account", {"cash_rate": -1}),
            ("account", {"cash": "1000"}),
            ("account", {"cash": True}),
            ("account", {"principal": 1}),
            ("costs", {"buy_fee": 1}),
            ("costs", {"sell_fee": -0.1}),
            ("costs", {"sell_tax": 1.5}),
            ("costs", {"sell_tax": False}),
            ("costs", {"tax": 0.1}),
            ("costs", []),
        ]
        for section, value in bad:
            with self.assertRaises(ConfigError, msg=(section, value)) as cm:
                self.parse(**{section: value})
            self.assertTrue(str(cm.exception).startswith(f"[{section}]："), str(cm.exception))
        # 對照：邊界值放行
        self.parse(account={"cash": 1, "cash_rate": -0.5}, costs={"buy_fee": 0, "sell_tax": 0.999})

    def test_shared_account_same_passes_different_rejected(self):
        a = self.parse(account={"cash": 2_000_000})
        b = self.parse(account={"cash": 2_000_000})
        self.assertEqual(shared_account([a, b]), (a.account, a.costs))
        for other in (
            self.parse(account={"cash": 3_000_000}),
            self.parse(account={"cash": 2_000_000, "cash_rate": 0.02}),
            self.parse(account={"cash": 2_000_000}, costs={"sell_tax": 0.002}),
        ):
            with self.assertRaises(ConfigError):
                shared_account([a, other])


if __name__ == "__main__":
    unittest.main()
