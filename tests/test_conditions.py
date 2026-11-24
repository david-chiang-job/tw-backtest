import unittest

import numpy as np

from twbacktest.conditions import COMBINATORS, evaluate

NAN = np.nan
A = np.array([10, NAN, 7, NAN], dtype=float)
B = np.array([8, 9, NAN, NAN], dtype=float)
C = np.array([6, 5, 4, NAN], dtype=float)


def same(test, got, want):
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want, dtype=float))


class AnyAllChecks(unittest.TestCase):
    def test_any_takes_best_trigger_and_skips_nan(self):
        same(self, COMBINATORS.get("any")([A, B], 1), [10, 9, 7, NAN])
        same(self, COMBINATORS.get("any")([A, B], -1), [8, 9, 7, NAN])

    def test_all_needs_every_condition(self):
        same(self, COMBINATORS.get("all")([A, B], 1), [8, NAN, NAN, NAN])
        same(self, COMBINATORS.get("all")([A, B], -1), [10, NAN, NAN, NAN])


class WeightedChecks(unittest.TestCase):
    def run_weighted(self, side, arrays=(A, B, C), points=(50, 25, 25), n=60):
        return COMBINATORS.get("weighted")(list(arrays), side, points=list(points), threshold=n)

    def test_buy_accumulates_from_highest_price(self):
        # 第 1 天：B、C 湊 50 < 60 不觸發；第 2 天 A(50) 加 C(25) 才到 → 取 C 的價位 4
        same(self, self.run_weighted(1), [8, NAN, 4, NAN])

    def test_sell_accumulates_from_lowest_price(self):
        same(self, self.run_weighted(-1), [10, NAN, 7, NAN])

    def test_same_price_conditions_add_together(self):
        x = np.array([5.0])
        y = np.array([5.0])
        z = np.array([9.0])
        # 30+30 同價位一起到 60；若不一起加就會湊不到
        same(self, self.run_weighted(1, (x, y), (30, 30)), [5])
        same(self, self.run_weighted(1, (x, y, z), (30, 30, 40)), [5])


class Nested(unittest.TestCase):
    def test_weighted_with_any_child(self):
        arrays = {"a": A, "b": B, "c": np.array([12, 5, NAN, 3], dtype=float)}
        tree = {
            "score_at_least": 60,
            "weighted": [
                {"points": 50, "any": [{"indicator": "a"}, {"indicator": "b"}]},
                {"points": 25, "indicator": "c"},
            ],
        }
        got = evaluate(tree, lambda leaf: arrays[leaf["indicator"]], 1)
        # any(a,b)=[10,9,7,NaN]；第 2 天 c 為 NaN 不給分，只剩 50 → 不觸發
        same(self, got, [10, 5, NAN, NAN])


if __name__ == "__main__":
    unittest.main()
