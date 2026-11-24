import unittest

import numpy as np

from twbacktest.kernel import Accounts
from twbacktest.scoring import SCORES, WindowMetrics, rank, summarize, window_metrics
from twbacktest.windows import Windows

NAN = np.nan


def metrics(**given):
    n, e = np.atleast_2d(next(iter(given.values()))).shape
    return WindowMetrics(
        **{f: np.atleast_2d(np.asarray(given.get(f, np.zeros((n, e))), dtype=float)) for f in WindowMetrics.__dataclass_fields__}
    )


def wins(kind, key):
    e = len(kind)
    return Windows(np.array([0]), np.array([0, e]), np.arange(e), np.array(kind), np.array(key))


def accounts(nav, lowest, mdd, first):
    return Accounts(np.array(nav), np.array(mdd), np.array(lowest), np.array(first, dtype=np.int64))


class WindowMetricsTest(unittest.TestCase):
    def test_formulas(self):
        dates = np.datetime64("2024-01-01") + np.array([0, 10, 20, 30, 40])
        # 起跑 0 有終點 2、4（20 與 40 日），起跑 1 有終點 4（30 日）
        w = Windows(np.array([0, 1]), np.array([0, 2, 3]), np.array([2, 4, 4]), np.array(["x"] * 3), np.full(3, -1))
        acc = accounts([[1100.0, 1210.0, 950.0]], [[1000.0, 1100.0, 900.0]], [[0.1, 0.2, 0.3]], [[0, -1, 2]])
        base = accounts([[1050.0, 1000.0, 1200.0]], [[0.0] * 3], [[0.0] * 3], [[0] * 3])
        m = window_metrics(acc, base, w, dates, 1000.0)
        e = 365.2425
        for got, want in [
            (m.total_return, [0.1, 0.21, -0.05]),
            (m.annualized, [1.1 ** (e / 20) - 1, 1.21 ** (e / 40) - 1, 0.95 ** (e / 30) - 1]),
            (m.excess_total, [(0.1 - 0.05) * 100, (0.21 - 0.0) * 100, (-0.05 - 0.2) * 100]),
            (
                m.excess_annual,
                [
                    ((1.1 ** (e / 20) - 1) - (1.05 ** (e / 20) - 1)) * 100,
                    ((1.21 ** (e / 40) - 1) - (1.0 ** (e / 40) - 1)) * 100,
                    ((0.95 ** (e / 30) - 1) - (1.2 ** (e / 30) - 1)) * 100,
                ],
            ),
            (m.max_drawdown, [0.1, 0.2, 0.3]),
            (m.principal_loss, [0.0, 0.0, 0.1]),  # 最低淨值等於本金與高於本金都是 0
            (m.wait_days, [0.0, NAN, 10.0]),  # 起跑 1 的首次買入在索引 2，相隔 10 曆日
        ]:
            np.testing.assert_allclose(got[0], want, rtol=1e-12, atol=1e-12)

    def test_baseline_against_itself_has_zero_excess(self):
        dates = np.datetime64("2024-01-01") + np.array([0, 10])
        w = Windows(np.array([0]), np.array([0, 1]), np.array([1]), np.array(["x"]), np.array([-1]))
        base = accounts([[1100.0]], [[1000.0]], [[0.0]], [[0]])
        m = window_metrics(base, base, w, dates, 1000.0)
        self.assertEqual((m.excess_total[0, 0], m.excess_annual[0, 0]), (0.0, 0.0))


class ScoreFunctionsTest(unittest.TestCase):
    def test_median_and_p10_ignore_nan_with_linear_interpolation(self):
        row = np.array([[10.0, NAN, 20.0, 30.0, 40.0, 50.0]])
        k = np.zeros(6)
        # 五個有效值的中位數是 30；第 10 百分位位置 0.1 * 4 = 0.4，線性插值 14（最近值是 10）
        self.assertEqual(SCORES.get("median")(row, k)[0], 30.0)
        self.assertAlmostEqual(SCORES.get("p10")(row, k)[0], 14.0)

    def test_nan_share(self):
        row = np.array([[NAN, 5.0, NAN, 7.0], [1.0, 2.0, 3.0, NAN]])
        np.testing.assert_allclose(SCORES.get("nan_share")(row, np.zeros(4)), [0.5, 0.25])

    def test_group_mean_equal_custom_weights_and_errors(self):
        # 組 5 有 3 窗、組 9 有 1 窗
        vals = np.array([[1.0, 2.0, 3.0, 10.0], [2.0, 4.0, 6.0, 20.0]])
        keys = np.array([5, 5, 5, 9])
        gm = SCORES.get("group_mean")
        np.testing.assert_allclose(gm(vals, keys), [(2 + 10) / 2, (4 + 20) / 2])  # 6、12；窗口直接平均是 4、8
        np.testing.assert_allclose(gm(vals, keys, {5: 1, 9: 3}), [(2 + 30) / 4, (4 + 60) / 4])  # 8、16
        with self.assertRaises(ValueError):
            gm(vals, keys, {5: 1})
        with self.assertRaises(ValueError):
            gm(vals, keys, {5: 0, 9: 0})


class SummarizeRankTest(unittest.TestCase):
    def test_summarize_picks_kind_and_names_columns(self):
        w = wins(["720d"] * 3 + ["cycle1"] * 4, [-1, -1, -1, 5, 5, 5, 9])
        m = metrics(
            excess_total=[[1.0, 3.0, 2.0, 0, 0, 0, 0], [9.0, 8.0, 7.0, 0, 0, 0, 0]],
            total_return=[[0, 0, 0, 1.0, 2.0, 3.0, 10.0], [0, 0, 0, 2.0, 4.0, 6.0, 20.0]],
        )
        s = summarize(
            m, w,
            scores=(("720d", "excess_total", "median"), ("cycle1", "total_return", "group_mean"), ("1200d", "excess_total", "median")),
        )  # fmt: skip
        self.assertEqual(sorted(s), ["1200d.excess_total.median", "720d.excess_total.median", "cycle1.total_return.group_mean"])
        np.testing.assert_allclose(s["720d.excess_total.median"], [2.0, 8.0])
        np.testing.assert_allclose(s["cycle1.total_return.group_mean"], [6.0, 12.0])
        self.assertTrue(np.isnan(s["1200d.excess_total.median"]).all())  # 沒有該種類窗口：NaN 不丟錯

    def test_summarize_weights_passed_to_group_mean(self):
        w = wins(["cycle1"] * 4, [5, 5, 5, 9])
        m = metrics(total_return=[[1.0, 2.0, 3.0, 10.0]])
        scores = (("cycle1", "total_return", "group_mean"),)
        np.testing.assert_allclose(summarize(m, w, scores, weights={5: 1, 9: 3})["cycle1.total_return.group_mean"], [8.0])
        with self.assertRaises(ValueError):
            summarize(m, w, scores, weights={5: 1})

    def test_rank_descending_nan_last_ties_by_index(self):
        r = rank({"x": np.array([1.0, NAN, 3.0, 3.0, 0.0])}, by="x")
        self.assertEqual(r.tolist(), [2, 3, 0, 4, 1])
        with self.assertRaises(KeyError):
            rank({"x": np.array([1.0])})

    def test_fake_score_extension(self):
        @SCORES.register("test_range")
        def _range(values, keys, weights=None):
            return values.max(axis=1) - values.min(axis=1)

        w = wins(["cycle1"] * 3, [1, 1, 1])
        m = metrics(total_return=[[1.0, 4.0, 2.0]])
        s = summarize(m, w, scores=(("cycle1", "total_return", "test_range"),))
        np.testing.assert_allclose(s["cycle1.total_return.test_range"], [3.0])


if __name__ == "__main__":
    unittest.main()
