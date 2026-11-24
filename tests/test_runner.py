import tempfile
import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from twbacktest import synthetic
from twbacktest.data import load_market
from twbacktest.grid import expand
from twbacktest.runner import run_strategies

DATA = {
    "windows": {"fixed": {"lengths": [240, 720]}},
    "tiers": [
        {"name": "買", "target_stock": 0.8, "reset": "recross", "when": {
            "indicator": "ma_deviation", "days": {"from": 5, "to": 7, "step": 1},
            "at_most": {"from": -0.2, "to": -0.02, "step": 0.09}}},
        {"name": "賣", "target_stock": 0.3, "reset": "recross", "when": {
            "indicator": "ma_deviation", "days": 5,
            "at_least": {"from": 0.01, "to": 0.02, "step": 0.01}}},
    ],
}


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


class ChunkInvariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.market = load_market(*synthetic.write_csv(tmp, days=900))
        cls.strategies = expand(DATA)
        cls.n = len(cls.strategies)
        cls.keep = [cls.n - 1, 0, 7, 3]  # 故意亂序、散在不同塊
        cls.whole = run_strategies(cls.strategies, cls.market, cls.n, keep=cls.keep, threads=1)

    def test_scores_and_kept_windows_independent_of_chunk_size(self):
        whole = self.whole
        self.assertEqual(len(next(iter(whole.summary.values()))), self.n)
        for size in (1, 5, self.n):  # 5 除不盡
            with self.subTest(chunk_size=size):
                r = run_strategies(self.strategies, self.market, size, keep=self.keep, threads=1)
                self.assertEqual(set(r.summary), set(whole.summary))
                for k in whole.summary:
                    self.assertTrue(same(r.summary[k], whole.summary[k]), k)
                self.assertTrue(np.array_equal(r.ranking, whole.ranking))
                self.assertEqual(r.kept, whole.kept)
                for obj in ("kept_metrics", "kept_accounts"):
                    for f in fields(getattr(whole, obj)):
                        self.assertTrue(same(getattr(getattr(r, obj), f.name), getattr(getattr(whole, obj), f.name)), f"{obj}.{f.name}")
                for obj in ("kept_trades", "baseline_trades"):  # 第二趟的成交明細也不受分塊影響
                    a, b = getattr(r, obj), getattr(whole, obj)
                    self.assertTrue(same(a.trigger, b.trigger), f"{obj}.trigger")
                    for f in fields(b.fills):
                        self.assertTrue(same(getattr(a.fills, f.name), getattr(b.fills, f.name)), f"{obj}.{f.name}")

    def test_top_n_independent_of_chunk_size(self):
        whole = self.whole
        s = whole.summary["720d.excess_total.median"][whole.ranking]
        # 夾具要真的有同分：找一個 N，讓前 N 名的切線落在同分組中間
        ties = [p for p in range(self.n - 1) if s[p] == s[p + 1]]
        self.assertTrue(ties, "夾具沒有同分，測不到同分照索引小的先")
        top_n = ties[0] + 1
        expect = whole.ranking[:top_n]
        ref = run_strategies(self.strategies, self.market, self.n, keep=expect, threads=1)  # 另一條路取同批策略的逐窗結果
        for size in (1, 5, self.n, None):
            with self.subTest(chunk_size=size):
                r = run_strategies(self.strategies, self.market, size, keep_top=top_n, threads=1)
                self.assertEqual(list(r.top), list(expect))
                for obj, base in (("top_metrics", "kept_metrics"), ("top_accounts", "kept_accounts")):
                    for f in fields(getattr(r, obj)):
                        self.assertTrue(same(getattr(getattr(r, obj), f.name), getattr(getattr(ref, base), f.name)), f"{obj}.{f.name}")
                self.assertTrue(same(r.top_trades.trigger, ref.kept_trades.trigger))
                for f in fields(r.top_trades.fills):
                    self.assertTrue(same(getattr(r.top_trades.fills, f.name), getattr(ref.kept_trades.fills, f.name)), f.name)
        # 對照：同分的兩組策略逐窗結果確實是同一份數值但不是同一個索引（切線真的切在同分中間）
        self.assertEqual(s[top_n - 1], s[top_n])
        self.assertLess(whole.ranking[top_n - 1], whole.ranking[top_n])

    def test_auto_chunk_size_follows_budget(self):
        small = run_strategies(self.strategies, self.market, None, threads=1, memory_budget=1)
        self.assertEqual(small.chunk_size, 1)  # 預算再小也至少 1 組
        big = run_strategies(self.strategies, self.market, None, threads=1)
        self.assertGreaterEqual(big.chunk_size, small.chunk_size)
        for k in self.whole.summary:
            self.assertTrue(same(small.summary[k], self.whole.summary[k]), k)

    def test_kept_rows_match_requested_strategy(self):
        # 對照：另存的列就是該策略自己跑的結果（單獨跑一組，不經過分塊邏輯）
        for pos, idx in enumerate(self.keep):
            alone = run_strategies([self.strategies[idx]], self.market, 1, keep=[0], threads=1)
            for f in fields(alone.kept_accounts):
                self.assertTrue(same(getattr(alone.kept_accounts, f.name)[0], getattr(self.whole.kept_accounts, f.name)[pos]), f.name)

    def test_results_are_not_trivial(self):
        w = self.whole
        self.assertEqual(w.kept_accounts.nav.shape[0], len(self.keep))
        a = w.kept_accounts
        self.assertTrue((a.first_buy >= 0).any())
        self.assertTrue((a.first_buy == -1).any())
        self.assertGreater(len(np.unique(a.nav)), len(self.keep))
        for k, v in w.summary.items():
            if not np.isnan(v).all():
                self.assertGreater(len(np.unique(v[~np.isnan(v)])), 1, k)
        self.assertEqual(sorted(w.ranking), list(range(self.n)))

    def test_bad_arguments_rejected(self):
        for kw in (dict(chunk_size=0), dict(chunk_size=3, keep=[self.n])):
            kw.setdefault("chunk_size", 3)
            with self.assertRaises(ValueError):
                run_strategies(self.strategies, self.market, **kw)


class TriggerPrice(unittest.TestCase):
    """手造 12 天行情：第 3 天開盤 90 跳過觸發價 95（基準 100 × (1 - 0.05)）。"""

    DATES = np.datetime64("2024-01-01") + np.arange(12)
    OPEN = np.array([100, 100, 100, 90, 96, 97, 98, 99, 100, 100, 100, 100], dtype=float)
    CLOSE = np.array([100, 100, 100, 94, 97, 98, 99, 100, 100, 100, 100, 100], dtype=float)
    MARKET = SimpleNamespace(dates=DATES, open=OPEN, high=np.maximum(OPEN, CLOSE) + 1, low=np.minimum(OPEN, CLOSE) - 2,
                             close=CLOSE)

    def first_fill(self, model):
        data = {
            "windows": {"fixed": {"lengths": [8]}}, "fill": {"model": model},
            "tiers": [{"name": "買", "target_stock": 0.8, "reset": "recross",
                       "when": {"indicator": "ma_deviation", "days": 3, "at_most": -0.05}}],
        }
        r = run_strategies(expand(data), self.MARKET, keep=[0], keep_top=1, threads=1)
        for t in (r.kept_trades, r.top_trades):
            self.assertGreater(t.fills.offset[1], 0)  # 起跑日 0 有成交
        f, trig = r.kept_trades.fills, r.kept_trades.trigger
        self.assertEqual(int(f.tier[0]), 0)
        return int(f.day[0]), float(f.price[0]), float(trig[0]), r

    def test_touch_gap_down_price_is_open_and_trigger_is_same_day_level(self):
        day, price, trigger, r = self.first_fill("touch")
        self.assertEqual((day, price, trigger), (3, 90.0, 95.0))  # 價格是開盤、觸發價是 95，兩者不同
        # baseline：觸發價欄是當天開盤價
        b = r.baseline_trades
        self.assertEqual(b.trigger[0], self.OPEN[int(b.fills.day[0])])
        self.assertEqual(b.trigger[0], b.fills.price[0])

    def test_close_next_open_trigger_is_previous_day_level(self):
        day, price, trigger, _ = self.first_fill("close_next_open")
        # 第 3 天收盤 94 <= 95，第 4 天開盤 96 成交；觸發價是第 3 天的 95
        self.assertEqual((day, price, trigger), (4, 96.0, 95.0))


if __name__ == "__main__":
    unittest.main()
