import tempfile
import unittest
from dataclasses import fields

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


if __name__ == "__main__":
    unittest.main()
