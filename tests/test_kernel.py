import unittest
from types import SimpleNamespace

import numpy as np

from twbacktest.kernel import run_accounts
from twbacktest.signals import Signals

BF = 0.00049875
R = 1.015
# 週五、下週一（相隔 3 曆日）、下週二（相隔 1 曆日）
DATES = np.array(["2024-01-05", "2024-01-08", "2024-01-09"], dtype="datetime64[D]")
CLOSE = np.array([100.0, 110.0, 90.0])
MARKET = SimpleNamespace(dates=DATES, close=CLOSE)


def signals():
    # 一個買進檔位、目標 0.8；只有第 0 天觸發，成交價 100，之後不會重新待命
    hit = np.zeros((1, 3, 1), dtype=bool)
    hit[0, 0, 0] = True
    price = np.full((1, 3, 1), np.nan)
    price[0, 0, 0] = 100.0
    order = np.full((1, 3, 1), -1, dtype=np.int8)
    order[0, 0, 0] = 0
    return Signals(
        hit=hit, price=price, rearm=np.zeros((1, 3, 1), dtype=bool), order=order,
        target=np.array([[0.8]]), side=np.array([[1]], dtype=np.int8), both_sides=[[]],
    )


class AccountsBuy(unittest.TestCase):
    def run_it(self, **kw):
        # 起跑日 0（第 0 天有觸發）與起跑日 1（之後都沒觸發），終點都是 1、2
        starts = np.array([0, 1])
        ends = np.array([1, 2, 1, 2])
        ptr = np.array([0, 2, 4])
        return run_accounts(signals(), MARKET, starts, ends, ptr, **kw)

    def test_buy_fill_cash_units_fee_and_interest(self):
        a = self.run_it()
        # 目標 0.8 * 5,000,000 / 100 = 40,000 單位，成交當天不計息
        cash0 = 5_000_000 - 40_000 * 100 * (1 + BF)
        cash1 = cash0 * R ** (3 / 365)  # 週五到週一計 3 曆日
        cash2 = cash1 * R ** (1 / 365)
        nav0 = cash0 + 40_000 * 100
        nav1 = cash1 + 40_000 * 110
        nav2 = cash2 + 40_000 * 90
        self.assertAlmostEqual(a.nav[0, 0], nav1, places=4)
        self.assertAlmostEqual(a.nav[0, 1], nav2, places=4)
        # 高點從本金起算：第 0 天扣完費低於本金，回撤 = 1 - nav0 / 本金；第 2 天從第 1 天高點回落
        self.assertAlmostEqual(a.max_drawdown[0, 0], 1 - nav0 / 5_000_000, places=12)
        self.assertAlmostEqual(a.max_drawdown[0, 1], 1 - nav2 / nav1, places=12)
        self.assertAlmostEqual(a.lowest[0, 0], nav0, places=4)
        self.assertAlmostEqual(a.lowest[0, 1], min(nav0, nav2), places=4)
        self.assertEqual(a.first_buy[0, :2].tolist(), [0, 0])

    def test_start_without_trigger_only_earns_interest(self):
        a = self.run_it()
        # 起跑日當天不計息：第 1 天淨值恰為本金；第 2 天只多 1 曆日利息
        self.assertEqual(a.nav[0, 2], 5_000_000)
        self.assertAlmostEqual(a.nav[0, 3], 5_000_000 * R ** (1 / 365), places=4)
        self.assertEqual(a.first_buy[0, 2:].tolist(), [-1, -1])
        self.assertEqual(a.max_drawdown[0, 2:].tolist(), [0.0, 0.0])
        self.assertEqual(a.lowest[0, 2], 5_000_000)

    def test_threads_argument_restores_setting(self):
        import numba

        before = numba.get_num_threads()
        a = self.run_it(threads=1)
        self.assertEqual(numba.get_num_threads(), before)
        self.assertEqual(a.nav[0, 2], 5_000_000)


if __name__ == "__main__":
    unittest.main()
