import unittest
from types import SimpleNamespace

import numpy as np

from twbacktest.baseline import run_baseline
from twbacktest.config import Account, Costs
from twbacktest.windows import Windows

# 週五、下週一（相隔 3 曆日）、下週二（相隔 1 曆日）
DATES = np.array(["2024-01-05", "2024-01-08", "2024-01-09"], dtype="datetime64[D]")
OPEN = np.array([100.0, 110.0, 90.0])
CLOSE = np.array([105.0, 120.0, 95.0])
MARKET = SimpleNamespace(dates=DATES, open=OPEN, high=CLOSE, low=OPEN, close=CLOSE)
ACCOUNT = Account(cash=1000.0, cash_rate=0.10)
COSTS = Costs(buy_fee=0.01, sell_fee=0.0, sell_tax=0.0)
R = 1.10


def windows():
    # 起跑 0 有兩個終點（1、2），起跑 1 一個終點（2）
    return Windows(
        starts=np.array([0, 1]), ptr=np.array([0, 2, 3]), ends=np.array([1, 2, 2]),
        kind=np.array(["x"] * 3), key=np.full(3, -1),
    )  # fmt: skip


class RunBaselineTest(unittest.TestCase):
    def test_full_position_at_open_with_fee_then_hold(self):
        a = run_baseline(MARKET, windows(), ACCOUNT, COSTS)
        self.assertEqual(a.nav.shape, (1, 3))
        # 起跑 0：開盤 100、含費 101 一單位，1000 / 101 = 9.9 買 9 單位，剩 1000 - 909 = 91
        # 後來價格變了也不再加碼，仍是 9 單位；現金 91 照曆日計息（第 1 天隔 3 日，第 2 天再隔 1 日）
        self.assertAlmostEqual(a.nav[0, 0], 91 * R ** (3 / 365) + 9 * 120, places=6)
        self.assertAlmostEqual(a.nav[0, 1], 91 * R ** (4 / 365) + 9 * 95, places=6)
        # 起跑 1 用自己的開盤 110、含費 111.1：1000 / 111.1 = 9.0009 買 9 單位，剩 1000 - 999.9 = 0.1
        self.assertAlmostEqual(a.nav[0, 2], 0.1 * R ** (1 / 365) + 9 * 95, places=6)
        self.assertEqual(a.first_buy[0].tolist(), [0, 0, 1])

    def test_cash_too_small_buys_nothing(self):
        # 本金不夠買一單位時什麼都不買，淨值只剩現金
        a = run_baseline(MARKET, windows(), Account(cash=50.0, cash_rate=0.0), COSTS)
        self.assertEqual(a.nav[0].tolist(), [50.0, 50.0, 50.0])
        self.assertEqual(a.first_buy[0].tolist(), [-1, -1, -1])

    def test_buys_maximum_units_including_fee(self):
        # 開盤 99.005、含費 99.99505：1000 / 99.99505 = 10.0005 買 10 單位（不是只買到九成九目標的 9 單位）
        mk = SimpleNamespace(
            dates=DATES[:2], open=np.array([99.005, 99.005]), high=np.array([100.0, 100.0]),
            low=np.array([99.0, 99.0]), close=np.array([100.0, 100.0]),
        )  # fmt: skip
        w = Windows(np.array([0]), np.array([0, 1]), np.array([1]), np.array(["x"]), np.array([-1]))
        a = run_baseline(mk, w, Account(cash=1000.0, cash_rate=0.0), COSTS)
        self.assertAlmostEqual(a.nav[0, 0], 1000 - 10 * 99.005 * 1.01 + 10 * 100, places=6)


if __name__ == "__main__":
    unittest.main()
