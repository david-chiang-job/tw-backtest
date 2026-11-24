import tempfile
import unittest
from types import SimpleNamespace

import numba
import numpy as np

from twbacktest import synthetic
from twbacktest.config import Account, Costs
from twbacktest.data import load_market
from twbacktest.fills import FILLS
from twbacktest.grid import IndicatorCache, expand
from twbacktest.indicators import Indicator
from twbacktest.kernel import run_accounts
from twbacktest.registry import Registry
from twbacktest.signals import Signals, build_signals
from twbacktest.tiers import RESETS

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


NO_COST = Costs(0.0, 0.0, 0.0)
NO_INTEREST = Account(cash=5_000_000, cash_rate=0.0)


def make_signals(hit, price, order, target, side, rearm=None):
    """hit、price、order 是 [D, K]（單一策略），其餘照形狀補。"""
    hit = np.asarray(hit, dtype=bool)[None]
    rearm = np.zeros_like(hit) if rearm is None else np.asarray(rearm, dtype=bool)[None]
    return Signals(
        hit=hit, price=np.asarray(price, dtype=float)[None], rearm=rearm,
        order=np.asarray(order, dtype=np.int8)[None], target=np.asarray([target], dtype=float),
        side=np.asarray([side], dtype=np.int8), both_sides=[[]],
    )


def market(close, start="2024-01-01"):
    close = np.asarray(close, dtype=float)
    return SimpleNamespace(dates=np.datetime64(start) + np.arange(len(close)), close=close)


def final_nav(sig, mk):
    last = len(mk.close) - 1
    a = run_accounts(sig, mk, np.array([0]), np.array([last]), np.array([0, 1]), account=NO_INTEREST, costs=NO_COST)
    return a.nav[0, 0]


class SameDayOrder(unittest.TestCase):
    def sig(self, order):
        # 單日、兩個買進檔位：0 號價 100 目標 0.6，1 號價 50 目標 0.3
        return make_signals([[True, True]], [[100.0, 50.0]], [order], [0.6, 0.3], [1, 1])

    def run_order(self, order):
        return run_accounts(self.sig(order), market([100.0]), np.array([0]), np.array([0]), np.array([0, 1]))

    def test_later_tier_uses_cash_and_units_after_earlier_fill(self):
        # 1 號先：買 30000 單位 @50；0 號再以新淨值補 8995 單位 @100
        a = self.run_order([1, 0])
        cash = 5_000_000 - 30_000 * 50 * (1 + BF) - 8_995 * 100 * (1 + BF)
        self.assertAlmostEqual(a.nav[0, 0], cash + (30_000 + 8_995) * 100, places=4)
        # 對照：0 號先：買 30000 單位 @100，1 號到手時已超過目標而不成交
        b = self.run_order([0, 1])
        cash = 5_000_000 - 30_000 * 100 * (1 + BF)
        self.assertAlmostEqual(b.nav[0, 0], cash + 30_000 * 100, places=4)
        self.assertGreater(abs(a.nav[0, 0] - b.nav[0, 0]), 100)


class SellFill(unittest.TestCase):
    def test_sell_pays_fee_and_tax_and_interest(self):
        # 第 0 天買進（目標 0.8 @100）、第 2 天賣出（目標 0.2 @120）；日期有 3 曆日間隔
        hit = [[True, False], [False, False], [False, True]]
        price = [[100.0, np.nan], [np.nan, np.nan], [np.nan, 120.0]]
        order = [[0, -1], [-1, -1], [1, -1]]
        sig = make_signals(hit, price, order, [0.8, 0.2], [1, -1])
        a = run_accounts(sig, MARKET, np.array([0]), np.array([2]), np.array([0, 1]))
        cash2 = (5_000_000 - 40_000 * 100 * (1 + BF)) * R ** (3 / 365) * R ** (1 / 365)
        # 賣出量 = floor((40000 * 120 - 0.2 * (cash2 + 40000 * 120)) / 120) = 30336
        cash = cash2 + 30_336 * 120 * (1 - BF - 0.001)
        self.assertAlmostEqual(a.nav[0, 0], cash + (40_000 - 30_336) * 90, places=3)
        self.assertEqual(a.first_buy[0, 0], 0)


class Rearm(unittest.TestCase):
    P = [100.0, 90.0, 80.0, 72.0, 60.0]  # 每天成交價都比前一天低，所以只要待命就會成交
    C = [100.0, 90.0, 110.0, 95.0, 120.0]

    def sig(self, rearm):
        n = len(self.P)
        return make_signals([[True]] * n, [[x] for x in self.P], [[0]] * n, [0.5], [1], rearm=[[r] for r in rearm])

    def nav(self, rearm):
        return final_nav(self.sig(rearm), market(self.C))

    def test_waiting_until_rearm_and_fill_day_rearm_counts(self):
        # 目標 0.5、無費無息、結束價 120
        f = False
        # rearm 一直為假：只在第 0 天成交 25000 單位
        self.assertAlmostEqual(self.nav([f, f, f, f, f]), 5_000_000 - 25_000 * 100 + 25_000 * 120, places=4)
        # 成交當天 rearm 為真：第 1 天再成交 1388 單位 @90，之後不再
        self.assertAlmostEqual(
            self.nav([True, f, f, f, f]), 5_000_000 - 25_000 * 100 - 1_388 * 90 + (25_000 + 1_388) * 120, places=4
        )
        # rearm 到第 1 天才為真：第 1 天仍在等待，第 2 天成交 3125 單位 @80
        self.assertAlmostEqual(
            self.nav([f, True, f, f, f]), 5_000_000 - 25_000 * 100 - 3_125 * 80 + (25_000 + 3_125) * 120, places=4
        )

    def test_untouched_tier_stays_armed_until_filled(self):
        # 0 號先成交到目標；1 號同天觸發但已達目標（不成交，仍待命），隔天價低才成交
        hit = [[True, True], [False, True]]
        price = [[100.0, 100.0], [np.nan, 50.0]]
        sig = make_signals(hit, price, [[0, 1], [1, -1]], [0.5, 0.5], [1, 1])
        got = final_nav(sig, market([100.0, 100.0]))
        self.assertAlmostEqual(got, 5_000_000 - 25_000 * 100 - 12_500 * 50 + 37_500 * 100, places=4)

    def test_builtin_reset_rules_through_core(self):
        close = np.array(self.C)
        trig = np.full(len(close), 100.0)
        want = {
            "recross": 5_000_000 - 25_000 * 100 - 4_861 * 72 + (25_000 + 4_861) * 120,
            "new_high": 5_000_000 - 25_000 * 100 - 1_388 * 90 - 3_299 * 72 + (25_000 + 1_388 + 3_299) * 120,
            "none": 5_000_000 - 25_000 * 100 - 1_388 * 90 - 1_650 * 80 - 1_557 * 72 - 2_960 * 60
            + (25_000 + 1_388 + 1_650 + 1_557 + 2_960) * 120,
        }
        for name, nav in want.items():
            rearm = RESETS.get(name)(close, trig, 1)
            self.assertAlmostEqual(self.nav(rearm), nav, places=3, msg=name)


FAKE_REGISTRY = Registry("指標")
FAKE_REGISTRY.register("fake_level")(Indicator("fake_level", lambda close: np.full(len(close), 1.0), lambda b, t: b * t))


@FILLS.register("fake_fill_ext")
def fake_fill(open, high, low, close, trigger, side):
    hit = np.zeros(len(close), dtype=bool)
    hit[2] = True  # 只在第 2 天成交，價格 33 不是行情裡的任何價
    return hit, np.where(hit, 33.0, np.nan)


class ExtensionPoints(unittest.TestCase):
    def test_fake_indicator_and_fill_run_without_touching_core(self):
        data = {
            "fill": {"model": "fake_fill_ext"},
            "account": {"cash_rate": 0.0},
            "costs": {"buy_fee": 0.0},
            "tiers": [{
                "name": "x", "target_stock": 0.5, "reset": "none",
                "when": {"indicator": "fake_level", "at_most": 1.0},
            }],
        }
        strategies = expand(data, registry=FAKE_REGISTRY)
        close = np.array([10.0, 11.0, 12.0, 13.0])
        mk = SimpleNamespace(
            dates=np.datetime64("2024-01-01") + np.arange(4), open=close, high=close, low=close, close=close
        )
        sig = build_signals(strategies, mk, cache=IndicatorCache(close, registry=FAKE_REGISTRY))
        cfg = strategies[0].config
        a = run_accounts(sig, mk, np.array([0]), np.array([1, 3]), np.array([0, 2]), account=cfg.account, costs=cfg.costs)
        self.assertEqual(a.first_buy[0].tolist(), [-1, 2])
        self.assertEqual(a.nav[0, 0], 5_000_000)
        # 第 2 天用假成交價 33 買 floor(2500000 / 33) = 75757 單位
        self.assertAlmostEqual(a.nav[0, 1], 5_000_000 - 75_757 * 33 + 75_757 * 13, places=4)


class ThreadsDeterministic(unittest.TestCase):
    def test_same_results_for_different_thread_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            mk = load_market(*synthetic.write_csv(tmp, days=1500))
        data = {
            "tiers": [
                {"name": "買", "target_stock": 0.8, "reset": "recross", "when": {
                    "indicator": "ma_deviation", "days": {"from": 5, "to": 7, "step": 1},
                    "at_most": {"from": -0.03, "to": -0.02, "step": 0.005}}},
                {"name": "賣", "target_stock": 0.3, "reset": "recross", "when": {
                    "indicator": "ma_deviation", "days": 5,
                    "at_least": {"from": 0.01, "to": 0.02, "step": 0.01}}},
            ]
        }
        strategies = expand(data)
        self.assertGreaterEqual(len(strategies), 12)
        sig = build_signals(strategies, mk)
        starts = np.arange(0, 1200, 50)
        ends = np.concatenate([[s + 100, s + 250] for s in starts])
        ptr = np.arange(0, 2 * len(starts) + 1, 2)
        one = run_accounts(sig, mk, starts, ends, ptr, threads=1)
        many = run_accounts(sig, mk, starts, ends, ptr, threads=min(4, numba.config.NUMBA_NUM_THREADS))
        for name in ("nav", "max_drawdown", "lowest", "first_buy"):
            self.assertTrue(np.array_equal(getattr(one, name), getattr(many, name)), name)
        # 對照：結果不是空的
        self.assertTrue((one.first_buy >= 0).any())
        self.assertTrue((one.first_buy == -1).any())
        self.assertGreater(len(np.unique(one.nav)), 1)


class WindowChecks(unittest.TestCase):
    def call(self, starts=(0,), ends=(1, 2), ptr=(0, 2), close=(1.0, 1.0, 1.0)):
        sig = make_signals([[False]] * 3, [[np.nan]] * 3, [[-1]] * 3, [0.5], [1])
        return run_accounts(sig, market(close), np.array(starts), np.array(ends), np.array(ptr))

    def test_valid_window_passes(self):
        self.assertEqual(self.call().nav.shape, (1, 2))

    def test_bad_windows_rejected(self):
        bad = {
            "ptr 長度": dict(ptr=(0, 1, 2)),
            "ptr[0]": dict(ptr=(1, 2)),
            "ptr[-1]": dict(ptr=(0, 1)),
            "沒有終點": dict(starts=(0, 1), ends=(1, 2), ptr=(0, 0, 2)),
            "早於起跑日": dict(starts=(2,), ends=(1,), ptr=(0, 1)),
            "超出 D": dict(ends=(1, 3)),
            "遞減": dict(ends=(2, 1)),
        }
        for why, kw in bad.items():
            with self.assertRaises(ValueError, msg=why):
                self.call(**kw)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):  # 行情天數跟訊號表不同
            self.call(close=(1.0, 1.0, 1.0, 1.0))
        sig = make_signals([[False]] * 3, [[np.nan]] * 3, [[-1]] * 3, [0.5], [1])
        for name, arr in (("price", np.zeros((1, 3, 2))), ("target", np.zeros((1, 2)))):
            bad = Signals(**{**sig.__dict__, name: arr})
            with self.assertRaises(ValueError, msg=name):
                run_accounts(bad, market([1.0] * 3), np.array([0]), np.array([1]), np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
