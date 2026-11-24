import unittest

import numpy as np

from twbacktest.signals import day_order

# 單日，6 個檔位；檔位編號順序刻意跟觸發價順序不同
#            0     1     2     3     4     5
SIDE = np.array([1, -1, 1, -1, 1, 1])
TRIG = np.array([[100.0, 50.0, 120.0, 40.0, 110.0, 130.0]])
HIT = np.array([[True, True, True, True, True, False]])  # 5 號沒觸發


def order(hit=HIT, trig=TRIG, side=SIDE, same_day="buy_first"):
    return day_order(hit, trig, side, same_day)[0].tolist()


class DayOrder(unittest.TestCase):
    def test_buy_high_to_low_sell_low_to_high(self):
        # 買進 2(120) 4(110) 0(100)；賣出 3(40) 1(50)
        self.assertEqual(order(same_day="buy_first"), [2, 4, 0, 3, 1, -1])
        self.assertEqual(order(same_day="sell_first"), [3, 1, 2, 4, 0, -1])

    def test_only_hit_tiers_appear(self):
        self.assertNotIn(5, order())
        # 對照：5 號觸發時（價 130 最高）會排進買進最前面
        hit = HIT.copy()
        hit[0, 5] = True
        self.assertEqual(order(hit=hit), [5, 2, 4, 0, 3, 1])

    def test_nan_trigger_last_within_side_then_by_index(self):
        trig = TRIG.copy()
        trig[0, 2] = np.nan
        trig[0, 0] = np.nan
        # 買進：4(110) 先，NaN 的 0、2 依編號排後；賣出不受影響
        self.assertEqual(order(trig=trig), [4, 0, 2, 3, 1, -1])

    def test_ties_by_index_and_days_independent(self):
        trig = np.array([[100.0, 100.0], [90.0, 95.0]])
        hit = np.array([[True, True], [True, True]])
        side = np.array([1, 1])
        got = day_order(hit, trig, side, "buy_first").tolist()
        self.assertEqual(got, [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
