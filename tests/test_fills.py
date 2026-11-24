import unittest

import numpy as np

from twbacktest.fills import FILLS

# 四個例子：開、高、低、收、買進觸發價；預期是否觸發與成交價
O = np.array([35500.0, 35500.0, 36000.0, 33500.0])
H = np.array([36000.0, 36000.0, 36000.0, 34000.0])
L = np.array([34000.0, 34950.0, 34000.0, 33000.0])
C = np.array([35001.0, 35800.0, 34000.0, 33800.0])
P = np.array([35000.0, 35000.0, 34100.0, 35000.0])
HIT = np.array([True, False, True, True])
PRICE = np.array([35000.0, np.nan, 34100.0, 33500.0])

M = 70000.0


def touch(*args, **kw):
    return FILLS.get("touch")(*args, **kw)


class TouchBuy(unittest.TestCase):
    def test_four_examples(self):
        hit, price = touch(O, H, L, C, P, 1)
        np.testing.assert_array_equal(hit, HIT)
        np.testing.assert_allclose(price, PRICE)  # NaN 位置也要一致


class TouchSellMirror(unittest.TestCase):
    def test_mirrored_examples(self):
        hit, price = touch(M - O, M - L, M - H, M - C, M - P, -1)
        np.testing.assert_array_equal(hit, HIT)
        np.testing.assert_allclose(price, M - PRICE)


class TouchEdges(unittest.TestCase):
    def test_nan_trigger_never_hits(self):
        p = P.copy()
        p[1] = np.nan
        p[3] = np.nan
        hit, price = touch(O, H, L, C, p, 1)
        np.testing.assert_array_equal(hit, [True, False, True, False])
        self.assertTrue(np.isnan(price[1]) and np.isnan(price[3]))
        self.assertEqual(price[0], 35000.0)

    def test_band_is_used(self):
        # 第二例：只有最低影線碰到；band=(0,1) 時下緣就是最低價，觸發
        hit, price = touch(O, H, L, C, P, 1, band=(0.0, 1.0))
        self.assertTrue(hit[1])
        self.assertEqual(price[1], 35000.0)

    def test_invalid_band_rejected(self):
        for bad in [(0.9, 0.1), (-0.1, 0.5), (0.5, 1.1), (0.5,), 0.5, "ab", (0.1, 0.5, 0.9)]:
            with self.subTest(band=bad):
                with self.assertRaises(ValueError) as cm:
                    touch(O, H, L, C, P, 1, band=bad)
                self.assertIn(repr(bad), str(cm.exception))

    def test_valid_band_accepted(self):
        touch(O, H, L, C, P, 1, band=[0.2, 0.8])
        touch(O, H, L, C, P, 1, band=(0.5, 0.5))


class CloseNextOpen(unittest.TestCase):
    def test_prev_close_decides_and_today_open_fills(self):
        # 買進、觸發價 100。第 1 天收盤 95 越過；第 2 天收盤 105 沒越過但盤中低到 90。
        o = np.array([110.0, 98.0, 97.0])
        h = np.array([112.0, 99.0, 106.0])
        l = np.array([108.0, 94.0, 90.0])
        c = np.array([110.0, 95.0, 105.0])
        p = np.array([100.0, 100.0, 100.0])
        hit, price = FILLS.get("close_next_open")(o, h, l, c, p, 1)
        np.testing.assert_array_equal(hit, [False, False, True])
        np.testing.assert_allclose(price, [np.nan, np.nan, 97.0])

    def test_sell_and_nan_trigger(self):
        o = np.array([100.0, 101.0, 102.0, 103.0])
        c = np.array([100.0, 120.0, 100.0, 120.0])
        p = np.array([110.0, 110.0, np.nan, 110.0])
        hit, price = FILLS.get("close_next_open")(o, o, o, c, p, -1)
        # 第 2 天看第 1 天（120 >= 110）觸發；第 3 天看第 2 天，其觸發價是 NaN，不觸發
        np.testing.assert_array_equal(hit, [False, False, True, False])
        self.assertEqual(price[2], 102.0)


if __name__ == "__main__":
    unittest.main()
