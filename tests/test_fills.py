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


if __name__ == "__main__":
    unittest.main()
