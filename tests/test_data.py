import os
import tempfile
import unittest

from twbacktest.data import DataError, load_market, load_volume

PRICE = "date,open,high,low,close\n"
TR = "date,close\n"
VOL = "date,volume,turnover,transactions\n"


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return p


class LoadMarketChecks(unittest.TestCase):
    def test_normal_files_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "p.csv", PRICE + "2024-01-02,100,110,90,105\n2024-01-03,105,120,100,110\n")
            t = _write(d, "t.csv", TR + "2024-01-02,200\n2024-01-03,220\n")
            m = load_market(p, t)
        self.assertEqual(len(m.close), 2)
        self.assertEqual(m.price_index.rows, 2)

    def test_reversed_dates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "p.csv", PRICE + "2024-01-03,105,120,100,110\n2024-01-02,100,110,90,105\n")
            t = _write(d, "t.csv", TR + "2024-01-02,200\n2024-01-03,220\n")
            with self.assertRaises(DataError):
                load_market(p, t)

    def _load(self, price_rows, tr_rows):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "p.csv", PRICE + price_rows)
            t = _write(d, "t.csv", TR + tr_rows)
            return load_market(p, t)

    def test_ex_dividend_day_uses_same_day_ratio(self):
        # 第二天比例由 200/105 跳到 252/120 = 2.1；手算：
        # 第一天比例 1.9047619：開 190.47619、高 209.52381、低 171.42857
        # 第二天比例 2.1：開 231、高 262.5、低 220.5（若用前一天比例開盤只有 209.52）
        m = self._load(
            "2024-01-02,100,110,90,105\n2024-01-03,110,125,105,120\n",
            "2024-01-02,200\n2024-01-03,252\n",
        )
        for got, want in [
            (m.open, [190.47619, 231.0]),
            (m.high, [209.52381, 262.5]),
            (m.low, [171.42857, 220.5]),
            (m.close, [200.0, 252.0]),
        ]:
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=4)

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(DataError):
            self._load(
                "2024-01-02,100,110,90,105\n2024-01-02,100,110,90,105\n",
                "2024-01-02,200\n",
            )

    def test_missing_price_day_rejected(self):
        with self.assertRaises(DataError) as cm:
            self._load("2024-01-02,100,110,90,105\n", "2024-01-02,200\n2024-01-03,220\n")
        self.assertIn("2024-01-03", str(cm.exception))

    def test_unreasonable_ohlc_rejected(self):
        with self.assertRaises(DataError):
            self._load("2024-01-02,100,110,90,115\n", "2024-01-02,200\n")

    def test_extra_earlier_price_days_ignored(self):
        # 較早那列開高低收不合理，但在報酬指數第一天以前，要放行
        m = self._load(
            "2023-12-29,1,2,3,4\n2024-01-02,100,110,90,105\n2024-01-03,105,120,100,110\n",
            "2024-01-02,200\n2024-01-03,220\n",
        )
        self.assertEqual(str(m.dates[0]), "2024-01-02")
        self.assertEqual(len(m.dates), 2)
        self.assertEqual(m.price_index.rows, 3)
        self.assertEqual(m.price_index.start, "2023-12-29")
        self.assertEqual(m.price_index.end, "2024-01-03")
        self.assertEqual(m.total_return.start, "2024-01-02")

    def test_extra_later_price_day_rejected(self):
        with self.assertRaises(DataError):
            self._load(
                "2024-01-02,100,110,90,105\n2024-01-03,105,120,100,110\n",
                "2024-01-02,200\n",
            )

    def test_load_market_with_volume_aligns_and_ignores_earlier_days(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "p.csv", PRICE + "2024-01-02,100,110,90,105\n2024-01-03,105,120,100,110\n")
            t = _write(d, "t.csv", TR + "2024-01-02,200\n2024-01-03,220\n")
            v = _write(
                d,
                "v.csv",
                VOL
                + "2023-12-29,999,888,777\n"
                + "2024-01-02,1000,250000,1500\n"
                + "2024-01-03,1200,310000,1800\n",
            )
            m = load_market(p, t, volume=v)
        self.assertIsNotNone(m.volume)
        self.assertIsNotNone(m.turnover)
        self.assertIsNotNone(m.transactions)
        self.assertEqual(list(m.volume), [1000.0, 1200.0])
        self.assertEqual(list(m.turnover), [250000.0, 310000.0])
        self.assertEqual(list(m.transactions), [1500.0, 1800.0])
        self.assertEqual(m.volume_source.rows, 3)
        self.assertEqual(m.volume_source.start, "2023-12-29")

    def test_load_market_without_volume_keeps_none(self):
        m = self._load("2024-01-02,100,110,90,105\n", "2024-01-02,200\n")
        self.assertIsNone(m.volume)
        self.assertIsNone(m.turnover)
        self.assertIsNone(m.transactions)
        self.assertIsNone(m.volume_source)

    def test_missing_volume_day_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "p.csv", PRICE + "2024-01-02,100,110,90,105\n2024-01-03,105,120,100,110\n")
            t = _write(d, "t.csv", TR + "2024-01-02,200\n2024-01-03,220\n")
            v = _write(d, "v.csv", VOL + "2024-01-02,1000,250000,1500\n")
            with self.assertRaises(DataError) as cm:
                load_market(p, t, volume=v)
        self.assertIn("2024-01-03", str(cm.exception))

    def test_extra_later_volume_day_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "p.csv", PRICE + "2024-01-02,100,110,90,105\n")
            t = _write(d, "t.csv", TR + "2024-01-02,200\n")
            v = _write(
                d,
                "v.csv",
                VOL + "2024-01-02,1000,250000,1500\n2024-01-03,1200,310000,1800\n",
            )
            with self.assertRaises(DataError) as cm:
                load_market(p, t, volume=v)
        self.assertIn("2024-01-03", str(cm.exception))

    def test_load_volume_standalone_and_bad_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            v = _write(d, "v.csv", VOL + "2024-01-02,1000,250000,1500\n2024-01-03,1200,310000,1800\n")
            vd = load_volume(v)
            self.assertEqual(len(vd.dates), 2)
            self.assertEqual(list(vd.turnover), [250000.0, 310000.0])

            bad = _write(d, "bad.csv", "date,volume\n2024-01-02,1000\n")
            with self.assertRaises(DataError):
                load_volume(bad)


if __name__ == "__main__":
    unittest.main()
