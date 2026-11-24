import os
import tempfile
import unittest

from twbacktest.data import DataError, load_market

PRICE = "date,open,high,low,close\n"
TR = "date,close\n"


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


if __name__ == "__main__":
    unittest.main()
