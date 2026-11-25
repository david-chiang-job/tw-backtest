import csv
import json
import os
import re
import shutil
import tempfile
import unittest

from twbacktest import synthetic
from twbacktest.cli import main
from .test_cli import ACTIVE

FILES = ["Chart Data.json", "Chart Prices.csv", "Chart Levels.csv", "Chart Events.csv"]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


class ChartData(unittest.TestCase):
    """三個契約共用同一次 run。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        synthetic.write_csv(cls.tmp, days=900)
        cfg = os.path.join(cls.tmp, "c.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(ACTIVE)
        cls.out = os.path.join(cls.tmp, "out")
        assert main(["run", cfg, "--out", cls.out]) == 0
        with open(os.path.join(cls.out, FILES[0]), encoding="utf-8") as f:
            cls.doc = json.load(f)
        _, cls.orders = read(os.path.join(cls.out, "orders.csv"))
        cls.events = read(os.path.join(cls.out, "Chart Events.csv"))[1]
        cls.levels = read(os.path.join(cls.out, "Chart Levels.csv"))[1]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def slot_of(self, strategy):
        return {t["name"]: t["slot"] for t in self.doc["strategies"][strategy]["tiers"]}

    def test_run_writes_four_chart_files_with_windows(self):
        for name in FILES:
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)
        self.assertTrue(self.doc["windows"])
        for w in self.doc["windows"]:
            self.assertTrue(os.path.isfile(os.path.join(self.out, w["file"])), w["file"])

    def test_chart_svg_trades_match_orders_csv(self):
        total, sides = 0, set()
        for w in self.doc["windows"]:
            want = [(r[6], str(self.slot_of(w["strategy"])[r[5]]), r[8]) for r in self.orders
                    if r[:4] == [w["strategy"], w["kind"], w["start"], w["end"]]]
            with open(os.path.join(self.out, w["file"]), encoding="utf-8") as f:
                svg = f.read()
            got = [(m[0], m[1], m[2]) for m in re.findall(
                r'<polygon data-kind="trade" data-date="([^"]*)" data-slot="([^"]*)" data-price="([^"]*)"', svg)]
            self.assertEqual(want, got, w)
            total += len(got)
            sides |= {r[5] for r in self.events if r[0] == w["strategy"] and r[2] == w["start"]}
        self.assertGreater(total, 0)
        self.assertEqual(sides, {"buy", "sell"})

    def test_level_threshold_equals_order_trigger(self):
        thr = {(r[0], r[2], r[4]): r[6] for r in self.levels if r[3] in ("buy", "sell")}
        checked = 0
        for w in self.doc["windows"]:
            for r in self.orders:
                if r[:4] == [w["strategy"], w["kind"], w["start"], w["end"]]:
                    slot = str(self.slot_of(w["strategy"])[r[5]])
                    self.assertEqual(thr[(r[6], w["strategy"], slot)], r[7], r)
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
