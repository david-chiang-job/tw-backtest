import csv
import hashlib
import json
import os
import platform
import tempfile
import tomllib
import unittest
from importlib import metadata

import numba
import numpy as np

from twbacktest import synthetic
from twbacktest.cli import main
from twbacktest.data import load_market
from twbacktest.grid import expand
from twbacktest.config import load_config
from twbacktest.runner import run_strategies
from twbacktest.scoring import summarize

CONFIG = """
[data]
dir = "."
[run]
threads = 1
keep_windows = 3
[windows]
fixed = { lengths = [240, 720] }
cycles = { counts = [1, 2], drawdown = 0.10 }
[[tiers]]
name = "買"
target_stock = 0.8
reset = "recross"
[tiers.when]
indicator = "ma_deviation"
days = { from = 5, to = 7, step = 1 }
at_most = { from = -0.2, to = -0.02, step = 0.09 }
[[tiers]]
name = "賣"
target_stock = 0.3
reset = "recross"
[tiers.when]
indicator = "ma_deviation"
days = 5
at_least = { from = 0.01, to = 0.02, step = 0.01 }
"""


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


class RunThenScore(unittest.TestCase):
    def test_score_from_windows_csv_matches_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic.write_csv(tmp, days=900)
            cfg = os.path.join(tmp, "c.toml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(CONFIG)
            out = os.path.join(tmp, "out")
            self.assertEqual(main(["run", cfg, "--out", out]), 0)
            again = os.path.join(tmp, "again.csv")
            self.assertEqual(main(["score", os.path.join(out, "windows.csv"), "--out", again]), 0)

            head, rows = read(os.path.join(out, "scores.csv"))
            self.assertEqual(head[:2], ["rank", "strategy"])
            self.assertEqual(len(rows), 18)
            whead, wrows = read(os.path.join(out, "windows.csv"))
            self.assertEqual(whead[:5], ["strategy", "kind", "start", "end", "cycle"])
            self.assertEqual({r[0] for r in wrows}, {rows[0][1], rows[1][1], rows[2][1], "baseline"})
            self.assertTrue(any(r[4] for r in wrows))  # 有循環窗口
            self.assertTrue(any(not r[4] for r in wrows))  # 也有固定期窗口

            shead, srows = read(again)
            scored = {r[1]: r for r in srows}
            col = {n: i for i, n in enumerate(head)}
            for r in rows[:3]:  # 前 3 名：重算的分數要跟 run 當場算的相同
                for name in shead[2:]:
                    a, b = float(r[col[name]]), float(scored[r[1]][shead.index(name)])
                    self.assertTrue(a == b or (np.isnan(a) and np.isnan(b)), f"{r[1]} {name}")
            self.assertEqual([r[1] for r in srows if r[0]], [r[1] for r in rows[:3]])  # 名次順序也相同
            # baseline：獨立從行情再算一次開場全倉的分數
            mk = load_market(*paths)
            data = load_config(cfg)
            base = run_strategies(expand(data), mk, keep_top=0, threads=1)
            want = summarize(base.baseline_metrics, base.windows)
            self.assertEqual(scored["baseline"][0], "")
            for name, v in want.items():
                a, b = float(v[0]), float(scored["baseline"][shead.index(name)])
                # 只有 1 列的 summarize 與 4 列的加總順序不同，末位可差 1 ulp，所以用相對誤差
                self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=0, equal_nan=True), f"{name} {a!r} {b!r}")
            # 對照：前 3 名彼此、與 baseline 的分數不是同一份常數
            vals = {tuple(r[2:]) for r in srows}
            self.assertGreater(len(vals), 1)
            self.assertNotEqual(scored["baseline"][2:], scored[rows[0][1]][2:])


# 淺門檻、買賣都有，確保前 3 名都有成交
ACTIVE = CONFIG.replace("from = -0.2, to = -0.02, step = 0.09", "from = -0.03, to = -0.01, step = 0.01")


class OrdersCsv(unittest.TestCase):
    def test_orders_cover_same_strategies_and_windows_as_windows_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic.write_csv(tmp, days=900)
            cfg = os.path.join(tmp, "c.toml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(ACTIVE)
            out = os.path.join(tmp, "out")
            self.assertEqual(main(["run", cfg, "--out", out]), 0)
            _, wrows = read(os.path.join(out, "windows.csv"))
            ohead, orows = read(os.path.join(out, "orders.csv"))
        self.assertEqual(ohead, "strategy,kind,start,end,cycle,tier,trade_date,trigger,price,units,fee,tax,"
                                "cash_before,cash_after,units_after".split(","))
        self.assertEqual({r[0] for r in orows}, {r[0] for r in wrows})  # 策略集合相同
        keys = {tuple(r[:5]) for r in wrows}
        self.assertTrue(all(tuple(r[:5]) in keys for r in orows))  # 窗口鍵都對得上
        self.assertTrue(all(r[2] <= r[6] <= r[3] for r in orows))  # 成交日在起迄之間（ISO 日期可直接比字串）
        self.assertTrue(any(r[0] != "baseline" for r in orows))
        self.assertTrue(any(r[0] == "baseline" for r in orows))
        self.assertEqual({r[5] for r in orows if r[0] == "baseline"}, {"baseline"})
        self.assertTrue({r[5] for r in orows if r[0] != "baseline"} <= {"買", "賣"})


def _expected_source(path):
    """從檔案本身獨立算出 run.json 該記的筆數、起訖與 SHA-256。"""
    with open(path, "rb") as f:
        raw = f.read()
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))[1:]
    return {"name": os.path.basename(path), "rows": len(rows), "start": rows[0][0], "end": rows[-1][0],
            "sha256": hashlib.sha256(raw).hexdigest()}


class RunRecord(unittest.TestCase):
    """run.json 記下對帳要的欄位；--data 蓋過設定檔的 [data] dir，不帶時照設定檔。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        cls.own = synthetic.write_csv(tmp, days=900)  # 設定檔 dir = "." 指到的行情
        cls.other_dir = os.path.join(tmp, "other")
        cls.other = synthetic.write_csv(cls.other_dir, days=950)  # 只有 --data 會讀到的行情
        cls.cfg = os.path.join(tmp, "c.toml")
        with open(cls.cfg, "w", encoding="utf-8") as f:
            f.write(CONFIG)
        cls.out_own = os.path.join(tmp, "own")
        cls.out_other = os.path.join(tmp, "other-out")
        cls.code_own = main(["run", cls.cfg, "--out", cls.out_own])
        cls.code_other = main(["run", cls.cfg, "--out", cls.out_other, "--data", cls.other_dir])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def record(self, out):
        with open(os.path.join(out, "run.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_record_has_every_field(self):
        self.assertEqual(self.code_own, 0)
        rec = self.record(self.out_own)
        self.assertEqual(rec["framework"]["name"], "tw-backtest")
        self.assertEqual(rec["framework"]["version"], metadata.version("tw-backtest"))
        self.assertIn("commit", rec["framework"])
        self.assertEqual(rec["runtime"], {"python": platform.python_version(), "numpy": np.__version__,
                                          "numba": numba.__version__})
        with open(self.cfg, "rb") as f:
            raw = f.read()
        self.assertEqual(rec["config"], {"name": "c.toml", "sha256": hashlib.sha256(raw).hexdigest(),
                                         "values": tomllib.loads(CONFIG)})
        self.assertIsInstance(rec["seconds"], float)
        self.assertGreaterEqual(rec["seconds"], 0)

    def test_data_option_overrides_config_dir(self):
        self.assertEqual((self.code_own, self.code_other), (0, 0))
        own = {"price_index": _expected_source(self.own[0]), "total_return": _expected_source(self.own[1])}
        other = {"price_index": _expected_source(self.other[0]), "total_return": _expected_source(self.other[1])}
        self.assertNotEqual(own, other)  # 兩份行情分得出來，下面兩條才有意義
        self.assertEqual(self.record(self.out_own)["inputs"], own)  # 不帶 --data：照設定檔
        self.assertEqual(self.record(self.out_other)["inputs"], other)  # 帶 --data：改讀那裡


if __name__ == "__main__":
    unittest.main()
