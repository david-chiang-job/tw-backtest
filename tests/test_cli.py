import csv
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
