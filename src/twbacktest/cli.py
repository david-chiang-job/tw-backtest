"""命令列：run 跑整批並寫 scores.csv 與 windows.csv；score 只用逐窗結果重新評分排名。"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import fields

import numpy as np

from . import synthetic
from .config import ConfigError, load_config, shared_run
from .data import load_market
from .grid import expand
from .kernel import Accounts
from .runner import run_strategies
from .scoring import RANK_BY, WindowMetrics, rank, summarize

WINDOW_HEAD = ["strategy", "kind", "start", "end", "cycle"]
METRIC_NAMES = [f.name for f in fields(WindowMetrics)]
BASELINE = "baseline"


class _W:  # summarize 只需要 kind 與 key
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


def _num(x):
    return repr(float(x))


def _path_name(path):
    return ".".join(str(p) for p in path)


def _write_scores(path, strategies, result):
    axes = list(strategies[0].picks) if strategies else []
    score_names = list(result.summary)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strategy"] + [_path_name(p) for p in axes] + score_names)
        for r, i in enumerate(result.ranking, start=1):
            w.writerow([r, int(i)] + [repr(strategies[i].picks[p]) for p in axes]
                       + [_num(result.summary[k][i]) for k in score_names])


def _window_rows(label_rows, windows, dates):
    start = np.repeat(windows.starts, np.diff(windows.ptr))
    for name, metrics in label_rows:
        for e in range(len(windows.ends)):
            key = int(windows.key[e])
            yield [name, str(windows.kind[e]), str(dates[start[e]]), str(dates[windows.ends[e]]),
                   str(dates[key]) if key >= 0 else ""] + [_num(getattr(metrics, m)[e]) for m in METRIC_NAMES]


def _write_windows(path, result, dates):
    def one(m, row):
        return WindowMetrics(**{n: getattr(m, n)[row] for n in METRIC_NAMES})

    rows = [(str(i), one(result.top_metrics, j)) for j, i in enumerate(result.top)]
    rows.append((BASELINE, one(result.baseline_metrics, 0)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(WINDOW_HEAD + METRIC_NAMES)
        w.writerows(_window_rows(rows, result.windows, dates))


def read_windows(path):
    """讀 windows.csv，回傳 (策略標籤 list, WindowMetrics [S, E], 窗口 kind/key 物件)。分組鍵同一個日期對到同一個整數。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows[0] != WINDOW_HEAD + METRIC_NAMES:
        raise ValueError(f"windows.csv 欄位不符：{rows[0]}")
    labels = list(dict.fromkeys(r[0] for r in rows[1:]))
    by = {k: [] for k in labels}
    for r in rows[1:]:
        by[r[0]].append(r)
    first = by[labels[0]]
    sig = [(r[1], r[2], r[3], r[4]) for r in first]
    for k in labels:
        if [(r[1], r[2], r[3], r[4]) for r in by[k]] != sig:
            raise ValueError(f"策略 {k} 的窗口跟 {labels[0]} 不同")
    kind = np.array([r[1] for r in first])
    key = np.array([int(np.datetime64(r[4], "D").astype(np.int64)) if r[4] else -1 for r in first], dtype=np.int64)
    cols = {n: np.array([[float(r[5 + j]) for r in by[k]] for k in labels]) for j, n in enumerate(METRIC_NAMES)}
    return labels, WindowMetrics(**cols), _W(kind, key)


def score_file(path, by=RANK_BY):
    """回傳 (標籤, summary, 名次)；baseline 列算分數但不參與名次。"""
    labels, metrics, w = read_windows(path)
    summary = summarize(metrics, w)
    if by not in summary:
        raise KeyError(f"沒有分數欄 {by!r}；可用：{sorted(summary)}")
    idx = [i for i, k in enumerate(labels) if k != BASELINE]
    order = rank({by: summary[by][idx]}, by)
    return labels, summary, [idx[i] for i in order]


def _market(data, base):
    d = data.get("data", {})
    folder = os.path.join(base, d.get("dir", "."))
    return load_market(os.path.join(folder, d.get("price_index", synthetic.PRICE_FILE)),
                       os.path.join(folder, d.get("total_return", synthetic.TOTAL_RETURN_FILE)))


def cmd_run(a):
    cfg_path = os.path.abspath(a.config)
    data = load_config(cfg_path)
    strategies = expand(data)
    run = shared_run([s.config for s in strategies])
    market = _market(data, os.path.dirname(cfg_path))
    t0 = time.perf_counter()
    result = run_strategies(strategies, market, keep_top=run.keep_windows, threads=run.threads)
    secs = time.perf_counter() - t0
    os.makedirs(a.out, exist_ok=True)
    _write_scores(os.path.join(a.out, "scores.csv"), strategies, result)
    _write_windows(os.path.join(a.out, "windows.csv"), result, market.dates)
    print(f"組數 {len(strategies)}，每塊 {result.chunk_size} 組，執行緒 {result.threads}，{secs:.2f} 秒")
    for r, i in enumerate(result.ranking[:5], start=1):
        print(f"{r}. 策略 {int(i)} {RANK_BY}={result.summary[RANK_BY][i]:.6g} {dict((_path_name(p), v) for p, v in strategies[i].picks.items())}")


def cmd_score(a):
    labels, summary, order = score_file(a.windows, a.by)
    names = list(summary)
    out = open(a.out, "w", newline="", encoding="utf-8") if a.out else sys.stdout
    try:
        w = csv.writer(out)
        w.writerow(["rank", "strategy"] + names)
        for r, i in enumerate(order, start=1):
            w.writerow([r, labels[i]] + [_num(summary[k][i]) for k in names])
        for i, k in enumerate(labels):
            if k == BASELINE:
                w.writerow(["", k] + [_num(summary[n][i]) for n in names])
    finally:
        if a.out:
            out.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twbacktest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="跑整批策略")
    r.add_argument("config")
    r.add_argument("--out", required=True, help="輸出資料夾")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score", help="用 windows.csv 重新評分排名")
    s.add_argument("windows")
    s.add_argument("--by", default=RANK_BY)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (ConfigError, ValueError, KeyError, OSError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
