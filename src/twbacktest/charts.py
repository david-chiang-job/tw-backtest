"""圖表資料：替已另存成交的策略挑代表窗口，寫出畫圖要用的四份檔（Chart Data.json、Chart Prices.csv、
Chart Levels.csv、Chart Events.csv）。觸發價與指標線一律用框架算訊號的同一套函式；不碰評分那一趟。"""

from __future__ import annotations

import calendar
import csv
import json
import os
from datetime import date

import numpy as np

from .chart_svg import render_charts
from .config import leaf_params, shared_windows
from .grid import IndicatorCache, tier_trigger
from .indicators import INDICATORS
from .runner import TRIGGER_LAG
from .windows import find_cycles

DATA_FILE = "Chart Data.json"
PRICES_FILE = "Chart Prices.csv"
LEVELS_FILE = "Chart Levels.csv"
EVENTS_FILE = "Chart Events.csv"
SERIES = "total_return"
LOOKBACK_MONTHS = 12
DEFAULT_DRAWDOWN = 0.20

PRICES_HEAD = ["date", "series", "open", "high", "low", "close"]
LEVELS_HEAD = ["date", "series", "strategy", "level", "slot", "moving_average", "threshold",
               "negative_deviation", "target_weight"]
EVENTS_HEAD = ["strategy", "cycle", "window_start", "signal_date", "trade_date", "side", "slot", "price", "series"]


def _num(x):
    """跟其他 CSV 相同的數字寫法；NaN 寫空字串。"""
    x = float(x)
    return "" if x != x else repr(x)


def _months_back(d, months):
    m = d.year * 12 + d.month - 1 - months
    y, mo = divmod(m, 12)
    mo += 1
    return date(y, mo, min(d.day, calendar.monthrange(y, mo)[1]))


def _leaves(node):
    if "indicator" in node:
        yield node
        return
    for v in node.values():
        if isinstance(v, list):
            for c in v:
                yield from _leaves(c)


def _line_name(key):
    ind, params = key
    p = dict(params)
    if ind == "ma_deviation":
        return f"MA{p['days']}"
    if ind == "drawdown":
        return "Prior high"
    return f"{ind}({', '.join(f'{k}={v}' for k, v in params)})"


def _strategy_lines(config):
    """該策略所有檔位條件用到的不同 (指標, 非門檻參數)，照第一次出現的順序。"""
    keys = {}
    for tier in config.tiers:
        for leaf in _leaves(tier.when):
            keys.setdefault((leaf["indicator"], tuple(sorted(leaf_params(leaf).items()))), None)
    return [(k, _line_name(k)) for k in keys]


def pick_windows(market, windows, drawdown=DEFAULT_DRAWDOWN):
    """代表窗口，回傳 [(窗口列索引 e, 起跑日列索引 j)]。
    有 cycle1：每場循環一個，起跑日是前高往前 12 個曆月後的第一個交易日（不早於該場 cycle1 最早起跑日）。
    沒有：最長固定期裡終點最晚（同終點取起跑日最晚）的一個。"""
    kind = np.asarray(windows.kind)
    job_of = np.repeat(np.arange(len(windows.starts)), np.diff(windows.ptr))
    starts = windows.starts[job_of]
    one = np.flatnonzero(kind == "cycle1")
    dates = np.asarray(market.dates).astype("datetime64[D]")
    if len(one):
        peak, trough, _ = find_cycles(market.close, drawdown)
        peak_of = {int(t): int(p) for p, t in zip(peak, trough)}
        out = []
        for key in sorted({int(windows.key[e]) for e in one}):
            es = [int(e) for e in one if int(windows.key[e]) == key]
            floor = min(int(starts[e]) for e in es)
            want = np.datetime64(_months_back(dates[peak_of[key]].astype(object), LOOKBACK_MONTHS), "D")
            s = max(int(np.searchsorted(dates, want, side="left")), floor)
            hit = [e for e in es if int(starts[e]) == s]
            if not hit:
                raise ValueError(f"循環 {dates[key]} 選不出代表窗口（起跑日索引 {s}）")
            out.append((hit[0], int(job_of[hit[0]])))
        return out
    fixed = [e for e in range(len(kind)) if kind[e].endswith("d") and kind[e][:-1].isdigit()]
    if not fixed:
        return []
    longest = max(int(kind[e][:-1]) for e in fixed)
    cand = [e for e in fixed if int(kind[e][:-1]) == longest]
    e = max(cand, key=lambda e: (int(windows.ends[e]), int(starts[e])))
    return [(e, int(job_of[e]))]


def write_chart_data(out_dir, result, strategies, market, which="top"):
    """which 是 "top" 或 "kept"；策略集合是空的就什麼都不寫。"""
    ids, trades = (result.top, result.top_trades) if which == "top" else (result.kept, result.kept_trades)
    if not len(ids):
        return False
    windows, dates = result.windows, market.dates
    spec = shared_windows([s.config for s in strategies]).spec
    drawdown = (spec.get("cycles") or {}).get("drawdown", DEFAULT_DRAWDOWN)
    picked = pick_windows(market, windows, drawdown)
    cache = IndicatorCache(market.close)
    J = len(windows.starts)
    f = trades.fills

    chart_windows, strat_info, event_rows, level_rows = [], {}, [], []
    price_days = set()
    n = 0
    for r, i in enumerate(ids):
        cfg = strategies[i].config
        label = str(int(i))
        tiers = [{"slot": s, "name": t.name, "side": "buy" if t.side == 1 else "sell",
                  "target_stock": float(t.target_stock), "when": t.when}
                 for s, t in enumerate(cfg.tiers, start=1)]
        lines = _strategy_lines(cfg)
        strat_info[label] = {"tiers": tiers, "lines": [{"line": nm, "label": nm} for _, nm in lines]}
        lag = TRIGGER_LAG.get(cfg.fill.model)
        covered = set()
        for e, j in sorted(picked, key=lambda p: int(windows.starts[p[1]])):
            n += 1
            s_idx, e_idx = int(windows.starts[j]), int(windows.ends[e])
            key = int(windows.key[e])
            cycle = str(dates[key]) if key >= 0 else ""
            chart_windows.append({"strategy": label, "kind": str(windows.kind[e]), "cycle": cycle,
                                  "start": str(dates[s_idx]), "end": str(dates[e_idx]),
                                  "file": f"Chart {n:03d}.svg"})
            covered.update(range(s_idx, e_idx + 1))
            job = r * J + j
            lo, hi = int(f.offset[job]), int(f.offset[job + 1])
            hi = lo + int(np.searchsorted(f.day[lo:hi], e_idx, side="right"))
            for k in range(lo, hi):
                day, t = int(f.day[k]), int(f.tier[k])
                sig = str(dates[day - lag]) if lag is not None else ""
                event_rows.append([label, cycle, str(dates[s_idx]), sig, str(dates[day]),
                                   "buy" if cfg.tiers[t].side == 1 else "sell", t + 1, _num(f.price[k]), SERIES])
        price_days |= covered
        trig = [tier_trigger(t, cache) for t in cfg.tiers]
        base = [INDICATORS.get(ind).base(market.close, **dict(params)) for (ind, params), _ in lines]
        for d in sorted(covered):
            for s, t in enumerate(cfg.tiers, start=1):
                level_rows.append([str(dates[d]), SERIES, label, "buy" if t.side == 1 else "sell", s, "",
                                   _num(trig[s - 1][d]), "", _num(t.target_stock)])
            for (_, nm), b in zip(lines, base):
                level_rows.append([str(dates[d]), SERIES, label, "line", nm, _num(b[d]), "", "", ""])

    os.makedirs(out_dir, exist_ok=True)

    def src(s):
        return {"name": s.name, "sha256": s.sha256}

    doc = {"version": 1, "windows": chart_windows,
           "series": [{"id": SERIES, "label": "Total return index", "unit": "index points", "purpose": "ohlc"}],
           "strategies": strat_info,
           "inputs": {"price_index": src(market.price_index), "total_return": src(market.total_return)}}
    with open(os.path.join(out_dir, DATA_FILE), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    def write_csv(name, head, rows):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(rows)

    write_csv(PRICES_FILE, PRICES_HEAD,
              ([str(dates[d]), SERIES] + [_num(a[d]) for a in (market.open, market.high, market.low, market.close)]
               for d in sorted(price_days)))
    write_csv(LEVELS_FILE, LEVELS_HEAD, level_rows)
    write_csv(EVENTS_FILE, EVENTS_HEAD, event_rows)
    return True


def write_charts(out_dir, result, strategies, market, which="top"):
    """寫四份圖表資料檔再畫 SVG，回傳畫出的檔名清單；策略集合是空的就什麼都不做。"""
    if not write_chart_data(out_dir, result, strategies, market, which):
        return []
    return [name for _, name in render_charts(out_dir)]
