"""執行器：策略分塊送進帳戶核心，每塊評分完只留分數，逐窗結果只為指定的策略與分數前 N 名另存。"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

import numba
import numpy as np

from .baseline import run_baseline, run_baseline_fills
from .config import shared_account, shared_windows
from .grid import IndicatorCache, tier_trigger
from .kernel import Accounts, Fills, run_accounts, run_accounts_with_fills
from .scoring import RANK_BY, WindowMetrics, rank, summarize, window_metrics
from .signals import build_signals
from .windows import Windows, build_windows

MEMORY_BUDGET = 512 * 1024 * 1024  # 自動分塊時每塊可用的位元組數
_METRIC_COLS = len(fields(WindowMetrics))
_ACCOUNT_COLS = len(fields(Accounts))
_METRIC_TEMPS = 12  # window_metrics 內暫存的 [E] 陣列張數（估計值，偏高）
_SIGNAL_TEMPS = 2  # build_signals 建表時的暫存倍數（估計值，偏高）


# 成交模型判斷第 d 天成交時比對的是第 d - lag 天的觸發價；表裡沒有的自訂模型，trigger 欄寫 NaN。
TRIGGER_LAG = {"touch": 0, "close_next_open": 1}


@dataclass(frozen=True)
class Trades:
    """一組策略的逐筆成交：fills 的第 r 列策略是 [offset[r * J], offset[(r + 1) * J])，J 是起跑日數。"""

    fills: Fills
    trigger: np.ndarray  # [總筆數] float64，成交模型比對的觸發價；baseline 是當天開盤價


@dataclass(frozen=True)
class RunResult:
    windows: Windows  # 整批共用的窗口
    summary: dict  # {"種類.指標.評分名稱": [N]}，N 是全部策略，順序同輸入
    ranking: np.ndarray  # [N] 策略索引，分數由高到低
    kept: tuple  # 呼叫端指定另存逐窗結果的策略索引，照給的順序（去重）
    kept_metrics: WindowMetrics | None  # 各欄 [len(kept), E]
    kept_accounts: Accounts | None  # 各欄 [len(kept), E]
    top: tuple  # 分數前 keep_top 名的策略索引，順序同 ranking[:keep_top]
    top_metrics: WindowMetrics | None  # 各欄 [len(top), E]
    top_accounts: Accounts | None  # 各欄 [len(top), E]
    baseline_metrics: WindowMetrics  # 開場全倉的逐窗指標，各欄 [1, E]
    baseline_accounts: Accounts  # 開場全倉的帳戶結果，各欄 [1, E]
    kept_trades: Trades | None  # kept 各策略的逐筆成交，列順序同 kept_metrics
    top_trades: Trades | None  # top 各策略的逐筆成交，列順序同 top_metrics
    baseline_trades: Trades  # 開場全倉的逐筆成交，只有 1 列
    chunk_size: int  # 實際用的每塊組數
    threads: int | None  # 實際用的執行緒數


def resolve_threads(threads):
    """"auto" 是全部核心減 1（至少 1、不超過 numba 上限）；正整數照用；None 交給 numba 預設。"""
    if threads is None:
        return None
    if threads == "auto":
        return max(1, min(numba.config.NUMBA_NUM_THREADS, (os.cpu_count() or 2) - 1))
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError(f'threads 必須是 "auto"、None 或正整數（目前 {threads!r}）')
    return int(threads)


def auto_chunk_size(n_days, n_tiers, n_windows, budget=MEMORY_BUDGET):
    """依記憶體預算估一塊的組數，至少 1。每組：訊號表 D×K×11 bytes（含建表暫存）、Accounts 與 WindowMetrics 各欄 [E] float64、評分暫存。"""
    per = n_days * n_tiers * 11 * _SIGNAL_TEMPS + n_windows * 8 * (_ACCOUNT_COLS + _METRIC_COLS + _METRIC_TEMPS)
    return max(1, int(budget // max(per, 1)))


def _rows(obj, idx):
    return type(obj)(**{f.name: np.asarray(getattr(obj, f.name))[idx] for f in fields(obj)})


def _cat(cls, parts):
    return cls(**{f.name: np.concatenate([np.asarray(getattr(p, f.name)) for p in parts]) for f in fields(cls)})


def _cat_fills(parts):
    cols = {f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(Fills) if f.name != "offset"}
    base = np.cumsum([0] + [int(p.offset[-1]) for p in parts])
    offset = np.concatenate([parts[0].offset[:1]] + [p.offset[1:] + b for p, b in zip(parts, base)])
    return Fills(offset=offset, **cols)


def _take_rows(trades, n_jobs, rows):
    """取出第 rows 列策略（照給的順序）的成交。"""
    off = trades.fills.offset
    idx = np.concatenate([np.arange(off[r * n_jobs], off[(r + 1) * n_jobs]) for r in rows]).astype(np.int64)
    counts = np.concatenate([np.diff(off[r * n_jobs : (r + 1) * n_jobs + 1]) for r in rows])
    new = {f.name: getattr(trades.fills, f.name)[idx] for f in fields(Fills) if f.name != "offset"}
    return Trades(Fills(offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64), **new), trades.trigger[idx])


def _run_trades(strategies, market, windows, account, costs, threads, cache, chunk_size):
    """替給定的策略（整批、分塊只為省記憶體）跑記成交的核心，並另算每筆成交的觸發價。"""
    parts, trig = [], []
    for lo in range(0, len(strategies), chunk_size):
        chunk = strategies[lo : lo + chunk_size]
        sig = build_signals(chunk, market, cache=cache)
        _, f = run_accounts_with_fills(sig, market, windows.starts, windows.ends, windows.ptr, account, costs,
                                       threads=threads)
        J = len(windows.starts)
        t = np.full(len(f.day), np.nan)
        for r, st in enumerate(chunk):
            lag = TRIGGER_LAG.get(st.config.fill.model)
            lo_, hi_ = f.offset[r * J], f.offset[(r + 1) * J]
            if lag is None or lo_ == hi_:
                continue
            tr = np.stack([tier_trigger(tier, cache) for tier in st.config.tiers], axis=1)  # [D, 該策略檔位數]
            t[lo_:hi_] = tr[f.day[lo_:hi_] - lag, f.tier[lo_:hi_]]
        parts.append(f)
        trig.append(t)
    return Trades(_cat_fills(parts), np.concatenate(trig))


def run_strategies(strategies, market, chunk_size=None, keep=(), keep_top=0, rank_by=RANK_BY,
                   threads="auto", memory_budget=MEMORY_BUDGET) -> RunResult:
    """strategies 是 grid.expand 的輸出；每塊 chunk_size 組（None 依 memory_budget 自動決定）。
    keep 是要另存逐窗結果的策略索引；keep_top 是另存分數（rank_by）前幾名；兩者可並存。"""
    strategies = list(strategies)
    n = len(strategies)
    if n == 0:
        raise ValueError("沒有策略")
    if chunk_size is not None and (int(chunk_size) != chunk_size or chunk_size < 1):
        raise ValueError(f"chunk_size 必須是正整數或 None（目前 {chunk_size!r}）")
    if int(keep_top) != keep_top or keep_top < 0:
        raise ValueError(f"keep_top 必須是 0 以上的整數（目前 {keep_top!r}）")
    keep_top = min(int(keep_top), n)
    kept = tuple(dict.fromkeys(int(i) for i in keep))
    for i in kept:
        if not 0 <= i < n:
            raise ValueError(f"keep 索引 {i} 超出 0 到 {n - 1}")
    threads = resolve_threads(threads)

    configs = [s.config for s in strategies]
    wspec = shared_windows(configs)
    account, costs = shared_account(configs)
    windows = build_windows(market.dates, market.close, wspec.spec, wspec.score_end)
    baseline = run_baseline(market, windows, account, costs, threads=threads)  # 跟策略無關，只算一次
    baseline_metrics = window_metrics(baseline, baseline, windows, market.dates, account.cash)
    # 開場全倉自己當基準，所以 excess 兩欄恆為 0
    if chunk_size is None:
        k = max(len(s.config.tiers) for s in strategies)
        chunk_size = auto_chunk_size(len(market.close), k, len(windows.ends), memory_budget)
    chunk_size = int(chunk_size)
    cache = IndicatorCache(market.close)  # 整次執行共用

    parts = []
    kept_m, kept_a, kept_ids = [], [], []
    top_ids = np.empty(0, dtype=np.int64)  # 目前前 N 名的全域索引，索引遞增
    top_score = np.empty(0)
    top_m = top_a = None
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        sig = build_signals(strategies[lo:hi], market, cache=cache)
        acc = run_accounts(sig, market, windows.starts, windows.ends, windows.ptr, account, costs, threads=threads)
        metrics = window_metrics(acc, baseline, windows, market.dates, account.cash)
        summary = summarize(metrics, windows)
        parts.append(summary)
        local = [i - lo for i in kept if lo <= i < hi]
        if local:
            kept_ids.extend(i + lo for i in local)
            kept_m.append(_rows(metrics, local))
            kept_a.append(_rows(acc, local))
        if keep_top:
            # 本塊前 N 名與既有前 N 名合併後重排；合併時照全域索引遞增，同分才會照索引小的先
            cand = rank(summary, rank_by)[:keep_top]
            ids = np.concatenate([top_ids, cand + lo])
            score = np.concatenate([top_score, np.asarray(summary[rank_by])[cand]])
            ms = [_rows(metrics, cand)] if top_m is None else [top_m, _rows(metrics, cand)]
            as_ = [_rows(acc, cand)] if top_a is None else [top_a, _rows(acc, cand)]
            allm, alla = _cat(WindowMetrics, ms), _cat(Accounts, as_)
            asc = np.argsort(ids, kind="stable")
            order = asc[rank({rank_by: score[asc]}, rank_by)[:keep_top]]
            top_ids, top_score = ids[order], score[order]
            top_m, top_a = _rows(allm, order), _rows(alla, order)

    summary = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ranking = rank(summary, rank_by)

    km = ka = None
    if kept:
        order = [kept_ids.index(i) for i in kept]
        km = _rows(_cat(WindowMetrics, kept_m), order)
        ka = _rows(_cat(Accounts, kept_a), order)
    # 第二趟：評分與 top 都定案後，只替 kept ∪ top 與 baseline 另跑記成交的核心（整批一次，不受評分分塊影響）
    top = tuple(int(i) for i in top_ids)
    sel = list(dict.fromkeys(kept + top))
    kt = tt = None
    if sel:
        trades = _run_trades([strategies[i] for i in sel], market, windows, account, costs, threads, cache, chunk_size)
        row = {i: r for r, i in enumerate(sel)}
        J = len(windows.starts)
        kt = _take_rows(trades, J, [row[i] for i in kept]) if kept else None
        tt = _take_rows(trades, J, [row[i] for i in top]) if top else None
    _, bf = run_baseline_fills(market, windows, account, costs, threads=threads)
    bt = Trades(bf, np.asarray(market.open, dtype=np.float64)[bf.day])
    return RunResult(windows, summary, ranking, kept, km, ka, top, top_m, top_a,
                     baseline_metrics, baseline, kt, tt, bt, chunk_size, threads)
