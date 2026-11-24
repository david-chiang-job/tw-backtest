"""執行器：策略分塊送進帳戶核心，每塊評分完只留分數，逐窗結果只為指定的策略另存。"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .baseline import run_baseline
from .config import shared_account, shared_windows
from .grid import IndicatorCache
from .kernel import Accounts, run_accounts
from .scoring import RANK_BY, WindowMetrics, rank, summarize, window_metrics
from .signals import build_signals
from .windows import Windows, build_windows


@dataclass(frozen=True)
class RunResult:
    windows: Windows  # 整批共用的窗口
    summary: dict  # {"種類.指標.評分名稱": [N]}，N 是全部策略，順序同輸入
    ranking: np.ndarray  # [N] 策略索引，分數由高到低
    kept: tuple  # 另存逐窗結果的策略索引，照呼叫端給的順序（去重）
    kept_metrics: WindowMetrics | None  # 各欄 [len(kept), E]，列對應 kept
    kept_accounts: Accounts | None  # 各欄 [len(kept), E]，列對應 kept


def _rows(obj, idx):
    return type(obj)(**{f.name: np.asarray(getattr(obj, f.name))[idx] for f in fields(obj)})


def run_strategies(strategies, market, chunk_size, keep=(), rank_by=RANK_BY, threads=None) -> RunResult:
    """strategies 是 grid.expand 的輸出；每塊 chunk_size 組。keep 是要另存逐窗結果的策略索引。"""
    strategies = list(strategies)
    n = len(strategies)
    if n == 0:
        raise ValueError("沒有策略")
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError(f"chunk_size 必須是正整數（目前 {chunk_size!r}）")
    chunk_size = int(chunk_size)
    kept = tuple(dict.fromkeys(int(i) for i in keep))
    for i in kept:
        if not 0 <= i < n:
            raise ValueError(f"keep 索引 {i} 超出 0 到 {n - 1}")

    configs = [s.config for s in strategies]
    wspec = shared_windows(configs)
    account, costs = shared_account(configs)
    windows = build_windows(market.dates, market.close, wspec.spec, wspec.score_end)
    baseline = run_baseline(market, windows, account, costs, threads=threads)  # 跟策略無關，只算一次
    cache = IndicatorCache(market.close)  # 整次執行共用

    parts = []  # 每塊的分數
    kept_m, kept_a = [], []  # 每塊中被指定的列（依 kept 的順序之後再重排）
    kept_ids = []
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        sig = build_signals(strategies[lo:hi], market, cache=cache)
        acc = run_accounts(sig, market, windows.starts, windows.ends, windows.ptr, account, costs, threads=threads)
        metrics = window_metrics(acc, baseline, windows, market.dates, account.cash)
        parts.append(summarize(metrics, windows))
        local = [i - lo for i in kept if lo <= i < hi]
        if local:
            kept_ids.extend(i + lo for i in local)
            kept_m.append(_rows(metrics, local))
            kept_a.append(_rows(acc, local))

    summary = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    km = ka = None
    if kept:
        order = [kept_ids.index(i) for i in kept]
        km = _rows(WindowMetrics(**{f.name: np.concatenate([getattr(m, f.name) for m in kept_m]) for f in fields(WindowMetrics)}), order)
        ka = _rows(Accounts(**{f.name: np.concatenate([getattr(a, f.name) for a in kept_a]) for f in fields(Accounts)}), order)
    return RunResult(windows, summary, rank(summary, rank_by), kept, km, ka)
