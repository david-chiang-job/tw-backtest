"""帳戶核心：按（策略 × 起跑日）平行，逐日計息、依 order 成交、扣費稅，逐窗輸出。

只讀訊號表的 hit、price、rearm、order、target、side，不認任何具體元件，也不自己排序。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange

from .config import Account, Costs


@dataclass(frozen=True)
class Accounts:
    nav: np.ndarray  # [N, E] float64，該終點當天收盤淨值（元）
    max_drawdown: np.ndarray  # [N, E] float64，起跑日到該終點的最大回撤（0 到 1），高點從本金起算
    lowest: np.ndarray  # [N, E] float64，起跑日到該終點每天收盤淨值的最小值（元）
    first_buy: np.ndarray  # [N, E] int64，第一筆買進成交的交易日索引，沒有為 -1


@dataclass(frozen=True)
class Fills:
    """逐筆成交，按 job = 策略 * J + 起跑日序 分組，各陣列長度 = 總成交筆數。

    第 job 組的成交是 [offset[job], offset[job + 1])，依成交順序排列，只含到該起跑日最後一個終點為止的成交。
    """

    offset: np.ndarray  # [N * J + 1] int64
    day: np.ndarray  # int64，成交日的交易日索引
    tier: np.ndarray  # int64，檔位索引
    price: np.ndarray  # float64，成交價
    qty: np.ndarray  # float64，正數買進、負數賣出
    fee: np.ndarray  # float64，手續費（元）
    tax: np.ndarray  # float64，證交稅（元），只在賣出
    cash_before: np.ndarray  # float64，成交前現金（已含當日計息）
    cash_after: np.ndarray  # float64
    units_after: np.ndarray  # float64，成交後持股


def cash_growth(dates, rate):
    """[D] float64；growth[0] = 1，growth[d] = (1 + rate) ** (相隔曆日數 / 365)。"""
    dates = np.asarray(dates, dtype="datetime64[D]")
    g = np.ones(len(dates), dtype=np.float64)
    if len(dates) > 1:
        days = (dates[1:] - dates[:-1]).astype(np.int64).astype(np.float64)
        g[1:] = (1.0 + rate) ** (days / 365.0)
    return g


@njit(parallel=True, cache=True)
def _kernel(hit, price, rearm, order, target, side, close, growth, starts, ends, ptr,
            cash0, unit, bf, sf, tax, nav_out, mdd_out, low_out, first_out):
    S, _, K = hit.shape
    J = starts.shape[0]
    for job in prange(S * J):
        s = job // J
        j = job - s * J
        st = starts[j]
        e = ptr[j]
        e1 = ptr[j + 1]
        if e == e1:
            continue
        last = ends[e1 - 1]
        cash = cash0
        units = 0.0
        peak = cash0
        lowest = np.inf
        mdd = 0.0
        armed = np.ones(K, dtype=np.bool_)  # 待命旗標，每檔一格（K 不限 64）
        first = -1
        for d in range(st, last + 1):
            if d > st:
                cash *= growth[d]
            for n in range(K):
                k = order[s, d, n]
                if k < 0:
                    break
                if not armed[k]:
                    continue
                p = price[s, d, k]
                nav = cash + units * p
                if side[s, k] > 0:
                    gap = target[s, k] * nav - units * p
                    if gap > 0:
                        q = math.floor(min(cash / (p * (1 + bf)), gap / p) / unit) * unit
                        if q > 0:
                            cash -= q * p * (1 + bf)
                            units += q
                            armed[k] = False
                            if first < 0:
                                first = d
                elif side[s, k] < 0:
                    gap = units * p - target[s, k] * nav
                    if gap > 0:
                        q = math.floor(min(units, gap / p) / unit) * unit
                        if q > 0:
                            cash += q * p * (1 - sf - tax)
                            units -= q
                            armed[k] = False
            for k in range(K):
                if (not armed[k]) and rearm[s, d, k]:
                    armed[k] = True
            eq = cash + units * close[d]
            if eq > peak:
                peak = eq
            dd = 1 - eq / peak
            if dd > mdd:
                mdd = dd
            if eq < lowest:
                lowest = eq
            while e < e1 and ends[e] == d:
                nav_out[s, e] = eq
                mdd_out[s, e] = mdd
                low_out[s, e] = lowest
                first_out[s, e] = first
                e += 1


@njit(parallel=True, cache=True)
def _kernel_fills(hit, price, rearm, order, target, side, close, growth, starts, ends, ptr,
                  cash0, unit, bf, sf, tax, nav_out, mdd_out, low_out, first_out,
                  write, offset, f_day, f_tier, f_price, f_qty, f_fee, f_tax, f_cb, f_ca, f_units):
    """與 _kernel 同一套規則，另記逐筆成交。write=False 只數筆數寫進 offset[1:]，True 才寫明細。

    規則改動要兩邊一起改；tests.test_kernel 的 FillsKernelSameAsScoring 逐位比對兩者。
    """
    S, _, K = hit.shape
    J = starts.shape[0]
    for job in prange(S * J):
        s = job // J
        j = job - s * J
        st = starts[j]
        e = ptr[j]
        e1 = ptr[j + 1]
        if e == e1:
            continue
        last = ends[e1 - 1]
        cash = cash0
        units = 0.0
        peak = cash0
        lowest = np.inf
        mdd = 0.0
        armed = np.ones(K, dtype=np.bool_)
        first = -1
        w = offset[job] if write else 0
        cnt = 0
        for d in range(st, last + 1):
            if d > st:
                cash *= growth[d]
            for n in range(K):
                k = order[s, d, n]
                if k < 0:
                    break
                if not armed[k]:
                    continue
                p = price[s, d, k]
                nav = cash + units * p
                if side[s, k] > 0:
                    gap = target[s, k] * nav - units * p
                    if gap > 0:
                        q = math.floor(min(cash / (p * (1 + bf)), gap / p) / unit) * unit
                        if q > 0:
                            before = cash
                            cash -= q * p * (1 + bf)
                            units += q
                            armed[k] = False
                            if first < 0:
                                first = d
                            if write:
                                f_day[w] = d
                                f_tier[w] = k
                                f_price[w] = p
                                f_qty[w] = q
                                f_fee[w] = q * p * bf
                                f_tax[w] = 0.0
                                f_cb[w] = before
                                f_ca[w] = cash
                                f_units[w] = units
                                w += 1
                            cnt += 1
                elif side[s, k] < 0:
                    gap = units * p - target[s, k] * nav
                    if gap > 0:
                        q = math.floor(min(units, gap / p) / unit) * unit
                        if q > 0:
                            before = cash
                            cash += q * p * (1 - sf - tax)
                            units -= q
                            armed[k] = False
                            if write:
                                f_day[w] = d
                                f_tier[w] = k
                                f_price[w] = p
                                f_qty[w] = -q
                                f_fee[w] = q * p * sf
                                f_tax[w] = q * p * tax
                                f_cb[w] = before
                                f_ca[w] = cash
                                f_units[w] = units
                                w += 1
                            cnt += 1
            for k in range(K):
                if (not armed[k]) and rearm[s, d, k]:
                    armed[k] = True
            eq = cash + units * close[d]
            if eq > peak:
                peak = eq
            dd = 1 - eq / peak
            if dd > mdd:
                mdd = dd
            if eq < lowest:
                lowest = eq
            while e < e1 and ends[e] == d:
                nav_out[s, e] = eq
                mdd_out[s, e] = mdd
                low_out[s, e] = lowest
                first_out[s, e] = first
                e += 1
        if not write:
            offset[job + 1] = cnt


def _prepare(signals, market, starts, ends, ptr, account, unit):
    """檢查輸入並轉成核心要的連續陣列；run_accounts 與 run_accounts_with_fills 共用。"""
    close = np.ascontiguousarray(market.close, dtype=np.float64)
    D = len(close)
    if len(market.dates) != D:
        raise ValueError("market.dates 與 market.close 長度不同")
    hit = np.ascontiguousarray(signals.hit)
    N, Ds, K = hit.shape
    if Ds != D:
        raise ValueError(f"訊號天數 {Ds} 與行情天數 {D} 不同")
    for name in ("price", "rearm", "order"):
        if getattr(signals, name).shape != (N, D, K):
            raise ValueError(f"{name} 形狀必須是 {(N, D, K)}")
    for name in ("target", "side"):
        if getattr(signals, name).shape != (N, K):
            raise ValueError(f"{name} 形狀必須是 {(N, K)}")
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    ptr = np.ascontiguousarray(ptr, dtype=np.int64)
    J = len(starts)
    if len(ptr) != J + 1:
        raise ValueError(f"ptr 長度必須是起跑日數 + 1 = {J + 1}（目前 {len(ptr)}）")
    if ptr[0] != 0:
        raise ValueError("ptr[0] 必須是 0")
    if ptr[-1] != len(ends):
        raise ValueError(f"ptr[-1] 必須等於 len(ends) = {len(ends)}（目前 {ptr[-1]}）")
    if np.any(np.diff(ptr) <= 0):
        raise ValueError("ptr 必須嚴格遞增：每個起跑日至少要有一個終點")
    if J and (starts.min() < 0 or starts.max() >= D):
        raise ValueError(f"起跑日索引必須在 0 到 {D - 1}")
    for j in range(J):
        seg = ends[ptr[j] : ptr[j + 1]]
        if np.any(np.diff(seg) < 0):
            raise ValueError(f"第 {j} 個起跑日的終點不可遞減")
        if seg[0] < starts[j]:
            raise ValueError(f"第 {j} 個起跑日的終點不可早於起跑日")
        if seg[-1] >= D:
            raise ValueError(f"第 {j} 個起跑日的終點必須小於 {D}")
    if not unit > 0:
        raise ValueError("unit 必須大於 0")
    growth = cash_growth(market.dates, account.cash_rate)
    args = (
        hit, np.ascontiguousarray(signals.price, dtype=np.float64), np.ascontiguousarray(signals.rearm),
        np.ascontiguousarray(signals.order), np.ascontiguousarray(signals.target, dtype=np.float64),
        np.ascontiguousarray(signals.side), close, growth, starts, ends, ptr,
    )
    return args, N, J, len(ends)


def run_accounts(signals, market, starts, ends, ptr, account=Account(), costs=Costs(), threads=None, unit=1.0):
    args, N, J, E = _prepare(signals, market, starts, ends, ptr, account, unit)
    nav = np.full((N, E), np.nan)
    mdd = np.full((N, E), np.nan)
    low = np.full((N, E), np.nan)
    first = np.full((N, E), -1, dtype=np.int64)
    old = numba.get_num_threads()
    if threads is not None:
        numba.set_num_threads(int(threads))
    try:
        _kernel(
            *args,
            float(account.cash), float(unit), float(costs.buy_fee), float(costs.sell_fee), float(costs.sell_tax),
            nav, mdd, low, first,
        )
    finally:
        if threads is not None:
            numba.set_num_threads(old)
    return Accounts(nav, mdd, low, first)


def run_accounts_with_fills(signals, market, starts, ends, ptr, account=Account(), costs=Costs(), threads=None, unit=1.0):
    """與 run_accounts 同樣的輸入，回傳 (Accounts, Fills)。先數成交筆數再配置，記憶體只跟實際成交筆數走。"""
    args, N, J, E = _prepare(signals, market, starts, ends, ptr, account, unit)
    scalars = (float(account.cash), float(unit), float(costs.buy_fee), float(costs.sell_fee), float(costs.sell_tax))
    nav = np.full((N, E), np.nan)
    mdd = np.full((N, E), np.nan)
    low = np.full((N, E), np.nan)
    first = np.full((N, E), -1, dtype=np.int64)
    offset = np.zeros(N * J + 1, dtype=np.int64)
    empty_i, empty_f = np.zeros(0, dtype=np.int64), np.zeros(0)
    old = numba.get_num_threads()
    if threads is not None:
        numba.set_num_threads(int(threads))
    try:
        _kernel_fills(*args, *scalars, nav, mdd, low, first, False, offset,
                      empty_i, empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)
        np.cumsum(offset, out=offset)
        total = int(offset[-1])
        day, tier = np.empty(total, dtype=np.int64), np.empty(total, dtype=np.int64)
        cols = [np.empty(total) for _ in range(7)]
        # 第二趟 offset 只讀：第 job 組從 offset[job] 開始寫
        _kernel_fills(*args, *scalars, nav, mdd, low, first, True, offset, day, tier, *cols)
    finally:
        if threads is not None:
            numba.set_num_threads(old)
    return Accounts(nav, mdd, low, first), Fills(offset, day, tier, *cols)
