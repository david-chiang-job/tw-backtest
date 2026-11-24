"""窗口登記表：交易日與報酬指數收盤 → 起跑日與終點。產生器只看行情，不認帳戶核心。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import Registry

WINDOWS = Registry("窗口產生器")


@dataclass(frozen=True)
class Windows:
    starts: np.ndarray  # [J] int64，起跑日索引，嚴格遞增
    ptr: np.ndarray  # [J + 1] int64
    ends: np.ndarray  # [E] int64，每個起跑日內非遞減
    kind: np.ndarray  # [E] str，窗口種類
    key: np.ndarray  # [E] int64，分組鍵：循環是首場谷底的交易日索引，固定期 -1


def _check_counts(name, xs):
    ok = isinstance(xs, (list, tuple)) and len(xs) > 0
    ok = ok and all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 1 for x in xs)
    ok = ok and len(set(int(x) for x in xs)) == len(xs)
    if not ok:
        raise ValueError(f"{name} 必須是非空、不重複的正整數清單，收到 {xs!r}")
    return [int(x) for x in xs]


@WINDOWS.register("fixed")
def fixed(dates, close, lengths=(240, 720, 1200)):
    """固定期：每個交易日開盤起跑，第 N 個交易日收盤結算（end = start + N - 1）。行情不夠長的不產生。"""
    D = len(dates)
    s, e, k = [], [], []
    for n in _check_counts("lengths", lengths):
        count = max(D - n + 1, 0)
        start = np.arange(count, dtype=np.int64)
        s.append(start)
        e.append(start + (n - 1))
        k.append(np.full(count, f"{n}d"))
    if not s:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, str), np.empty(0, np.int64))
    start = np.concatenate(s)
    return start, np.concatenate(e), np.concatenate(k), np.full(len(start), -1, np.int64)


def find_cycles(close, drawdown=0.20):
    """回傳三個等長 int64 陣列 (peak, trough, recovery)，是每場完整循環的交易日索引，依時間排序。

    收盤 >= 目前前高時，若前高之後最低收盤 <= 前高 * (1 - drawdown) 就記一場，當天成為新前高；
    否則只在收盤嚴格低於目前最低時更新谷底。行情結束前沒收復的下跌不算。
    """
    if not (isinstance(drawdown, (int, float)) and not isinstance(drawdown, bool) and 0 < drawdown < 1):
        raise ValueError(f"drawdown 必須在 0 與 1 之間、不含兩端，收到 {drawdown!r}")
    close = np.asarray(close, dtype=float)
    peaks, troughs, recs = [], [], []
    if len(close):
        peak, pi, low, li = close[0], 0, close[0], 0
        for i in range(1, len(close)):
            c = close[i]
            if c >= peak:
                if low <= peak * (1 - drawdown):
                    peaks.append(pi)
                    troughs.append(li)
                    recs.append(i)
                peak, pi, low, li = c, i, c, i
            elif c < low:
                low, li = c, i
    return tuple(np.array(x, dtype=np.int64) for x in (peaks, troughs, recs))


@WINDOWS.register("cycles")
def cycles(dates, close, counts=(1, 2, 3), drawdown=0.20):
    """多空循環：起跑日配對第一場谷底在當天或之後的循環（第 k 場），N 場窗口在第 k + N - 1 場收復日結算。
    kind 是 f"cycle{場數}"，key 是第 k 場的谷底索引；場次不足的起跑日省略該長度。"""
    counts = _check_counts("counts", counts)
    _, trough, recovery = find_cycles(close, drawdown)
    n = len(trough)
    last = int(trough[-1]) + 1 if n else 0  # 起跑日超過最後一個谷底就沒有可配對的循環
    start = np.arange(last, dtype=np.int64)
    first = np.searchsorted(trough, start, side="left")
    s, e, k, g = [], [], [], []
    for c in counts:
        ok = first + (c - 1) < n
        idx = first[ok]
        s.append(start[ok])
        e.append(recovery[idx + (c - 1)])
        k.append(np.full(len(idx), f"cycle{c}"))
        g.append(trough[idx])
    if not s:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, str), np.empty(0, np.int64))
    return np.concatenate(s), np.concatenate(e), np.concatenate(k), np.concatenate(g)


def build_windows(dates, close, spec, score_end=None) -> Windows:
    """spec 是 {產生器名稱: 參數 dict}；合併輸出、套 score_end、依 (start, end) 穩定排序後組成 CSR。"""
    parts = [WINDOWS.get(name)(dates, close, **(params or {})) for name, params in spec.items()]
    if parts:
        start, end, kind, key = (np.concatenate([p[i] for p in parts]) for i in range(4))
        start, end, key = start.astype(np.int64), end.astype(np.int64), key.astype(np.int64)
        kind = kind.astype(str)
    else:
        start = end = key = np.empty(0, np.int64)
        kind = np.empty(0, str)
    if score_end is not None:
        keep = ~(np.asarray(dates)[end] > np.datetime64(score_end, "D"))
        start, end, kind, key = start[keep], end[keep], kind[keep], key[keep]
    order = np.lexsort((end, start))  # lexsort 是穩定排序
    start, end, kind, key = start[order], end[order], kind[order], key[order]
    starts, counts = np.unique(start, return_counts=True)
    ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return Windows(starts.astype(np.int64), ptr, end, kind, key)
