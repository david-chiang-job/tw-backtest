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


@WINDOWS.register("fixed")
def fixed(dates, close, lengths=(240, 720, 1200)):
    """固定期：每個交易日開盤起跑，第 N 個交易日收盤結算（end = start + N - 1）。行情不夠長的不產生。"""
    D = len(dates)
    s, e, k = [], [], []
    for n in lengths:
        n = int(n)
        if n < 1:
            raise ValueError(f"固定期天數必須 >= 1，收到 {n!r}")
        count = max(D - n + 1, 0)
        start = np.arange(count, dtype=np.int64)
        s.append(start)
        e.append(start + (n - 1))
        k.append(np.full(count, f"{n}d"))
    if not s:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, str), np.empty(0, np.int64))
    start = np.concatenate(s)
    return start, np.concatenate(e), np.concatenate(k), np.full(len(start), -1, np.int64)


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
