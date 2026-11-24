"""條件組合：多條觸發價序列與方向 → 一條合成觸發價序列。

side：1 買進（at_most，觸發價越高越容易成立）、-1 賣出（at_least，越低越容易成立）。
不 import indicators；葉子的觸發價由呼叫端經 leaf_trigger 傳入。
"""

from __future__ import annotations

import warnings

import numpy as np

from .registry import Registry

COMBINATORS = Registry("條件組合")


def _stack(arrays):
    return np.vstack([np.asarray(a, dtype=float) for a in arrays])


@COMBINATORS.register("any")
def combine_any(arrays, side, *, points=None, threshold=None):
    t = _stack(arrays)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(t, axis=0) if side == 1 else np.nanmin(t, axis=0)


@COMBINATORS.register("all")
def combine_all(arrays, side, *, points=None, threshold=None):
    t = _stack(arrays)  # min/max 遇 NaN 會傳染，正好是「有任一 NaN 就不觸發」
    return t.min(axis=0) if side == 1 else t.max(axis=0)


@COMBINATORS.register("weighted")
def combine_weighted(arrays, side, *, points, threshold):
    t = _stack(arrays)
    nan = np.isnan(t)
    pts = np.where(nan, 0.0, np.asarray(points, dtype=float)[:, None])
    key = np.where(nan, np.inf, -side * t)  # 買進高到低、賣出低到高，NaN 墊底
    order = np.argsort(key, axis=0, kind="stable")
    v = np.take_along_axis(t, order, axis=0)
    cs = np.cumsum(np.take_along_axis(pts, order, axis=0), axis=0)
    reached = cs >= threshold
    idx = reached.argmax(axis=0)
    picked = v[idx, np.arange(t.shape[1])]
    # 同價位的條件分數一起加：累積到門檻的價位就是該價位，同價者排序先後不影響結果
    return np.where(reached.any(axis=0), picked, np.nan)


def combinator_name(node):
    """節點含哪個組合名稱的鍵；沒有或不只一個回傳 None。"""
    hits = [n for n in COMBINATORS.names() if n in node]
    return hits[0] if len(hits) == 1 else None


def evaluate(node, leaf_trigger, side):
    """遞迴走訪條件樹（單一條件有 "indicator" 鍵），葉子用 leaf_trigger(node) 取觸發價。"""
    if "indicator" in node:
        return np.asarray(leaf_trigger(node), dtype=float)
    name = combinator_name(node)
    children = node[name]
    arrays = [evaluate(c, leaf_trigger, side) for c in children]
    kw = {}
    if name == "weighted":
        kw = {"points": [c["points"] for c in children], "threshold": node["score_at_least"]}
    return COMBINATORS.get(name)(arrays, side, **kw)
