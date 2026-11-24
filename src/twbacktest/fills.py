"""成交模型登記表：每天開高低收、觸發價序列與方向 → 每天觸發與否、成交價。純函式，不保存跨日狀態。"""

from __future__ import annotations

import numpy as np

from .registry import Registry

FILLS = Registry("成交模型")


def _check_band(band):
    try:
        lo, hi = band
        lo, hi = float(lo), float(hi)
        ok = 0.0 <= lo <= hi <= 1.0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"band 必須是兩個數且滿足 0 <= 下 <= 上 <= 1，收到 {band!r}")
    return lo, hi


@FILLS.register("touch")
def touch(open, high, low, close, trigger, side, band=(0.10, 0.90)):
    """觸價成交：開盤與收盤都算進區間，只有影線最外側 band 以外的價位不算。"""
    lo, hi = _check_band(band)
    rng = high - low
    lower = np.minimum(np.minimum(open, close), low + lo * rng)
    upper = np.maximum(np.maximum(open, close), low + hi * rng)
    with np.errstate(invalid="ignore"):  # trigger 為 NaN 時比較結果為 False
        if side == 1:
            hit = lower <= trigger
            gap = open <= trigger
        else:
            hit = upper >= trigger
            gap = open >= trigger
    hit = hit & ~np.isnan(trigger)
    price = np.where(hit, np.where(gap, open, trigger), np.nan)
    return hit, price
