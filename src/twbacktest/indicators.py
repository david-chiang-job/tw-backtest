"""指標：整段報酬指數收盤與參數 → 每天的觸發價序列（算不出填 NaN）。

切法：每個指標登記一個「基準函式」base(close, **非門檻參數) -> 陣列，
與門檻無關，上層可依 (名稱, 非門檻參數) 快取；
觸發價 = apply_threshold(基準, 門檻)，預設為 基準 × (1 + 門檻)，便宜且可套多個門檻。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .registry import Registry

DIRECTIONS = ("at_most", "at_least")


@dataclass(frozen=True)
class Indicator:
    name: str
    base: Callable[..., np.ndarray]
    apply_threshold: Callable[[np.ndarray, float], np.ndarray]

    def trigger(self, close, threshold, **params):
        return self.apply_threshold(self.base(close, **params), threshold)


INDICATORS = Registry("指標")


def _times_one_plus(base, threshold):
    return base * (1.0 + threshold)


def register_indicator(name, apply_threshold=_times_one_plus):
    """decorator：被裝飾的函式是基準函式 base(close, **參數)，回傳 numpy 陣列。"""

    def deco(fn):
        INDICATORS.register(name)(Indicator(name, fn, apply_threshold))
        return fn

    return deco


@register_indicator("ma_deviation")
def ma_deviation(close, *, days):
    """第 t 天的基準 = close[t-days..t-1] 平均（不含當天）；t < days 為 NaN。"""
    close = np.asarray(close, dtype=float)
    if days < 1:
        raise ValueError("days 必須 >= 1")
    out = np.full(len(close), np.nan)
    if len(close) <= days:
        return out
    c = np.concatenate(([0.0], np.cumsum(close)))
    out[days:] = (c[days : len(close)] - c[: len(close) - days]) / days
    return out
