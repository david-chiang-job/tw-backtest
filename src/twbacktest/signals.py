"""訊號表：唯一把指標、條件組合、成交模型、重置規則串起來的模組。帳戶核心只讀它的輸出。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConfigError
from .fills import FILLS
from .grid import IndicatorCache, tier_trigger
from .tiers import RESETS


@dataclass(frozen=True)
class Signals:
    hit: np.ndarray  # [N, D, K] bool
    price: np.ndarray  # [N, D, K] float64，沒觸發為 NaN
    rearm: np.ndarray  # [N, D, K] bool
    order: np.ndarray  # [N, D, K] int8，第 j 個要處理的檔位編號，之後填 -1
    target: np.ndarray  # [N, K] float64
    side: np.ndarray  # [N, K] int8，1 買進、-1 賣出、0 空檔位
    both_sides: list  # 長度 N，每個是同一天買賣都觸發的日期索引


def day_order(hit, trigger, side, same_day):
    """每天的處理順序。hit、trigger 是 [D, K]，side 是 [K]；回傳 [D, K] int8。"""
    hit = np.asarray(hit, dtype=bool)
    trigger = np.asarray(trigger, dtype=float)
    side = np.asarray(side)
    first = 1 if same_day == "buy_first" else -1
    group = np.where(side == first, 0, 1)[None, :]
    primary = np.where(hit, group, 2)  # 沒觸發的排最後
    isnan = np.isnan(trigger)
    # 買進價高者先、賣出價低者先；NaN 同方向最後；其餘平手由 lexsort 的穩定性依檔位編號
    pk = np.where(isnan, 0.0, trigger * (-side[None, :]))
    idx = np.lexsort((pk, isnan, primary), axis=-1)
    count = hit.sum(axis=-1, keepdims=True)
    pos = np.arange(hit.shape[-1])[None, :]
    return np.where(pos < count, idx, -1).astype(np.int8)


def build_signals(strategies, market, cache=None):
    cache = cache or IndicatorCache(market.close)
    n = len(strategies)
    d = len(market.close)
    k = max((len(s.config.tiers) for s in strategies), default=0)
    if k > 127:
        raise ValueError(f"檔位數 {k} 超過 127，order 的 int8 放不下")
    hit = np.zeros((n, d, k), dtype=bool)
    price = np.full((n, d, k), np.nan)
    rearm = np.zeros((n, d, k), dtype=bool)
    order = np.full((n, d, k), -1, dtype=np.int8)
    target = np.zeros((n, k))
    side = np.zeros((n, k), dtype=np.int8)
    both = []
    for i, st in enumerate(strategies):
        fill = st.config.fill
        fn = FILLS.get(fill.model)
        trig = np.full((d, k), np.nan)
        for j, tier in enumerate(st.config.tiers):
            trig[:, j] = tier_trigger(tier, cache)
            try:
                h, p = fn(market.open, market.high, market.low, market.close, trig[:, j], tier.side, **fill.params)
            except ValueError as e:
                raise ConfigError(f"第 {i + 1} 組策略的檔位 {tier.name!r}：{e}") from None
            hit[i, :, j] = h
            price[i, :, j] = p
            rearm[i, :, j] = RESETS.get(tier.reset)(market.close, trig[:, j], tier.side)
            target[i, j] = tier.target_stock
            side[i, j] = tier.side
        order[i] = day_order(hit[i], trig, side[i], fill.same_day)
        both.append(np.flatnonzero((hit[i] & (side[i] == 1)).any(axis=1) & (hit[i] & (side[i] == -1)).any(axis=1)))
    return Signals(hit, price, rearm, order, target, side, both)
