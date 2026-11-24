"""重置規則登記表：每天收盤、觸發價序列與方向 → 每天收盤後等待中的檔位能否重新待命。

純函式，只產出陣列；待命與等待的跨日狀態在帳戶核心。
"""

from __future__ import annotations

import numpy as np

from .registry import Registry

RESETS = Registry("重置規則")


@RESETS.register("recross")
def recross(close, trigger, side):
    """收盤回到觸發價的不利側（買進 close > P、賣出 close < P，嚴格不等）；P 為 NaN 為 False。"""
    with np.errstate(invalid="ignore"):
        return close > trigger if side == 1 else close < trigger


@RESETS.register("new_high")
def new_high(close, trigger, side):
    """收盤不低於截至當天的全歷史最高收盤；不分買賣方向。"""
    return close >= np.maximum.accumulate(close)


@RESETS.register("none")
def none(close, trigger, side):
    """每天都可重新待命。"""
    return np.ones(len(close), dtype=bool)
