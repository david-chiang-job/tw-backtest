"""跨週期無前視重採樣：日線轉週線 OHLCV 與映射回日線序列。

防前視設計：
- resample_weekly 依 ISO 週 (iso_year, iso_week) 將日線彙總為週線，並以該週最後一個
  交易日作為 week_end_date。
- weekly_to_daily_ffill 預設 strict_prior_week=True：任意日線交易日 d 只取
  week_end_date < d 的最新已定案週線，確保當週盤中或收盤絕不偷看本週最後交易日收盤，
  且把日線截斷到任意一天重算時，該日結果與全序列完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeeklyBars:
    week_end_dates: np.ndarray  # datetime64[D]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray | None
    turnover: np.ndarray | None
    bar_end_indices: np.ndarray  # int64，每週最後一個交易日在原日線的索引


def _as_dates(dates, name: str) -> np.ndarray:
    d = np.asarray(dates, dtype="datetime64[D]")
    if d.ndim != 1:
        raise ValueError(f"{name} 必須是一維日期陣列（目前 ndim={d.ndim}）")
    if len(d) > 1 and np.any(d[1:] <= d[:-1]):
        raise ValueError(f"{name} 必須嚴格遞增")
    return d


def resample_weekly(
    dates,
    open_,
    high,
    low,
    close,
    volume=None,
    turnover=None,
) -> WeeklyBars:
    """將日線 OHLCV（與可選 turnover）依 ISO 週彙總為週線。"""
    d = _as_dates(dates, "dates")
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(d)
    for name, arr in (("open_", o), ("high", h), ("low", lo), ("close", c)):
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"{name} 長度必須與 dates 相同（{n}）")

    vol = None
    if volume is not None:
        vol = np.asarray(volume, dtype=np.float64)
        if vol.ndim != 1 or len(vol) != n:
            raise ValueError(f"volume 長度必須與 dates 相同（{n}）")

    turn = None
    if turnover is not None:
        turn = np.asarray(turnover, dtype=np.float64)
        if turn.ndim != 1 or len(turn) != n:
            raise ValueError(f"turnover 長度必須與 dates 相同（{n}）")

    if n == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return WeeklyBars(
            week_end_dates=np.empty(0, dtype="datetime64[D]"),
            open=empty_f,
            high=empty_f.copy(),
            low=empty_f.copy(),
            close=empty_f.copy(),
            volume=empty_f.copy() if vol is not None else None,
            turnover=empty_f.copy() if turn is not None else None,
            bar_end_indices=np.empty(0, dtype=np.int64),
        )

    py_dates = d.astype(object)
    groups: list[tuple[int, int]] = []
    start = 0
    prev_key = py_dates[0].isocalendar()[:2]
    for i in range(1, n):
        key = py_dates[i].isocalendar()[:2]
        if key != prev_key:
            groups.append((start, i))
            start = i
            prev_key = key
    groups.append((start, n))

    m = len(groups)
    w_dates = np.empty(m, dtype="datetime64[D]")
    w_open = np.empty(m, dtype=np.float64)
    w_high = np.empty(m, dtype=np.float64)
    w_low = np.empty(m, dtype=np.float64)
    w_close = np.empty(m, dtype=np.float64)
    w_vol = np.empty(m, dtype=np.float64) if vol is not None else None
    w_turn = np.empty(m, dtype=np.float64) if turn is not None else None
    end_idx = np.empty(m, dtype=np.int64)

    for k, (s, e) in enumerate(groups):
        w_dates[k] = d[e - 1]
        w_open[k] = o[s]
        w_high[k] = np.max(h[s:e])
        w_low[k] = np.min(lo[s:e])
        w_close[k] = c[e - 1]
        if w_vol is not None and vol is not None:
            w_vol[k] = np.sum(vol[s:e])
        if w_turn is not None and turn is not None:
            w_turn[k] = np.sum(turn[s:e])
        end_idx[k] = e - 1

    return WeeklyBars(
        week_end_dates=w_dates,
        open=w_open,
        high=w_high,
        low=w_low,
        close=w_close,
        volume=w_vol,
        turnover=w_turn,
        bar_end_indices=end_idx,
    )


def weekly_to_daily_ffill(
    daily_dates,
    week_end_dates,
    weekly_values,
    *,
    strict_prior_week: bool = True,
) -> np.ndarray:
    """將週線序列前向填充回日線序列。

    strict_prior_week=True（預設）：
        日線第 t 天只取 week_end_date < daily_dates[t] 的最新已完成週線；
        第一週所有交易日填 NaN。嚴格保證任何截斷點重算皆零前視偏差。
    strict_prior_week=False：
        日線第 t 天取 week_end_date <= daily_dates[t] 的最新週線（僅供明確確認
        週五收盤後使用的場景）。
    """
    d_dates = _as_dates(daily_dates, "daily_dates")
    w_dates = _as_dates(week_end_dates, "week_end_dates")
    w_vals = np.asarray(weekly_values, dtype=np.float64)
    if w_vals.ndim != 1 or len(w_vals) != len(w_dates):
        raise ValueError(f"weekly_values 長度必須與 week_end_dates 相同（{len(w_dates)}）")

    out = np.full(len(d_dates), np.nan, dtype=np.float64)
    if len(d_dates) == 0 or len(w_dates) == 0:
        return out

    side = "left" if strict_prior_week else "right"
    idx = np.searchsorted(w_dates, d_dates, side=side) - 1
    valid = idx >= 0
    out[valid] = w_vals[idx[valid]]
    return out
