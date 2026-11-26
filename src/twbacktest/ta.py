"""常用技術指標工具庫（純 NumPy、嚴格因果零前視偏差）。

所有指標皆滿足因果契約：第 t 天的值僅使用索引 0..t（截至第 t 天收盤）的資料，
暖機期算不出的日子一律填 np.nan。將輸入截斷至 [:t+1] 重算時，第 t 天結果不變。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class MACDResult:
    dif: np.ndarray
    dea: np.ndarray
    histogram: np.ndarray


@dataclass(frozen=True)
class ADXResult:
    plus_di: np.ndarray
    minus_di: np.ndarray
    adx: np.ndarray


@dataclass(frozen=True)
class StochasticResult:
    rsv: np.ndarray
    k: np.ndarray
    d: np.ndarray


@dataclass(frozen=True)
class BollingerResult:
    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    bandwidth: np.ndarray
    percent_b: np.ndarray


@dataclass(frozen=True)
class DonchianResult:
    upper: np.ndarray
    lower: np.ndarray
    middle: np.ndarray


@dataclass(frozen=True)
class ChandelierResult:
    long_exit: np.ndarray
    short_exit: np.ndarray


@dataclass(frozen=True)
class CandleRatios:
    body_ratio: np.ndarray
    upper_shadow_ratio: np.ndarray
    lower_shadow_ratio: np.ndarray
    signed_body_ratio: np.ndarray


def _as_1d(arr, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} 必須是一維陣列（目前 ndim={a.ndim}）")
    return a


def _check_same_length(**arrays: np.ndarray) -> int:
    lengths = {k: len(v) for k, v in arrays.items()}
    unique = set(lengths.values())
    if len(unique) > 1:
        detail = ", ".join(f"{k}={n}" for k, n in lengths.items())
        raise ValueError(f"輸入陣列長度不一致：{detail}")
    return next(iter(unique), 0)


def _check_window(window: int, name: str = "window") -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or int(window) < 1:
        raise ValueError(f"{name} 必須為 >= 1 的整數（目前 {window!r}）")
    return int(window)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[valid] = num[valid] / den[valid]
    return out


# ---------------------------------------------------------------------------
# 1. 均線與趨勢
# ---------------------------------------------------------------------------


def sma(values, window: int) -> np.ndarray:
    """簡單移動平均；t < window - 1 或視窗內含 NaN 時為 NaN。"""
    x = _as_1d(values, "values")
    w = _check_window(window)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    out[w - 1 :] = sliding_window_view(x, w).mean(axis=1)
    return out


def ema(values, window: int) -> np.ndarray:
    """指數移動平均（alpha = 2 / (window + 1)，以首個連續 window 筆有限值的 SMA 為種子）。"""
    x = _as_1d(values, "values")
    w = _check_window(window)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out

    # 找到第一個連續 w 筆皆為有限數值的起點
    finite = np.isfinite(x)
    start = -1
    run = 0
    for i in range(n):
        if finite[i]:
            run += 1
            if run == w:
                start = i - w + 1
                break
        else:
            run = 0
    if start < 0:
        return out

    seed_idx = start + w - 1
    alpha = 2.0 / (w + 1.0)
    one_minus = 1.0 - alpha
    val = float(np.mean(x[start : seed_idx + 1]))
    out[seed_idx] = val
    for i in range(seed_idx + 1, n):
        xi = x[i]
        if not np.isfinite(xi):
            val = np.nan
            out[i] = np.nan
        elif np.isfinite(val):
            val = alpha * xi + one_minus * val
            out[i] = val
        else:
            out[i] = np.nan
    return out


def ma_bias(close, window: int, *, kind: str = "sma") -> np.ndarray:
    """均線乖離率：(close - ma) / ma。kind 支援 'sma' 或 'ema'。"""
    c = _as_1d(close, "close")
    if kind == "sma":
        ma = sma(c, window)
    elif kind == "ema":
        ma = ema(c, window)
    else:
        raise ValueError(f"kind 必須是 'sma' 或 'ema'（目前 {kind!r}）")
    return _safe_div(c - ma, ma)


def ma_slope(close, window: int, *, lookback: int = 1, kind: str = "sma") -> np.ndarray:
    """均線斜率（近 lookback 日變化率）：(ma[t] - ma[t - lookback]) / ma[t - lookback]。"""
    c = _as_1d(close, "close")
    lb = _check_window(lookback, "lookback")
    if kind == "sma":
        ma = sma(c, window)
    elif kind == "ema":
        ma = ema(c, window)
    else:
        raise ValueError(f"kind 必須是 'sma' 或 'ema'（目前 {kind!r}）")
    out = np.full(len(c), np.nan, dtype=np.float64)
    if len(c) <= lb:
        return out
    out[lb:] = _safe_div(ma[lb:] - ma[:-lb], ma[:-lb])
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> MACDResult:
    """MACD（DIF、DEA、Histogram = DIF - DEA）。"""
    c = _as_1d(close, "close")
    f = _check_window(fast, "fast")
    s = _check_window(slow, "slow")
    sig = _check_window(signal, "signal")
    if f >= s:
        raise ValueError(f"fast 必須小於 slow（目前 fast={f}, slow={s}）")
    dif = ema(c, f) - ema(c, s)
    dea = ema(dif, sig)
    hist = dif - dea
    return MACDResult(dif=dif, dea=dea, histogram=hist)


def adx(high, low, close, window: int = 14) -> ADXResult:
    """Wilder 平滑 DMI / ADX（+DI、-DI、ADX）。

    +DI / -DI 自索引 window 起有效，ADX 自索引 2 * window - 1 起有效。
    """
    h = _as_1d(high, "high")
    lo = _as_1d(low, "low")
    c = _as_1d(close, "close")
    n = _check_same_length(high=h, low=lo, close=c)
    w = _check_window(window)

    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    adx_out = np.full(n, np.nan, dtype=np.float64)
    if n <= w:
        return ADXResult(plus_di=plus_di, minus_di=minus_di, adx=adx_out)

    tr = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        tr[t] = max(h[t] - lo[t], abs(h[t] - c[t - 1]), abs(lo[t] - c[t - 1]))
        up = h[t] - h[t - 1]
        down = lo[t - 1] - lo[t]
        if up > down and up > 0.0:
            pdm[t] = up
        if down > up and down > 0.0:
            mdm[t] = down

    tr_s = float(np.sum(tr[1 : w + 1]))
    pdm_s = float(np.sum(pdm[1 : w + 1]))
    mdm_s = float(np.sum(mdm[1 : w + 1]))

    dx = np.full(n, np.nan, dtype=np.float64)
    for t in range(w, n):
        if t > w:
            tr_s = tr_s - tr_s / w + tr[t]
            pdm_s = pdm_s - pdm_s / w + pdm[t]
            mdm_s = mdm_s - mdm_s / w + mdm[t]
        pdi = 100.0 * pdm_s / tr_s if tr_s > 0.0 else 0.0
        mdi = 100.0 * mdm_s / tr_s if tr_s > 0.0 else 0.0
        plus_di[t] = pdi
        minus_di[t] = mdi
        denom = pdi + mdi
        dx[t] = 100.0 * abs(pdi - mdi) / denom if denom > 0.0 else 0.0

    adx_start = 2 * w - 1
    if n > adx_start:
        val = float(np.mean(dx[w : adx_start + 1]))
        adx_out[adx_start] = val
        for t in range(adx_start + 1, n):
            val = (val * (w - 1) + dx[t]) / w
            adx_out[t] = val

    return ADXResult(plus_di=plus_di, minus_di=minus_di, adx=adx_out)


# ---------------------------------------------------------------------------
# 2. 動能震盪
# ---------------------------------------------------------------------------


def rsi(close, window: int = 14) -> np.ndarray:
    """Wilder 平滑 RSI；自索引 window 起有效。"""
    c = _as_1d(close, "close")
    w = _check_window(window)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= w:
        return out

    delta = np.diff(c)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    avg_g = float(np.mean(gain[:w]))
    avg_l = float(np.mean(loss[:w]))

    def _rsi_val(g: float, l_: float) -> float:
        if g == 0.0 and l_ == 0.0:
            return 50.0
        if l_ == 0.0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + g / l_)

    out[w] = _rsi_val(avg_g, avg_l)
    for t in range(w + 1, n):
        avg_g = (avg_g * (w - 1) + gain[t - 1]) / w
        avg_l = (avg_l * (w - 1) + loss[t - 1]) / w
        out[t] = _rsi_val(avg_g, avg_l)
    return out


def stochastic(
    high,
    low,
    close,
    window: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
    *,
    method: str = "sma",
) -> StochasticResult:
    """KD / Stochastic 指標。

    method='sma'：標準慢速隨機指標（K = SMA(RSV, k_smooth)、D = SMA(K, d_smooth)）。
    method='recursive'：台股傳統遞迴平滑（初值 50，alpha = 1/k_smooth 與 1/d_smooth）。
    """
    h = _as_1d(high, "high")
    lo = _as_1d(low, "low")
    c = _as_1d(close, "close")
    n = _check_same_length(high=h, low=lo, close=c)
    w = _check_window(window)
    ks = _check_window(k_smooth, "k_smooth")
    ds = _check_window(d_smooth, "d_smooth")
    if method not in ("sma", "recursive"):
        raise ValueError(f"method 必須是 'sma' 或 'recursive'（目前 {method!r}）")

    rsv = np.full(n, np.nan, dtype=np.float64)
    if n >= w:
        hi_w = sliding_window_view(h, w).max(axis=1)
        lo_w = sliding_window_view(lo, w).min(axis=1)
        span = hi_w - lo_w
        c_w = c[w - 1 :]
        rsv_slice = np.full_like(c_w, 50.0)
        np.divide(100.0 * (c_w - lo_w), span, out=rsv_slice, where=span > 0.0)
        rsv[w - 1 :] = rsv_slice

    if method == "sma":
        k = sma(rsv, ks)
        d = sma(k, ds)
        return StochasticResult(rsv=rsv, k=k, d=d)

    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)
    if n >= w:
        ak = 1.0 / ks
        ad = 1.0 / ds
        kv = 50.0
        dv = 50.0
        for t in range(w - 1, n):
            kv = (1.0 - ak) * kv + ak * rsv[t]
            dv = (1.0 - ad) * dv + ad * kv
            k[t] = kv
            d[t] = dv
    return StochasticResult(rsv=rsv, k=k, d=d)


def roc(close, window: int = 10) -> np.ndarray:
    """變動率（Rate of Change）：(close[t] - close[t - window]) / close[t - window]。"""
    c = _as_1d(close, "close")
    w = _check_window(window)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= w:
        return out
    out[w:] = _safe_div(c[w:] - c[:-w], c[:-w])
    return out


# ---------------------------------------------------------------------------
# 3. 通道與波動
# ---------------------------------------------------------------------------


def bollinger_bands(close, window: int = 20, num_std: float = 2.0) -> BollingerResult:
    """布林通道（母體標準差 ddof=0；回傳中軌、上軌、下軌、帶寬與 %B）。"""
    c = _as_1d(close, "close")
    w = _check_window(window)
    if num_std < 0.0 or not np.isfinite(num_std):
        raise ValueError(f"num_std 必須為有限非負數（目前 {num_std!r}）")
    n = len(c)
    middle = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n >= w:
        win = sliding_window_view(c, w)
        m = win.mean(axis=1)
        s = win.std(axis=1, ddof=0)
        middle[w - 1 :] = m
        upper[w - 1 :] = m + num_std * s
        lower[w - 1 :] = m - num_std * s
    width = upper - lower
    bandwidth = _safe_div(width, middle)
    percent_b = _safe_div(c - lower, width)
    return BollingerResult(
        middle=middle,
        upper=upper,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b,
    )


def donchian_channel(high, low, window: int = 20) -> DonchianResult:
    """Donchian 通道（近 window 日最高、最低與中線）。"""
    h = _as_1d(high, "high")
    lo = _as_1d(low, "low")
    n = _check_same_length(high=h, low=lo)
    w = _check_window(window)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n >= w:
        upper[w - 1 :] = sliding_window_view(h, w).max(axis=1)
        lower[w - 1 :] = sliding_window_view(lo, w).min(axis=1)
    middle = 0.5 * (upper + lower)
    return DonchianResult(upper=upper, lower=lower, middle=middle)


def true_range(high, low, close) -> np.ndarray:
    """真實波幅 True Range；第 0 天為 high[0] - low[0]。"""
    h = _as_1d(high, "high")
    lo = _as_1d(low, "low")
    c = _as_1d(close, "close")
    n = _check_same_length(high=h, low=lo, close=c)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[0] = h[0] - lo[0]
    if n > 1:
        hl = h[1:] - lo[1:]
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(lo[1:] - c[:-1])
        out[1:] = np.maximum(hl, np.maximum(hc, lc))
    return out


def atr(high, low, close, window: int = 14) -> np.ndarray:
    """Wilder 平滑平均真實波幅（ATR）；自索引 window - 1 起有效。"""
    tr = true_range(high, low, close)
    w = _check_window(window)
    n = len(tr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    val = float(np.mean(tr[:w]))
    out[w - 1] = val
    for t in range(w, n):
        val = (val * (w - 1) + tr[t]) / w
        out[t] = val
    return out


def chandelier_exit(high, low, close, window: int = 22, mult: float = 3.0) -> ChandelierResult:
    """Chandelier Exit（ATR 吊燈停損／回落線）。"""
    if mult < 0.0 or not np.isfinite(mult):
        raise ValueError(f"mult 必須為有限非負數（目前 {mult!r}）")
    dc = donchian_channel(high, low, window)
    a = atr(high, low, close, window)
    return ChandelierResult(
        long_exit=dc.upper - mult * a,
        short_exit=dc.lower + mult * a,
    )


def tr_ratio(high, low, close, window: int = 20) -> np.ndarray:
    """振幅倍數：當日 (TR / close) 相對近 window 日平均 (TR / close) 的倍數。"""
    c = _as_1d(close, "close")
    tr = true_range(high, low, c)
    tr_pct = _safe_div(tr, c)
    avg = sma(tr_pct, window)
    return _safe_div(tr_pct, avg)


# ---------------------------------------------------------------------------
# 4. 量價指標
# ---------------------------------------------------------------------------


def volume_ratio(volume, window: int = 20) -> np.ndarray:
    """量比：volume / SMA(volume, window)（適用於成交股數或成交金額）。"""
    v = _as_1d(volume, "volume")
    avg = sma(v, window)
    return _safe_div(v, avg)


def obv(close, volume) -> np.ndarray:
    """能量潮（On-Balance Volume）；第 0 天為 0.0。"""
    c = _as_1d(close, "close")
    v = _as_1d(volume, "volume")
    n = _check_same_length(close=c, volume=v)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    if n > 1:
        direction = np.sign(np.diff(c))
        out[1:] = np.cumsum(direction * v[1:])
    return out


def mfi(high, low, close, volume, window: int = 14) -> np.ndarray:
    """資金流量指標（Money Flow Index）；自索引 window 起有效。"""
    h = _as_1d(high, "high")
    lo = _as_1d(low, "low")
    c = _as_1d(close, "close")
    v = _as_1d(volume, "volume")
    n = _check_same_length(high=h, low=lo, close=c, volume=v)
    w = _check_window(window)

    out = np.full(n, np.nan, dtype=np.float64)
    if n <= w:
        return out

    tp = (h + lo + c) / 3.0
    rmf = tp * v
    tp_diff = np.diff(tp)
    pos = np.where(tp_diff > 0.0, rmf[1:], 0.0)
    neg = np.where(tp_diff < 0.0, rmf[1:], 0.0)

    pos_sum = sliding_window_view(pos, w).sum(axis=1)
    neg_sum = sliding_window_view(neg, w).sum(axis=1)

    res = np.empty_like(pos_sum)
    both_zero = (pos_sum == 0.0) & (neg_sum == 0.0)
    neg_zero = (neg_sum == 0.0) & ~both_zero
    normal = ~(both_zero | neg_zero)
    res[both_zero] = 50.0
    res[neg_zero] = 100.0
    res[normal] = 100.0 - 100.0 / (1.0 + pos_sum[normal] / neg_sum[normal])
    out[w:] = res
    return out


# ---------------------------------------------------------------------------
# 5. K 線與結構輔助
# ---------------------------------------------------------------------------


def candle_ratios(open_, high, low, close) -> CandleRatios:
    """K 線實體與上下影線比例（除以 high - low；無振幅時皆為 0.0）。"""
    o = _as_1d(open_, "open_")
    h = _as_1d(high, "high")
    lo = _as_1d(low, "low")
    c = _as_1d(close, "close")
    n = _check_same_length(open_=o, high=h, low=lo, close=c)

    span = h - lo
    valid = span > 0.0
    body = np.zeros(n, dtype=np.float64)
    signed_body = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)

    if np.any(valid):
        s = span[valid]
        body[valid] = np.abs(c[valid] - o[valid]) / s
        signed_body[valid] = (c[valid] - o[valid]) / s
        upper[valid] = (h[valid] - np.maximum(o[valid], c[valid])) / s
        lower[valid] = (np.minimum(o[valid], c[valid]) - lo[valid]) / s

    return CandleRatios(
        body_ratio=body,
        upper_shadow_ratio=upper,
        lower_shadow_ratio=lower,
        signed_body_ratio=signed_body,
    )


def rolling_drawdown(close, window: int) -> np.ndarray:
    """波段高點回落：(close - rolling_max(close, window)) / rolling_max(close, window)（<= 0）。"""
    c = _as_1d(close, "close")
    w = _check_window(window)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    peak = sliding_window_view(c, w).max(axis=1)
    out[w - 1 :] = _safe_div(c[w - 1 :] - peak, peak)
    return out


def rolling_runup(close, window: int) -> np.ndarray:
    """波段低點漲幅：(close - rolling_min(close, window)) / rolling_min(close, window)（>= 0）。"""
    c = _as_1d(close, "close")
    w = _check_window(window)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    trough = sliding_window_view(c, w).min(axis=1)
    out[w - 1 :] = _safe_div(c[w - 1 :] - trough, trough)
    return out
