"""技術指標工具庫單元測試：純 Python 獨立手算比對、暖機與邊界、截斷零前視偏差。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from twbacktest import ta


def _make_sample_ohlcv(n: int = 80):
    """確定性合成 OHLCV 序列（含趨勢、震盪與跳空，僅供測試）。"""
    rng = np.random.default_rng(20261126)
    rets = rng.normal(0.001, 0.018, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0.0, 0.006, size=n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.015, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.015, size=n))
    volume = rng.uniform(1e6, 5e6, size=n)
    return open_, high, low, close, volume


def _assert_close_with_nan(tc: unittest.TestCase, got: np.ndarray, want: list[float] | np.ndarray, tol: float = 1e-11):
    want_arr = np.asarray(want, dtype=np.float64)
    tc.assertEqual(got.shape, want_arr.shape)
    nan_got = np.isnan(got)
    nan_want = np.isnan(want_arr)
    tc.assertTrue(np.array_equal(nan_got, nan_want), f"NaN 遮罩不一致：got={nan_got}, want={nan_want}")
    valid = ~nan_want
    if np.any(valid):
        diff = np.max(np.abs(got[valid] - want_arr[valid]))
        tc.assertLessEqual(diff, tol, f"最大誤差 {diff} 超過容許值 {tol}")


# ---------------------------------------------------------------------------
# 獨立純 Python 手算參考實作（不與 ta.py 共用向量化邏輯）
# ---------------------------------------------------------------------------


def _ref_sma(vals: list[float], w: int) -> list[float]:
    out = []
    for t in range(len(vals)):
        if t < w - 1:
            out.append(math.nan)
        else:
            win = vals[t - w + 1 : t + 1]
            if any(math.isnan(v) for v in win):
                out.append(math.nan)
            else:
                out.append(sum(win) / w)
    return out


def _ref_ema(vals: list[float], w: int) -> list[float]:
    n = len(vals)
    out = [math.nan] * n
    start = None
    run = 0
    for i, v in enumerate(vals):
        if math.isfinite(v):
            run += 1
            if run == w:
                start = i - w + 1
                break
        else:
            run = 0
    if start is None:
        return out
    alpha = 2.0 / (w + 1.0)
    seed_i = start + w - 1
    prev = sum(vals[start : seed_i + 1]) / w
    out[seed_i] = prev
    for i in range(seed_i + 1, n):
        if not math.isfinite(vals[i]) or not math.isfinite(prev):
            prev = math.nan
        else:
            prev = alpha * vals[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _ref_rsi(close: list[float], w: int) -> list[float]:
    n = len(close)
    out = [math.nan] * n
    if n <= w:
        return out
    gains = [max(close[i] - close[i - 1], 0.0) for i in range(1, n)]
    losses = [max(close[i - 1] - close[i], 0.0) for i in range(1, n)]
    ag = sum(gains[:w]) / w
    al = sum(losses[:w]) / w

    def _calc(g, l_):
        if g == 0.0 and l_ == 0.0:
            return 50.0
        if l_ == 0.0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + g / l_)

    out[w] = _calc(ag, al)
    for t in range(w + 1, n):
        ag = (ag * (w - 1) + gains[t - 1]) / w
        al = (al * (w - 1) + losses[t - 1]) / w
        out[t] = _calc(ag, al)
    return out


def _ref_true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    if not close:
        return []
    out = [high[0] - low[0]]
    for t in range(1, len(close)):
        out.append(max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1])))
    return out


def _ref_adx(high: list[float], low: list[float], close: list[float], w: int):
    n = len(close)
    pdi = [math.nan] * n
    mdi = [math.nan] * n
    adx_ = [math.nan] * n
    if n <= w:
        return pdi, mdi, adx_
    tr = [0.0] * n
    pdm = [0.0] * n
    mdm = [0.0] * n
    for t in range(1, n):
        tr[t] = max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1]))
        up = high[t] - high[t - 1]
        dn = low[t - 1] - low[t]
        if up > dn and up > 0.0:
            pdm[t] = up
        if dn > up and dn > 0.0:
            mdm[t] = dn
    trs = sum(tr[1 : w + 1])
    pds = sum(pdm[1 : w + 1])
    mds = sum(mdm[1 : w + 1])
    dx = [math.nan] * n
    for t in range(w, n):
        if t > w:
            trs = trs - trs / w + tr[t]
            pds = pds - pds / w + pdm[t]
            mds = mds - mds / w + mdm[t]
        p = 100.0 * pds / trs if trs > 0 else 0.0
        m = 100.0 * mds / trs if trs > 0 else 0.0
        pdi[t] = p
        mdi[t] = m
        dx[t] = 100.0 * abs(p - m) / (p + m) if (p + m) > 0 else 0.0
    if n > 2 * w - 1:
        val = sum(dx[w : 2 * w]) / w
        adx_[2 * w - 1] = val
        for t in range(2 * w, n):
            val = (val * (w - 1) + dx[t]) / w
            adx_[t] = val
    return pdi, mdi, adx_


class ReferenceValueChecks(unittest.TestCase):
    def setUp(self):
        self.o, self.h, self.lo, self.c, self.v = _make_sample_ohlcv(50)

    def test_trend_indicators_match_pure_python_reference(self):
        c_list = self.c.tolist()
        ref_s = _ref_sma(c_list, 5)
        ref_e = _ref_ema(c_list, 5)
        _assert_close_with_nan(self, ta.sma(self.c, 5), ref_s)
        _assert_close_with_nan(self, ta.ema(self.c, 5), ref_e)

        ref_bias = [math.nan if math.isnan(m) else (cv - m) / m for cv, m in zip(c_list, ref_s)]
        _assert_close_with_nan(self, ta.ma_bias(self.c, 5, kind="sma"), ref_bias)

        ref_slope = [math.nan] * len(c_list)
        for t in range(3, len(c_list)):
            if not math.isnan(ref_e[t - 3]) and not math.isnan(ref_e[t]):
                ref_slope[t] = (ref_e[t] - ref_e[t - 3]) / ref_e[t - 3]
        _assert_close_with_nan(self, ta.ma_slope(self.c, 5, lookback=3, kind="ema"), ref_slope)

        res_macd = ta.macd(self.c, fast=4, slow=8, signal=3)
        ef = _ref_ema(c_list, 4)
        es = _ref_ema(c_list, 8)
        dif = [math.nan if (math.isnan(a) or math.isnan(b)) else a - b for a, b in zip(ef, es)]
        dea = _ref_ema(dif, 3)
        hist = [math.nan if (math.isnan(a) or math.isnan(b)) else a - b for a, b in zip(dif, dea)]
        _assert_close_with_nan(self, res_macd.dif, dif)
        _assert_close_with_nan(self, res_macd.dea, dea)
        _assert_close_with_nan(self, res_macd.histogram, hist)

        res_adx = ta.adx(self.h, self.lo, self.c, window=6)
        rp, rm, ra = _ref_adx(self.h.tolist(), self.lo.tolist(), c_list, 6)
        _assert_close_with_nan(self, res_adx.plus_di, rp)
        _assert_close_with_nan(self, res_adx.minus_di, rm)
        _assert_close_with_nan(self, res_adx.adx, ra)

    def test_momentum_indicators_match_pure_python_reference(self):
        c_list = self.c.tolist()
        h_list = self.h.tolist()
        l_list = self.lo.tolist()

        _assert_close_with_nan(self, ta.rsi(self.c, 6), _ref_rsi(c_list, 6))

        # Stochastic (SMA & Recursive)
        w = 5
        rsv_ref = []
        for t in range(len(c_list)):
            if t < w - 1:
                rsv_ref.append(math.nan)
            else:
                hi = max(h_list[t - w + 1 : t + 1])
                lo = min(l_list[t - w + 1 : t + 1])
                rsv_ref.append(50.0 if hi == lo else 100.0 * (c_list[t] - lo) / (hi - lo))
        k_sma = _ref_sma(rsv_ref, 3)
        d_sma = _ref_sma(k_sma, 3)
        st_sma = ta.stochastic(self.h, self.lo, self.c, window=w, k_smooth=3, d_smooth=3, method="sma")
        _assert_close_with_nan(self, st_sma.rsv, rsv_ref)
        _assert_close_with_nan(self, st_sma.k, k_sma)
        _assert_close_with_nan(self, st_sma.d, d_sma)

        k_rec = [math.nan] * len(c_list)
        d_rec = [math.nan] * len(c_list)
        kv = dv = 50.0
        for t in range(w - 1, len(c_list)):
            kv = (2.0 / 3.0) * kv + (1.0 / 3.0) * rsv_ref[t]
            dv = (2.0 / 3.0) * dv + (1.0 / 3.0) * kv
            k_rec[t] = kv
            d_rec[t] = dv
        st_rec = ta.stochastic(self.h, self.lo, self.c, window=w, k_smooth=3, d_smooth=3, method="recursive")
        _assert_close_with_nan(self, st_rec.k, k_rec)
        _assert_close_with_nan(self, st_rec.d, d_rec)

        roc_ref = [math.nan if t < 4 else (c_list[t] - c_list[t - 4]) / c_list[t - 4] for t in range(len(c_list))]
        _assert_close_with_nan(self, ta.roc(self.c, 4), roc_ref)

    def test_channel_and_volatility_indicators_match_pure_python_reference(self):
        c_list = self.c.tolist()
        h_list = self.h.tolist()
        l_list = self.lo.tolist()
        w = 7

        bb = ta.bollinger_bands(self.c, window=w, num_std=2.0)
        mid, up, lo, bw, pb = [], [], [], [], []
        for t in range(len(c_list)):
            if t < w - 1:
                mid.append(math.nan); up.append(math.nan); lo.append(math.nan)
                bw.append(math.nan); pb.append(math.nan)
            else:
                win = c_list[t - w + 1 : t + 1]
                m = sum(win) / w
                std = math.sqrt(sum((x - m) ** 2 for x in win) / w)
                u = m + 2.0 * std
                l_ = m - 2.0 * std
                mid.append(m); up.append(u); lo.append(l_)
                bw.append((u - l_) / m)
                pb.append((c_list[t] - l_) / (u - l_))
        _assert_close_with_nan(self, bb.middle, mid)
        _assert_close_with_nan(self, bb.upper, up)
        _assert_close_with_nan(self, bb.lower, lo)
        _assert_close_with_nan(self, bb.bandwidth, bw)
        _assert_close_with_nan(self, bb.percent_b, pb)

        dc = ta.donchian_channel(self.h, self.lo, window=w)
        dc_u = [math.nan if t < w - 1 else max(h_list[t - w + 1 : t + 1]) for t in range(len(c_list))]
        dc_l = [math.nan if t < w - 1 else min(l_list[t - w + 1 : t + 1]) for t in range(len(c_list))]
        dc_m = [math.nan if math.isnan(a) else 0.5 * (a + b) for a, b in zip(dc_u, dc_l)]
        _assert_close_with_nan(self, dc.upper, dc_u)
        _assert_close_with_nan(self, dc.lower, dc_l)
        _assert_close_with_nan(self, dc.middle, dc_m)

        tr_ref = _ref_true_range(h_list, l_list, c_list)
        _assert_close_with_nan(self, ta.true_range(self.h, self.lo, self.c), tr_ref)

        atr_ref = [math.nan] * len(c_list)
        av = sum(tr_ref[:w]) / w
        atr_ref[w - 1] = av
        for t in range(w, len(c_list)):
            av = (av * (w - 1) + tr_ref[t]) / w
            atr_ref[t] = av
        _assert_close_with_nan(self, ta.atr(self.h, self.lo, self.c, window=w), atr_ref)

        ce = ta.chandelier_exit(self.h, self.lo, self.c, window=w, mult=2.5)
        ce_long = [math.nan if math.isnan(a) else a - 2.5 * b for a, b in zip(dc_u, atr_ref)]
        ce_short = [math.nan if math.isnan(a) else a + 2.5 * b for a, b in zip(dc_l, atr_ref)]
        _assert_close_with_nan(self, ce.long_exit, ce_long)
        _assert_close_with_nan(self, ce.short_exit, ce_short)

        tr_pct = [tr_ref[t] / c_list[t] for t in range(len(c_list))]
        tr_pct_ma = _ref_sma(tr_pct, w)
        tr_ratio_ref = [math.nan if math.isnan(m) else p / m for p, m in zip(tr_pct, tr_pct_ma)]
        _assert_close_with_nan(self, ta.tr_ratio(self.h, self.lo, self.c, window=w), tr_ratio_ref)

    def test_volume_and_structure_indicators_match_pure_python_reference(self):
        o_list = self.o.tolist()
        h_list = self.h.tolist()
        l_list = self.lo.tolist()
        c_list = self.c.tolist()
        v_list = self.v.tolist()
        w = 6

        v_ma = _ref_sma(v_list, w)
        vr_ref = [math.nan if math.isnan(m) else vv / m for vv, m in zip(v_list, v_ma)]
        _assert_close_with_nan(self, ta.volume_ratio(self.v, window=w), vr_ref)

        obv_ref = [0.0]
        for t in range(1, len(c_list)):
            if c_list[t] > c_list[t - 1]:
                obv_ref.append(obv_ref[-1] + v_list[t])
            elif c_list[t] < c_list[t - 1]:
                obv_ref.append(obv_ref[-1] - v_list[t])
            else:
                obv_ref.append(obv_ref[-1])
        _assert_close_with_nan(self, ta.obv(self.c, self.v), obv_ref)

        tp = [(h_list[t] + l_list[t] + c_list[t]) / 3.0 for t in range(len(c_list))]
        rmf = [tp[t] * v_list[t] for t in range(len(c_list))]
        mfi_ref = [math.nan] * len(c_list)
        for t in range(w, len(c_list)):
            pos = neg = 0.0
            for k in range(t - w + 1, t + 1):
                if tp[k] > tp[k - 1]:
                    pos += rmf[k]
                elif tp[k] < tp[k - 1]:
                    neg += rmf[k]
            if pos == 0.0 and neg == 0.0:
                mfi_ref[t] = 50.0
            elif neg == 0.0:
                mfi_ref[t] = 100.0
            else:
                mfi_ref[t] = 100.0 - 100.0 / (1.0 + pos / neg)
        _assert_close_with_nan(self, ta.mfi(self.h, self.lo, self.c, self.v, window=w), mfi_ref)

        cr = ta.candle_ratios(self.o, self.h, self.lo, self.c)
        b_ref, sb_ref, us_ref, ls_ref = [], [], [], []
        for o_, h_, l_, c_ in zip(o_list, h_list, l_list, c_list):
            sp = h_ - l_
            b_ref.append(abs(c_ - o_) / sp)
            sb_ref.append((c_ - o_) / sp)
            us_ref.append((h_ - max(o_, c_)) / sp)
            ls_ref.append((min(o_, c_) - l_) / sp)
        _assert_close_with_nan(self, cr.body_ratio, b_ref)
        _assert_close_with_nan(self, cr.signed_body_ratio, sb_ref)
        _assert_close_with_nan(self, cr.upper_shadow_ratio, us_ref)
        _assert_close_with_nan(self, cr.lower_shadow_ratio, ls_ref)

        dd_ref = [
            math.nan if t < w - 1 else (c_list[t] - max(c_list[t - w + 1 : t + 1])) / max(c_list[t - w + 1 : t + 1])
            for t in range(len(c_list))
        ]
        ru_ref = [
            math.nan if t < w - 1 else (c_list[t] - min(c_list[t - w + 1 : t + 1])) / min(c_list[t - w + 1 : t + 1])
            for t in range(len(c_list))
        ]
        _assert_close_with_nan(self, ta.rolling_drawdown(self.c, w), dd_ref)
        _assert_close_with_nan(self, ta.rolling_runup(self.c, w), ru_ref)


class ZeroLookaheadChecks(unittest.TestCase):
    """截斷零前視偏差不變性測試：截斷至 [:t+1] 重算，第 t 天值與全序列逐點一致。"""

    def test_all_indicators_are_strictly_causal(self):
        o, h, lo, c, v = _make_sample_ohlcv(75)

        specs = [
            ("sma", lambda s: [ta.sma(c[:s], 10)]),
            ("ema", lambda s: [ta.ema(c[:s], 10)]),
            ("ma_bias", lambda s: [ta.ma_bias(c[:s], 10, kind="ema")]),
            ("ma_slope", lambda s: [ta.ma_slope(c[:s], 10, lookback=3, kind="sma")]),
            ("macd", lambda s: list(ta.macd(c[:s], 6, 13, 5).__dict__.values())),
            ("adx", lambda s: list(ta.adx(h[:s], lo[:s], c[:s], 7).__dict__.values())),
            ("rsi", lambda s: [ta.rsi(c[:s], 7)]),
            ("stochastic_sma", lambda s: list(ta.stochastic(h[:s], lo[:s], c[:s], 9, 3, 3, method="sma").__dict__.values())),
            ("stochastic_rec", lambda s: list(ta.stochastic(h[:s], lo[:s], c[:s], 9, 3, 3, method="recursive").__dict__.values())),
            ("roc", lambda s: [ta.roc(c[:s], 8)]),
            ("bollinger", lambda s: list(ta.bollinger_bands(c[:s], 10, 2.0).__dict__.values())),
            ("donchian", lambda s: list(ta.donchian_channel(h[:s], lo[:s], 10).__dict__.values())),
            ("true_range", lambda s: [ta.true_range(h[:s], lo[:s], c[:s])]),
            ("atr", lambda s: [ta.atr(h[:s], lo[:s], c[:s], 7)]),
            ("chandelier", lambda s: list(ta.chandelier_exit(h[:s], lo[:s], c[:s], 10, 2.5).__dict__.values())),
            ("tr_ratio", lambda s: [ta.tr_ratio(h[:s], lo[:s], c[:s], 10)]),
            ("volume_ratio", lambda s: [ta.volume_ratio(v[:s], 10)]),
            ("obv", lambda s: [ta.obv(c[:s], v[:s])]),
            ("mfi", lambda s: [ta.mfi(h[:s], lo[:s], c[:s], v[:s], 7)]),
            ("candle_ratios", lambda s: list(ta.candle_ratios(o[:s], h[:s], lo[:s], c[:s]).__dict__.values())),
            ("rolling_drawdown", lambda s: [ta.rolling_drawdown(c[:s], 10)]),
            ("rolling_runup", lambda s: [ta.rolling_runup(c[:s], 10)]),
        ]

        cut_points = [1, 5, 10, 15, 25, 45, 74]
        for name, fn in specs:
            full_list = fn(len(c))
            for t in cut_points:
                trunc_list = fn(t + 1)
                for idx, (full_arr, trunc_arr) in enumerate(zip(full_list, trunc_list)):
                    fv = full_arr[t]
                    tv = trunc_arr[t]
                    if np.isnan(fv):
                        self.assertTrue(np.isnan(tv), f"{name}[{idx}] 在 t={t} 應為 NaN，實際 {tv}")
                    else:
                        self.assertLessEqual(
                            abs(fv - tv),
                            1e-12,
                            f"{name}[{idx}] 在 t={t} 發生前視偏差：full={fv}, trunc={tv}",
                        )


class EdgeAndValidationChecks(unittest.TestCase):
    def test_flat_series_and_zero_span_handled_without_div_zero(self):
        flat = np.full(20, 100.0)
        vol = np.full(20, 1000.0)
        r = ta.rsi(flat, 5)
        self.assertTrue(np.isnan(r[:5]).all())
        self.assertTrue(np.all(r[5:] == 50.0))

        m = ta.mfi(flat, flat, flat, vol, 5)
        self.assertTrue(np.isnan(m[:5]).all())
        self.assertTrue(np.all(m[5:] == 50.0))

        st = ta.stochastic(flat, flat, flat, 5, 3, 3)
        self.assertTrue(np.all(st.rsv[4:] == 50.0))

        bb = ta.bollinger_bands(flat, 5, 2.0)
        self.assertTrue(np.isnan(bb.percent_b).all())
        self.assertTrue(np.all(bb.bandwidth[4:] == 0.0))

        cr = ta.candle_ratios(flat, flat, flat, flat)
        self.assertTrue(np.all(cr.body_ratio == 0.0))
        self.assertTrue(np.all(cr.upper_shadow_ratio == 0.0))
        self.assertTrue(np.all(cr.lower_shadow_ratio == 0.0))

    def test_monotonic_up_gives_rsi_and_mfi_100(self):
        up = np.arange(10.0, 30.0)
        vol = np.full(len(up), 500.0)
        self.assertTrue(np.all(ta.rsi(up, 5)[5:] == 100.0))
        self.assertTrue(np.all(ta.mfi(up + 1, up - 1, up, vol, 5)[5:] == 100.0))

    def test_invalid_arguments_raise_value_error(self):
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ta.sma(x, 0)
        with self.assertRaises(ValueError):
            ta.macd(x, fast=12, slow=12)
        with self.assertRaises(ValueError):
            ta.ma_bias(x, 2, kind="wma")
        with self.assertRaises(ValueError):
            ta.stochastic(x, x[:2], x)
        with self.assertRaises(ValueError):
            ta.bollinger_bands(x, 2, num_std=-1.0)


if __name__ == "__main__":
    unittest.main()
