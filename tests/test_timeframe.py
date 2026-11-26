"""跨週期週線重採樣與嚴格零前視映射測試。"""

from __future__ import annotations

import unittest

import numpy as np

from twbacktest import ta
from twbacktest.timeframe import resample_weekly, weekly_to_daily_ffill


class WeeklyResampleChecks(unittest.TestCase):
    def test_resample_weekly_handles_holidays_and_year_boundary(self):
        # 2024-12-30 (一), 2024-12-31 (二), 2025-01-02 (四, 週三放假), 2025-01-03 (五) -> 同一 ISO 週 (2025, W1)
        # 2025-01-06 (一), 2025-01-07 (二) -> 下一 ISO 週 (2025, W2)
        dates = np.array(
            [
                "2024-12-30",
                "2024-12-31",
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-01-07",
            ],
            dtype="datetime64[D]",
        )
        o = np.array([100.0, 102.0, 101.0, 104.0, 110.0, 112.0])
        h = np.array([105.0, 108.0, 106.0, 109.0, 115.0, 113.0])
        lo = np.array([99.0, 101.0, 98.0, 103.0, 109.0, 107.0])
        c = np.array([103.0, 104.0, 105.0, 107.0, 111.0, 108.0])
        vol = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        turn = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])

        wb = resample_weekly(dates, o, h, lo, c, volume=vol, turnover=turn)
        self.assertEqual([str(d) for d in wb.week_end_dates], ["2025-01-03", "2025-01-07"])
        self.assertEqual(list(wb.open), [100.0, 110.0])
        self.assertEqual(list(wb.high), [109.0, 115.0])
        self.assertEqual(list(wb.low), [98.0, 107.0])
        self.assertEqual(list(wb.close), [107.0, 108.0])
        self.assertEqual(list(wb.volume), [100.0, 110.0])
        self.assertEqual(list(wb.turnover), [10000.0, 11000.0])
        self.assertEqual(list(wb.bar_end_indices), [3, 5])

    def test_weekly_to_daily_ffill_strict_prior_week_vs_same_day(self):
        # 兩週：W1 2024-01-02..2024-01-05；W2 2024-01-08..2024-01-12
        dates = np.array(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-05",
                "2024-01-08",
                "2024-01-10",
                "2024-01-12",
                "2024-01-15",
            ],
            dtype="datetime64[D]",
        )
        w_ends = np.array(["2024-01-05", "2024-01-12", "2024-01-15"], dtype="datetime64[D]")
        w_vals = np.array([10.0, 20.0, 30.0])

        # strict_prior_week=True：W1 全 NaN，W2（含週五 01-12）皆看 W1=10，W3 (01-15) 看 W2=20
        strict = weekly_to_daily_ffill(dates, w_ends, w_vals, strict_prior_week=True)
        self.assertTrue(np.isnan(strict[:3]).all())
        self.assertEqual(list(strict[3:6]), [10.0, 10.0, 10.0])
        self.assertEqual(strict[6], 20.0)

        # strict_prior_week=False（對照）：週五 01-05 當日即取 W1=10，01-12 當日取 W2=20
        same_day = weekly_to_daily_ffill(dates, w_ends, w_vals, strict_prior_week=False)
        self.assertTrue(np.isnan(same_day[:2]).all())
        self.assertEqual(list(same_day[2:5]), [10.0, 10.0, 10.0])
        self.assertEqual(same_day[5], 20.0)
        self.assertEqual(same_day[6], 30.0)

    def test_weekly_pipeline_truncation_has_zero_lookahead(self):
        """整條「日線 -> 週線 -> 週線指標 -> 日線映射」在任意截斷日重算皆零前視偏差。"""
        # 產生 12 週的工作日序列（跳過幾個週五模擬假日）
        all_days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-03-29"))
        weekdays = [d for d in all_days if d.astype(object).weekday() < 5]
        # 移除某個週五與某個週三模擬假日
        holidays = {np.datetime64("2024-01-19"), np.datetime64("2024-02-14")}
        dates = np.array([d for d in weekdays if d not in holidays], dtype="datetime64[D]")

        rng = np.random.default_rng(42)
        n = len(dates)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.015, size=n))
        open_ = close * (1.0 + rng.normal(0.0, 0.005, size=n))
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99

        def _pipeline(k: int):
            wb = resample_weekly(dates[:k], open_[:k], high[:k], low[:k], close[:k])
            w_sma = ta.sma(wb.close, 3)
            w_rsi = ta.rsi(wb.close, 4)
            return (
                weekly_to_daily_ffill(dates[:k], wb.week_end_dates, w_sma),
                weekly_to_daily_ffill(dates[:k], wb.week_end_dates, w_rsi),
            )

        full_sma, full_rsi = _pipeline(n)
        for t in range(n):
            t_sma, t_rsi = _pipeline(t + 1)
            for label, fv, tv in (("w_sma", full_sma[t], t_sma[t]), ("w_rsi", full_rsi[t], t_rsi[t])):
                if np.isnan(fv):
                    self.assertTrue(np.isnan(tv), f"{label} 在 t={t} ({dates[t]}) 應為 NaN，實際 {tv}")
                else:
                    self.assertLessEqual(
                        abs(fv - tv),
                        1e-12,
                        f"{label} 在 t={t} ({dates[t]}) 發生前視偏差：full={fv}, trunc={tv}",
                    )

    def test_unsorted_dates_rejected(self):
        dates = np.array(["2024-01-03", "2024-01-02"], dtype="datetime64[D]")
        x = np.array([100.0, 101.0])
        with self.assertRaises(ValueError):
            resample_weekly(dates, x, x, x, x)


if __name__ == "__main__":
    unittest.main()
