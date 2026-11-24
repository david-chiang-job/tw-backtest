import unittest

import numpy as np

from twbacktest.config import parse_config
from twbacktest.windows import WINDOWS, build_windows, find_cycles

# 10 個交易日，含週末間隔；索引 0..9
DATES = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
     "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"],
    dtype="datetime64[D]",
)  # fmt: skip
CLOSE = np.array([100.0, 101, 99, 98, 102, 103, 97, 96, 100, 104])
SPEC = {"fixed": {"lengths": [3, 5]}}


@WINDOWS.register("test_fake")
def _fake(dates, close, pairs=((0, 2),)):
    start = np.array([p[0] for p in pairs], dtype=np.int64)
    end = np.array([p[1] for p in pairs], dtype=np.int64)
    return start, end, np.full(len(pairs), "fake"), np.full(len(pairs), -1, np.int64)


def ints(a):
    return [int(x) for x in a]


class FixedWindowsTest(unittest.TestCase):
    def test_all_windows_without_score_end(self):
        w = build_windows(DATES, CLOSE, SPEC)
        # 3 日：起跑 0..7；5 日：起跑 0..5。起跑 6、7 只有 3 日窗口，8、9 沒有
        self.assertEqual(ints(w.starts), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(ints(w.ptr), [0, 2, 4, 6, 8, 10, 12, 13, 14])
        self.assertEqual(ints(w.ends), [2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 9])
        self.assertEqual(list(w.kind), ["3d", "5d"] * 6 + ["3d", "3d"])
        self.assertEqual(ints(w.key), [-1] * 14)

    def test_score_end_cuts_late_windows(self):
        # 2024-01-09 是索引 6：結算日晚於它的不輸出，等於它的保留
        w = build_windows(DATES, CLOSE, SPEC, score_end="2024-01-09")
        self.assertEqual(ints(w.starts), [0, 1, 2, 3, 4])
        self.assertEqual(ints(w.ptr), [0, 2, 4, 6, 7, 8])
        self.assertEqual(ints(w.ends), [2, 4, 3, 5, 4, 6, 5, 6])
        self.assertEqual(list(w.kind), ["3d", "5d", "3d", "5d", "3d", "5d", "3d", "3d"])
        self.assertEqual(ints(w.key), [-1] * 8)

    def test_start_with_all_windows_cut_is_dropped(self):
        full = build_windows(DATES, CLOSE, SPEC)
        self.assertIn(7, ints(full.starts))
        cut = build_windows(DATES, CLOSE, SPEC, score_end="2024-01-09")
        self.assertNotIn(7, ints(cut.starts))
        self.assertEqual(len(cut.ptr), len(cut.starts) + 1)

    def test_too_short_market_gives_no_windows(self):
        w = build_windows(DATES[:2], CLOSE[:2], SPEC)
        self.assertEqual(ints(w.starts), [])
        self.assertEqual(ints(w.ptr), [0])
        self.assertEqual(ints(w.ends), [])

    def test_duplicate_end_keeps_spec_order(self):
        # fixed 的 3 日窗口與假產生器剛好給同一對 (0, 2)：保留 spec 順序；核心支援同終點
        a = build_windows(DATES[:4], CLOSE[:4], {"fixed": {"lengths": [3]}, "test_fake": {}})
        self.assertEqual(ints(a.starts), [0, 1])
        self.assertEqual(ints(a.ptr), [0, 2, 3])
        self.assertEqual(ints(a.ends), [2, 2, 3])
        self.assertEqual(list(a.kind), ["3d", "fake", "3d"])
        b = build_windows(DATES[:4], CLOSE[:4], {"test_fake": {}, "fixed": {"lengths": [3]}})
        self.assertEqual(list(b.kind), ["fake", "3d", "3d"])

    def test_bad_lengths_rejected(self):
        for bad in ([], [3, 3], [0], [-1], [2.5], [True], 3):
            with self.assertRaises(ValueError, msg=repr(bad)):
                build_windows(DATES, CLOSE, {"fixed": {"lengths": bad}})


# 21 個連續日；drawdown 用 0.25
CDATES = np.datetime64("2024-01-01") + np.arange(21)
CCLOSE = np.array([
    100, 90, 95, 100, 70, 70, 80, 100, 110, 82.5,
    90, 110, 120, 60, 90, 120, 130, 100, 130, 120, 80,
], dtype=float)  # fmt: skip
CSPEC = {"cycles": {"counts": [1, 2, 3], "drawdown": 0.25}}


class FindCyclesTest(unittest.TestCase):
    def test_cycles_of_series(self):
        peak, trough, rec = find_cycles(CCLOSE, 0.25)
        # 0 到 3 只跌 10% 不成場，且高點同價取最新（3）；4、5 谷底同價取首次（4）
        # 8 到 9 剛好跌 25% 成場；16 到 17 只跌 23% 不成場；最後 19 到 20 沒收復不成場
        self.assertEqual(ints(peak), [3, 8, 12])
        self.assertEqual(ints(trough), [4, 9, 13])
        self.assertEqual(ints(rec), [7, 11, 15])

    def test_no_cycle_when_market_flat_or_short(self):
        for c in ([], [100.0], [100.0, 100.0, 100.0], [100.0, 50.0]):
            peak, trough, rec = find_cycles(np.array(c), 0.25)
            self.assertEqual((len(peak), len(trough), len(rec)), (0, 0, 0))

    def test_bad_drawdown_rejected(self):
        for bad in (0, 1, -0.1, 1.5, True, "0.2"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                find_cycles(CCLOSE, bad)


class CycleWindowsTest(unittest.TestCase):
    def test_pairing_with_full_market(self):
        w = build_windows(CDATES, CCLOSE, CSPEC)
        # 起跑 0 到 4 配第一場（谷底 4，含谷底當天）；5 到 9 配第二場；10 到 13 配第三場；14 起無循環
        self.assertEqual(ints(w.starts), list(range(14)))
        self.assertEqual(ints(w.ptr), [0, 3, 6, 9, 12, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29])
        self.assertEqual(ints(w.ends), [7, 11, 15] * 5 + [11, 15] * 5 + [15] * 4)
        self.assertEqual(list(w.kind), ["cycle1", "cycle2", "cycle3"] * 5 + ["cycle1", "cycle2"] * 5 + ["cycle1"] * 4)
        self.assertEqual(ints(w.key), [4] * 15 + [9] * 10 + [13] * 4)

    def test_start_on_trough_pairs_that_cycle_and_after_pairs_next(self):
        w = build_windows(CDATES, CCLOSE, {"cycles": {"counts": [1], "drawdown": 0.25}})
        i4, i5 = ints(w.starts).index(4), ints(w.starts).index(5)
        self.assertEqual((int(w.ends[w.ptr[i4]]), int(w.key[w.ptr[i4]])), (7, 4))
        self.assertEqual((int(w.ends[w.ptr[i5]]), int(w.key[w.ptr[i5]])), (11, 9))

    def test_not_enough_cycles_omits_longer_windows(self):
        w = build_windows(CDATES[:12], CCLOSE[:12], CSPEC)  # 只有前兩場循環
        self.assertEqual(ints(w.starts), list(range(10)))
        self.assertEqual(ints(w.ptr), [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15])
        self.assertEqual(ints(w.ends), [7, 11] * 5 + [11] * 5)
        self.assertEqual(list(w.kind), ["cycle1", "cycle2"] * 5 + ["cycle1"] * 5)

    def test_score_end_cuts_windows_by_recovery_day(self):
        # 索引 11 是 2024-01-12；收復日 15 晚於它，配到第三場的窗口不輸出
        cut = build_windows(CDATES, CCLOSE, CSPEC, score_end="2024-01-12")
        self.assertEqual(ints(cut.starts), list(range(10)))
        self.assertEqual(ints(cut.ends), [7, 11] * 5 + [11] * 5)
        self.assertEqual(list(cut.kind), ["cycle1", "cycle2"] * 5 + ["cycle1"] * 5)
        # 對照：沒設 score_end 時看得到被截掉的窗口
        full = build_windows(CDATES, CCLOSE, CSPEC)
        self.assertIn(15, ints(full.ends))
        self.assertNotIn(15, ints(cut.ends))

    def test_bad_params_rejected(self):
        for bad in ({"counts": []}, {"counts": [1, 1]}, {"counts": [0]}, {"drawdown": 0}, {"drawdown": 1}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                build_windows(CDATES, CCLOSE, {"cycles": bad})

    def test_counts_not_limited_to_three(self):
        w = build_windows(CDATES, CCLOSE, {"cycles": {"counts": [2], "drawdown": 0.25}})
        self.assertEqual(list(w.kind[:1]), ["cycle2"])
        self.assertEqual(ints(w.ends[:1]), [11])


TIER = {"name": "t", "target_stock": 0.5, "reset": "none", "when": {"indicator": "drawdown", "at_most": -0.1}}


class ExtensionPoint(unittest.TestCase):
    def test_fake_generator_runs_through_config_and_build(self):
        data = {"tiers": [TIER], "windows": {"test_fake": {"pairs": [[1, 4], [2, 5]]}, "score_end": "2024-01-04"}}
        cfg = parse_config(data)
        self.assertEqual(cfg.windows.spec, {"test_fake": {"pairs": [[1, 4], [2, 5]]}})
        # DATES[4] 是 2024-01-05，晚於 score_end，兩個窗口都被截掉；不設 score_end 才看得到
        cut = build_windows(DATES, CLOSE, cfg.windows.spec, cfg.windows.score_end)
        self.assertEqual(ints(cut.starts), [])
        w = build_windows(DATES, CLOSE, cfg.windows.spec, None)
        self.assertEqual(ints(w.starts), [1, 2])
        self.assertEqual(ints(w.ends), [4, 5])
        self.assertEqual(list(w.kind), ["fake", "fake"])


if __name__ == "__main__":
    unittest.main()
