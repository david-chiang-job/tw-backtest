import unittest

import numpy as np

from twbacktest.windows import build_windows

# 10 個交易日，含週末間隔；索引 0..9
DATES = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
     "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"],
    dtype="datetime64[D]",
)  # fmt: skip
CLOSE = np.array([100.0, 101, 99, 98, 102, 103, 97, 96, 100, 104])
SPEC = {"fixed": {"lengths": [3, 5]}}


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
        # 3 日與 3 日重複：同一 (start, end) 保留 spec 順序，核心支援同終點
        w = build_windows(DATES[:4], CLOSE[:4], {"fixed": {"lengths": [3, 3]}})
        self.assertEqual(ints(w.starts), [0, 1])
        self.assertEqual(ints(w.ptr), [0, 2, 4])
        self.assertEqual(ints(w.ends), [2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
