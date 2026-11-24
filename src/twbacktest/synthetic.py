"""固定種子的合成行情，供範例與測試使用；不含任何真實資料。"""
from __future__ import annotations

import argparse
import os
import random
from datetime import date, timedelta

DEFAULT_DAYS = 5878
DEFAULT_SEED = 4
PRICE_FILE = "taiex.csv"
TOTAL_RETURN_FILE = "taiex-total-return.csv"


def _weekdays(start: str, days: int) -> list[date]:
    d = date.fromisoformat(start)
    out = []
    while len(out) < days:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def generate(days: int = DEFAULT_DAYS, seed: int = DEFAULT_SEED, start: str = "2003-01-02"):
    """回傳 (股價指數列, 報酬指數列)。

    股價指數列為 (日期, 開, 高, 低, 收)，報酬指數列為 (日期, 收)，數值已四捨五入到小數 2 位。
    每年有數天報酬指數相對股價指數的比例往上跳，模擬除息。
    """
    rng = random.Random(seed)
    dates = _weekdays(start, days)
    price_rows = []
    tr_rows = []
    close = 5000.0
    ratio = 1.0
    regime_left = 0
    drift = 0.0
    prev_year = None
    jump_days: set[int] = set()
    for i, d in enumerate(dates):
        if d.year != prev_year:
            prev_year = d.year
            same_year = [j for j, x in enumerate(dates) if x.year == d.year]
            jump_days = set(rng.sample(same_year, min(3, len(same_year))))
        if regime_left <= 0:
            bear = rng.random() < 0.3
            regime_left = rng.randint(60, 200) if bear else rng.randint(150, 600)
            drift = rng.uniform(-0.0025, -0.0012) if bear else rng.uniform(0.0001, 0.0006)
        regime_left -= 1
        gap = rng.gauss(0.0, 0.003)
        open_ = close * (1 + gap)
        new_close = open_ * (1 + rng.gauss(drift, 0.011))
        high = max(open_, new_close) * (1 + abs(rng.gauss(0.0, 0.004)))
        low = min(open_, new_close) * (1 - abs(rng.gauss(0.0, 0.004)))
        o, h, l, c = (round(x, 2) for x in (open_, high, low, new_close))
        h = max(h, o, c)
        l = min(l, o, c)
        close = new_close
        ratio *= 1 + 0.00005
        if i in jump_days:
            ratio *= 1 + rng.uniform(0.008, 0.03)
        price_rows.append((d.isoformat(), o, h, l, c))
        tr_rows.append((d.isoformat(), round(c * ratio, 2)))
    return price_rows, tr_rows


def write_csv(directory, days: int = DEFAULT_DAYS, seed: int = DEFAULT_SEED, start: str = "2003-01-02"):
    """寫出 taiex.csv 與 taiex-total-return.csv，回傳兩個路徑。"""
    price_rows, tr_rows = generate(days, seed, start)
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, PRICE_FILE)
    t = os.path.join(directory, TOTAL_RETURN_FILE)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("date,open,high,low,close\n")
        for d, o, h, l, c in price_rows:
            f.write(f"{d},{o:.2f},{h:.2f},{l:.2f},{c:.2f}\n")
    with open(t, "w", encoding="utf-8", newline="") as f:
        f.write("date,close\n")
        for d, c in tr_rows:
            f.write(f"{d},{c:.2f}\n")
    return p, t


def main(argv=None):
    ap = argparse.ArgumentParser(description="產生合成行情 CSV")
    ap.add_argument("--out", required=True, help="輸出資料夾")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args(argv)
    for p in write_csv(a.out, a.days, a.seed):
        print(p)


if __name__ == "__main__":
    main()
