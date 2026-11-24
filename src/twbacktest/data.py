"""讀入股價指數與報酬指數兩份 CSV，檢查後換算報酬指數的開高低。"""
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from datetime import date

import numpy as np


class DataError(ValueError):
    """資料檔不合格；訊息寫出檔名、日期或行號、原因。"""


@dataclass(frozen=True)
class SourceFile:
    name: str
    rows: int
    start: str
    end: str
    sha256: str


@dataclass(frozen=True)
class Market:
    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    price_index: SourceFile
    total_return: SourceFile


_PRICE_COLS = ["date", "open", "high", "low", "close"]
_TR_COLS = ["date", "close"]


def _read(path, cols):
    path = os.fspath(path)
    name = os.path.basename(path)
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise DataError(f"{name}: 不是 UTF-8（{e}）") from None
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header != cols:
        raise DataError(f"{name}: 標頭必須是 {','.join(cols)}，實際是 {header}")
    out = []
    prev = None
    for lineno, row in enumerate(reader, start=2):
        if len(row) != len(cols):
            raise DataError(f"{name}: 第 {lineno} 行欄位數應為 {len(cols)}")
        try:
            d = date.fromisoformat(row[0])
        except ValueError:
            raise DataError(f"{name}: 第 {lineno} 行日期不是 YYYY-MM-DD：{row[0]!r}") from None
        if len(row[0]) != 10:
            raise DataError(f"{name}: 第 {lineno} 行日期不是 YYYY-MM-DD：{row[0]!r}")
        if prev is not None and d <= prev:
            raise DataError(f"{name}: 第 {lineno} 行日期 {row[0]} 沒有嚴格遞增")
        prev = d
        vals = []
        for col, s in zip(cols[1:], row[1:]):
            try:
                v = float(s)
            except ValueError:
                raise DataError(f"{name}: {row[0]} 的 {col} 不是數字：{s!r}") from None
            if not np.isfinite(v) or v <= 0:
                raise DataError(f"{name}: {row[0]} 的 {col} 必須是有限正數：{s!r}")
            vals.append(v)
        out.append((row[0], vals))
    if not out:
        raise DataError(f"{name}: 沒有資料列")
    src = SourceFile(name, len(out), out[0][0], out[-1][0], sha)
    return out, src


def load_market(price_index, total_return) -> Market:
    p_rows, p_src = _read(price_index, _PRICE_COLS)
    t_rows, t_src = _read(total_return, _TR_COLS)
    first = t_rows[0][0]
    p_rows = [r for r in p_rows if r[0] >= first]
    p_by_date = {d: v for d, v in p_rows}
    for d, _ in t_rows:
        if d not in p_by_date:
            raise DataError(f"{p_src.name}: 缺少 {d}（報酬指數 {t_src.name} 有這一天）")
    t_dates = {d for d, _ in t_rows}
    for d, _ in p_rows:
        if d not in t_dates:
            raise DataError(f"{p_src.name}: {d} 在報酬指數 {t_src.name} 沒有")
    n = len(t_rows)
    o = np.empty(n); h = np.empty(n); lo = np.empty(n); c = np.empty(n)
    for i, (d, (tc,)) in enumerate(t_rows):
        po, ph, pl, pc = p_by_date[d]
        if not (pl <= po <= ph and pl <= pc <= ph):
            raise DataError(f"{p_src.name}: {d} 開高低收不合理（需 最低 ≤ 開、收 ≤ 最高）")
        ratio = tc / pc
        o[i] = po * ratio
        h[i] = ph * ratio
        lo[i] = pl * ratio
        c[i] = tc
    dates = np.array([d for d, _ in t_rows], dtype="datetime64[D]")
    return Market(dates, o, h, lo, c, p_src, t_src)
