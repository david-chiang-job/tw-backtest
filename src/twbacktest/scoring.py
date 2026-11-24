"""逐窗指標、評分登記表、彙總與排名。彙總只吃逐窗結果、窗口種類與分組鍵。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .registry import Registry

SCORES = Registry("評分")

YEAR_DAYS = 365.2425


@dataclass(frozen=True)
class WindowMetrics:  # 全部 [N, E] float64，E 跟 windows.ends 一一對應
    total_return: np.ndarray  # 期末淨值 / 本金 − 1
    annualized: np.ndarray  # (期末淨值 / 本金) ** (365.2425 / 曆日數) − 1
    excess_total: np.ndarray  # (策略總報酬 − 同窗全倉總報酬) × 100，百分點
    excess_annual: np.ndarray  # (策略年化 − 同窗全倉年化) × 100，百分點
    max_drawdown: np.ndarray  # 直接取 Accounts.max_drawdown
    principal_loss: np.ndarray  # max(0, (本金 − 最低淨值) / 本金)
    wait_days: np.ndarray  # 起跑日到首次買入的曆日數；沒買入是 NaN


def _returns(accounts, cash, cal_days):
    ratio = np.asarray(accounts.nav, dtype=np.float64) / cash
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.where(cal_days > 0, YEAR_DAYS / np.where(cal_days > 0, cal_days, 1), np.nan)
        annual = ratio**exp - 1
    return ratio - 1, annual


def window_metrics(accounts, baseline, windows, dates, cash) -> WindowMetrics:
    dates = np.asarray(dates, dtype="datetime64[D]")
    start = np.repeat(windows.starts, np.diff(windows.ptr))
    cal_days = (dates[windows.ends] - dates[start]).astype(np.int64).astype(np.float64)
    total, annual = _returns(accounts, cash, cal_days)
    b_total, b_annual = _returns(baseline, cash, cal_days)
    first = np.asarray(accounts.first_buy)
    bought = first >= 0
    wait = np.where(bought, (dates[np.where(bought, first, 0)] - dates[start]).astype(np.int64), 0).astype(np.float64)
    wait[~bought] = np.nan
    return WindowMetrics(
        total_return=total,
        annualized=annual,
        excess_total=(total - b_total) * 100,
        excess_annual=(annual - b_annual) * 100,
        max_drawdown=np.asarray(accounts.max_drawdown, dtype=np.float64),
        principal_loss=np.maximum(0.0, (cash - np.asarray(accounts.lowest, dtype=np.float64)) / cash),
        wait_days=wait,
    )


def _nan_safe(fn, values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(values, axis=1)


@SCORES.register("median")
def median(values, keys, weights=None):
    """忽略 NaN 的中位數。"""
    return _nan_safe(np.nanmedian, values)


@SCORES.register("p10")
def p10(values, keys, weights=None):
    """忽略 NaN 的第 10 百分位，線性插值。"""
    return _nan_safe(lambda v, axis: np.nanpercentile(v, 10, axis=axis), values)


@SCORES.register("nan_share")
def nan_share(values, keys, weights=None):
    """NaN 的比例；套在 wait_days 上就是零買入比例。"""
    return np.isnan(values).mean(axis=1)


@SCORES.register("group_mean")
def group_mean(values, keys, weights=None):
    """先組內平均，再依權重平均各組；預設等權。weights 是 {組鍵: 權重}。"""
    groups = np.unique(keys)
    means = np.stack([values[:, keys == g].mean(axis=1) for g in groups], axis=1)
    if weights is None:
        w = np.ones(len(groups))
    else:
        missing = [int(g) for g in groups if int(g) not in weights]
        if missing:
            raise ValueError(f"weights 缺少組鍵 {missing}")
        w = np.array([weights[int(g)] for g in groups], dtype=np.float64)
        if not w.sum() > 0:
            raise ValueError(f"權重總和必須大於 0（目前 {w.sum()!r}）")
    return (means * w).sum(axis=1) / w.sum()


DEFAULT_SCORES = (
    ("720d", "excess_total", "median"),  # 主排名
    ("720d", "total_return", "p10"),
    ("1200d", "excess_total", "median"),
    ("1200d", "total_return", "p10"),
    ("240d", "excess_annual", "median"),
    ("240d", "wait_days", "nan_share"),
    ("240d", "wait_days", "median"),
    ("cycle1", "total_return", "group_mean"), ("cycle1", "excess_total", "group_mean"),
    ("cycle2", "total_return", "group_mean"), ("cycle2", "excess_total", "group_mean"),
    ("cycle3", "total_return", "group_mean"), ("cycle3", "excess_total", "group_mean"),
)  # fmt: skip
RANK_BY = "720d.excess_total.median"


def summarize(metrics, windows, scores=DEFAULT_SCORES, weights=None) -> dict[str, np.ndarray]:
    """對每列 (種類, 指標, 評分名稱) 取出該種類的欄，回傳 {"種類.指標.評分名稱": [N]}。沒有該種類窗口時全是 NaN。"""
    n = metrics.total_return.shape[0]
    out = {}
    for kind, metric, name in scores:
        fn = SCORES.get(name)
        mask = np.asarray(windows.kind) == kind
        if not mask.any():
            out[f"{kind}.{metric}.{name}"] = np.full(n, np.nan)
            continue
        values = np.asarray(getattr(metrics, metric))[:, mask]
        out[f"{kind}.{metric}.{name}"] = np.asarray(fn(values, np.asarray(windows.key)[mask], weights), dtype=np.float64)
    return out


def rank(summary, by=RANK_BY) -> np.ndarray:
    """策略索引，分數由高到低；NaN 最後；同分照索引。"""
    if by not in summary:
        raise KeyError(f"summary 沒有 {by!r}；可用：{sorted(summary)}")
    s = np.asarray(summary[by], dtype=np.float64)
    return np.argsort(np.where(np.isnan(s), np.inf, -s), kind="stable")
