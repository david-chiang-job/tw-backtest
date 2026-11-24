"""開場全倉：每窗以同樣本金於起跑日開盤買到含費最大單位，之後不再動，現金照常計息。不另寫帳務，直接交給帳戶核心。"""

from __future__ import annotations

import numpy as np

from .config import Account, Costs
from .kernel import run_accounts, run_accounts_with_fills
from .signals import Signals


def full_position_signals(market) -> Signals:
    """N = 1、K = 1：每天都觸發、價格取開盤、永不重新待命，所以每窗只在起跑日買一次。"""
    d = len(market.open)
    return Signals(
        hit=np.ones((1, d, 1), dtype=bool),
        price=np.asarray(market.open, dtype=np.float64).reshape(1, d, 1).copy(),
        rearm=np.zeros((1, d, 1), dtype=bool),
        order=np.zeros((1, d, 1), dtype=np.int8),
        target=np.ones((1, 1)),
        side=np.ones((1, 1), dtype=np.int8),
        both_sides=[np.empty(0, dtype=np.int64)],
    )


def run_baseline(market, windows, account=Account(), costs=Costs(), threads=None, unit=1.0):
    """回傳 Accounts，形狀 [1, E]，跟 windows.ends 一一對應。"""
    return run_accounts(
        full_position_signals(market), market, windows.starts, windows.ends, windows.ptr,
        account=account, costs=costs, threads=threads, unit=unit,
    )  # fmt: skip


def run_baseline_fills(market, windows, account=Account(), costs=Costs(), threads=None, unit=1.0):
    """同 run_baseline，另回傳逐筆成交：(Accounts, Fills)，Fills 只有一個策略。"""
    return run_accounts_with_fills(
        full_position_signals(market), market, windows.starts, windows.ends, windows.ptr,
        account=account, costs=costs, threads=threads, unit=unit,
    )  # fmt: skip
