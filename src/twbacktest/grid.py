"""參數範圍展開與指標快取。"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass

from .conditions import evaluate
from .config import Config, ConfigError, leaf_params, leaf_threshold, parse_config
from .indicators import INDICATORS


def is_range(x):
    return isinstance(x, dict) and set(x) == {"from", "to", "step"}


def range_values(r, where=""):
    a, b, s = r["from"], r["to"], r["step"]
    for v in (a, b, s):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{where}範圍的 from、to、step 必須是數字")
    if s == 0 or (b - a) * s < 0:
        raise ConfigError(f"{where}範圍 step={s} 與 from={a}、to={b} 的方向不一致或為 0")
    n = math.floor((b - a) / s + 1e-9) + 1
    if all(isinstance(v, int) for v in (a, b, s)):
        return [a + i * s for i in range(n)]
    return [round(a + i * s, 10) for i in range(n)]


def _find_ranges(node, path, found):
    if is_range(node):
        found.append((path, node))
    elif isinstance(node, dict):
        for k, v in node.items():
            _find_ranges(v, path + (k,), found)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _find_ranges(v, path + (i,), found)


def _set(root, path, value):
    for k in path[:-1]:
        root = root[k]
    root[path[-1]] = value


@dataclass(frozen=True)
class Strategy:
    config: Config
    picks: dict  # 範圍位置（路徑 tuple）→ 這組取的值


def expand(data, registry=INDICATORS):
    """把設定字典裡所有範圍做笛卡兒積，每組各做一次檢查。"""
    found = []
    _find_ranges(data, (), found)
    axes = [range_values(r, f"{'.'.join(map(str, p))} 的") for p, r in found]
    out = []
    for k, combo in enumerate(itertools.product(*axes), start=1):
        d = copy.deepcopy(data)
        picks = {}
        for (p, _), v in zip(found, combo):
            _set(d, p, v)
            picks[p] = v
        try:
            cfg = parse_config(d, registry)
        except ConfigError as e:
            raise ConfigError(f"第 {k} 組（{picks}）：{e}") from None
        out.append(Strategy(cfg, picks))
    return out


class IndicatorCache:
    """同一指標、同一組非門檻參數，整段收盤只算一次基準。"""

    def __init__(self, close, registry=INDICATORS):
        self.close = close
        self.registry = registry
        self._bases = {}

    def trigger(self, leaf):
        ind = self.registry.get(leaf["indicator"])
        params = leaf_params(leaf)
        key = (leaf["indicator"], tuple(sorted(params.items())))
        if key not in self._bases:
            self._bases[key] = ind.base(self.close, **params)
        return ind.apply_threshold(self._bases[key], leaf_threshold(leaf))


def tier_trigger(tier, cache):
    """單一檔位每天的合成觸發價。"""
    return evaluate(tier.when, cache.trigger, tier.side)
