"""讀設定與檢查。這段只解析 [[tiers]]，其他區段原樣留在 extra。"""

from __future__ import annotations

import inspect
import tomllib
from dataclasses import dataclass, field

from .conditions import COMBINATORS
from .indicators import INDICATORS

SIDES = {"at_most": 1, "at_least": -1}
LEAF_STRUCT_KEYS = {"indicator", "points", *SIDES}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Tier:
    name: str
    target_stock: float
    reset: str
    when: dict
    side: int  # 1 買進、-1 賣出


@dataclass(frozen=True)
class Config:
    tiers: list
    extra: dict = field(default_factory=dict)


def load_config(path):
    """讀 TOML 成字典（可能含範圍寫法）；展開與檢查走 grid.expand 或 parse_config。"""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def parse_config(data, registry=INDICATORS):
    tiers = data.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ConfigError("缺少 [[tiers]]")
    out = []
    for i, t in enumerate(tiers):
        name = t.get("name", f"#{i}") if isinstance(t, dict) else f"#{i}"
        out.append(_parse_tier(t, name, registry))
    return Config(out, {k: v for k, v in data.items() if k != "tiers"})


def _parse_tier(t, name, registry):
    def err(where, why):
        raise ConfigError(f"檔位 {name!r} 的 {where}：{why}")

    if not isinstance(t, dict):
        err("內容", "必須是表")
    for k in ("name", "target_stock", "reset", "when"):
        if k not in t:
            err(k, "缺少此欄位")
    if not _is_num(t["target_stock"]):
        err("target_stock", "必須是數字")
    if not isinstance(t["reset"], str):
        err("reset", "必須是字串")
    sides = set()
    _check_node(t["when"], "when", registry, err, sides, in_weighted=False)
    if len(sides) != 1:
        err("when", "同一檔位買賣混用（at_most 與 at_least 不能並存）")
    return Tier(t["name"], t["target_stock"], t["reset"], t["when"], sides.pop())


def _check_node(node, where, registry, err, sides, in_weighted):
    if not isinstance(node, dict):
        err(where, "必須是表")
    if in_weighted:
        p = node.get("points")
        if not _is_num(p) or p <= 0:
            err(where, f"points 必須是大於 0 的數字（目前 {p!r}）")
    elif "points" in node:
        err(where, "points 只能用在 weighted 的子節點")
    if "indicator" in node:
        _check_leaf(node, where, registry, err, sides)
        return
    names = [n for n in COMBINATORS.names() if n in node]
    if len(names) != 1:
        err(where, f"必須是單一條件（indicator）或剛好一種組合（{'、'.join(COMBINATORS.names())}），目前鍵：{sorted(node)}")
    name = names[0]
    allowed = {name, "points"} | ({"score_at_least"} if name == "weighted" else set())
    for k in node:
        if k not in allowed:
            err(where, f"組合 {name} 不認得的鍵 {k!r}")
    children = node[name]
    if not isinstance(children, list) or not children:
        err(where, f"{name} 必須是非空的清單")
    n = None
    if name == "weighted":
        n = node.get("score_at_least")
        if not _is_num(n) or n <= 0:
            err(where, f"score_at_least 必須是大於 0 的數字（目前 {n!r}）")
    for j, c in enumerate(children):
        _check_node(c, f"{where}.{name}[{j}]", registry, err, sides, in_weighted=(name == "weighted"))
    if name == "weighted":
        total = sum(c["points"] for c in children)
        if total < n:
            err(where, f"全部分數加起來 {total} 到不了門檻 {n}")


def _check_leaf(node, where, registry, err, sides):
    try:
        ind = registry.get(node["indicator"])
    except KeyError as e:
        err(where, str(e))
    present = [k for k in SIDES if k in node]
    if len(present) != 1:
        err(where, "at_most 與 at_least 必須恰好寫一個（目前：" + ("、".join(present) or "都沒寫") + "）")
    if not _is_num(node[present[0]]):
        err(where, f"{present[0]} 必須是數字")
    sides.add(SIDES[present[0]])
    params = list(inspect.signature(ind.base).parameters.values())[1:]  # 第 1 個是 close
    need = {p.name for p in params if p.default is p.empty}
    known = {p.name for p in params}
    given = {k for k in node if k not in LEAF_STRUCT_KEYS}
    if need - given:
        err(where, f"指標 {node['indicator']} 缺參數：{sorted(need - given)}")
    if given - known:
        err(where, f"指標 {node['indicator']} 不認得的鍵：{sorted(given - known)}")


def leaf_threshold(leaf):
    return leaf["at_most"] if "at_most" in leaf else leaf["at_least"]


def leaf_params(leaf):
    return {k: v for k, v in leaf.items() if k not in LEAF_STRUCT_KEYS}
