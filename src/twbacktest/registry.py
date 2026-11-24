"""通用元件登記表：元件模組登記自己，登記表本身不 import 任何具體元件。"""

from __future__ import annotations


class UnknownComponentError(KeyError):
    def __str__(self):
        return self.args[0]


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, name):
        """decorator：@REGISTRY.register("名稱")。名稱重複會報錯。"""

        def deco(obj):
            if name in self._items:
                raise ValueError(f"{self.kind} 名稱重複登記：{name!r}")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name):
        try:
            return self._items[name]
        except KeyError:
            raise UnknownComponentError(
                f"未知的{self.kind}名稱 {name!r}；可用的名稱：{', '.join(sorted(self._items)) or '（無）'}"
            ) from None

    def names(self):
        return sorted(self._items)
