"""把测试窗口按官方方式拆开再拼回去。

GIFT-Eval 的测试集按「序列 0 的窗口 0、序列 0 的窗口 1、…」排列。
官方 Chronos-2 评测每次只把同一个窗口上的不同序列放进一批，
避免同一条序列的相邻窗口互相泄漏。
"""

from __future__ import annotations


def window_indexes(n_items: int, n_windows: int) -> list[list[int]]:
    if n_windows < 1:
        raise ValueError("窗口数必须为正")
    return [list(range(offset, n_items, n_windows)) for offset in range(n_windows)]


def restore_window_order(groups: list[list]) -> list:
    n_windows = len(groups)
    n_items = sum(len(group) for group in groups)
    ordered: list = [None] * n_items
    for offset, group in enumerate(groups):
        indexes = range(offset, n_items, n_windows)
        if len(group) != len(range(offset, n_items, n_windows)):
            raise RuntimeError("窗口条数和原始测试集对不上")
        for index, item in zip(indexes, group):
            ordered[index] = item
    if any(item is None for item in ordered):
        raise RuntimeError("窗口还原后有空位")
    return ordered
