"""步道坡线规划台 —— 测点可锁定标高分析。

沿用 planner 的全部规则（3～16 个测点、标高 0～200 整数厘米、首尾固定、
相邻高差 ≤ 8 厘米、任意连续 5 段至少 1 段水平），逐点回答：

    若只把第 i 个测点的方案标高锁定为 h，其余测点自由选择，
    是否仍存在一条完整可行的坡线？

不限总调整量，也不要求与现有最优方案一致，因此这是纯可行性判定。
每个测点返回精确的可锁定值集合，并压缩为升序、互不相邻的整数闭区间
（如 [[92, 108], [110, 112]]；空档不填平，绝不只用最小/最大值概括）。

算法
----
正向 / 反向两次布尔可达性 DP，状态与 planner 的成本 DP 相同：
(pos, mask, h)，mask 打包到达 pos 时最近 4 段的水平状态（bit 0 为最近
一段），不存在的更早段以“水平(1)”填充，使不足 5 段时不误触发约束。

- fwd[pos][mask][h]：能否从首点（标高固定为 ground[0]）走到 pos 点取 h；
- bwd[pos][mask][h]：从 pos 点取 h 出发，能否走到尾点（标高固定为
  ground[-1]）。

跨过 pos 的连续 5 段窗口由两侧共享的 mask 校验，因此第 i 点可锁定 h
当且仅当存在 mask 使 fwd[i][mask][h] 与 bwd[i][mask][h] 同时成立。
复杂度 O(n · 2^4 · 201 · (2·8+1))，16 个测点约两百万次基本操作。
"""

from __future__ import annotations

from typing import List

from .planner import (
    H_MAX,
    H_MIN,
    MAX_GRADE,
    RECENT,
    WINDOW,
    _MASK_RECENT,
    _infeasible_reason,
    _validate,
    _window_mask,
)


def _forward_reachable(ground: List[int]) -> list:
    """fwd[pos][mask][h]：前缀 0..pos 可行且 plan[pos] == h。"""
    n = len(ground)
    size = H_MAX + 1
    fwd = [[[False] * size for _ in range(1 << RECENT)] for _ in range(n)]
    fwd[0][_MASK_RECENT][ground[0]] = True      # 起点之前的虚拟段视为水平
    for pos in range(n - 1):
        check_window = pos >= WINDOW - 1        # 放下第 pos 段后凑满 5 段窗口
        nxt = fwd[pos + 1]
        for mask in range(1 << RECENT):
            row = fwd[pos][mask]
            nmask_flat = 1 | ((mask << 1) & _MASK_RECENT)    # 新段水平
            nmask_slope = (mask << 1) & _MASK_RECENT         # 新段非水平
            ok_flat = not check_window or _window_mask(mask, 1) != 0
            ok_slope = not check_window or _window_mask(mask, 0) != 0
            if not (ok_flat or ok_slope):
                continue
            row_flat = nxt[nmask_flat]
            row_slope = nxt[nmask_slope]
            for h in range(H_MIN, H_MAX + 1):
                if not row[h]:
                    continue
                if ok_flat:
                    row_flat[h] = True
                if ok_slope:
                    lo = max(H_MIN, h - MAX_GRADE)
                    hi = min(H_MAX, h + MAX_GRADE)
                    for nh in range(lo, h):
                        row_slope[nh] = True
                    for nh in range(h + 1, hi + 1):
                        row_slope[nh] = True
    return fwd


def _backward_reachable(ground: List[int]) -> list:
    """bwd[pos][mask][h]：plan[pos] == h 时后缀 pos..n-1 可补全。"""
    n = len(ground)
    size = H_MAX + 1
    bwd = [[[False] * size for _ in range(1 << RECENT)] for _ in range(n)]
    glast = ground[n - 1]
    for mask in range(1 << RECENT):
        bwd[n - 1][mask][glast] = True
    for pos in range(n - 2, -1, -1):
        check_window = pos >= WINDOW - 1        # 第 pos 段是某窗口的最后一段
        nxt = bwd[pos + 1]
        for mask in range(1 << RECENT):
            nmask_flat = 1 | ((mask << 1) & _MASK_RECENT)
            nmask_slope = (mask << 1) & _MASK_RECENT
            ok_flat = not check_window or _window_mask(mask, 1) != 0
            ok_slope = not check_window or _window_mask(mask, 0) != 0
            row_flat = nxt[nmask_flat]
            row_slope = nxt[nmask_slope]
            cell = bwd[pos][mask]
            for h in range(H_MIN, H_MAX + 1):
                ok = False
                if ok_flat and row_flat[h]:
                    ok = True
                elif ok_slope:
                    lo = max(H_MIN, h - MAX_GRADE)
                    hi = min(H_MAX, h + MAX_GRADE)
                    for nh in range(lo, h):
                        if row_slope[nh]:
                            ok = True
                            break
                    if not ok:
                        for nh in range(h + 1, hi + 1):
                            if row_slope[nh]:
                                ok = True
                                break
                cell[h] = ok
    return bwd


def _compress_intervals(values: List[int]) -> List[List[int]]:
    """把升序整数序列压缩为升序且互不相邻的闭区间 [[lo, hi], ...]。

    只合并真正连续的整数；相邻区间之间至少隔一个缺失值，
    不会用首尾最小/最大值把空档填平。
    """
    intervals: List[List[int]] = []
    for v in values:
        if intervals and v == intervals[-1][1] + 1:
            intervals[-1][1] = v
        else:
            intervals.append([v, v])
    return intervals


def lockable_analysis(ground: List[int]) -> dict:
    """逐测点计算可锁定标高区间。

    可行时返回::

        {
          "feasible": True,
          "ground": [...],
          "points": [{"index": 1, "intervals": [[lo, hi], ...]}, ...],
        }

    ``intervals`` 为升序、互不相邻的整数闭区间；锁定该测点方案标高为
    区间内任意整数时，其余测点必能补全为一条完整可行坡线，区间外则不能。
    全线无解时返回 {"feasible": False, "reason": "..."}；非法输入抛
    :class:`app.planner.PlanError`（复用 planner 的校验）。
    """
    _validate(ground)
    n = len(ground)
    fwd = _forward_reachable(ground)
    bwd = _backward_reachable(ground)

    points = []
    for pos in range(n):
        f_layer = fwd[pos]
        b_layer = bwd[pos]
        values = []
        for h in range(H_MIN, H_MAX + 1):
            for mask in range(1 << RECENT):
                if f_layer[mask][h] and b_layer[mask][h]:
                    values.append(h)
                    break
        points.append({"index": pos + 1, "intervals": _compress_intervals(values)})

    if not points[0]["intervals"]:
        # 首点只能取 ground[0]；它都无可锁定值即全线无解。
        return {"feasible": False, "reason": _infeasible_reason(ground)}

    return {"feasible": True, "ground": list(ground), "points": points}
