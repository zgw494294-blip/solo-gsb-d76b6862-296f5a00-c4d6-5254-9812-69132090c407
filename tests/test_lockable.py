"""可锁定标高分析测试：穷举对拍 + 结构不变量 + 区间压缩。"""

from __future__ import annotations

import random

import pytest

from app.planner import (
    H_MAX,
    H_MIN,
    MAX_GRADE,
    N_MAX,
    WINDOW,
    PlanError,
    lockable_heights,
    plan_trail,
)


# ---------------------------------------------------------------------------
# 穷举：按逐点可行带剪枝，收集“出现在某条完整可行坡线中”的全部标高
# ---------------------------------------------------------------------------

def _relaxed_band(ground):
    """只受相邻高差与首尾固定约束时，各点可行标高区间（双向传播）。"""
    n = len(ground)
    lo = [H_MIN] * n
    hi = [H_MAX] * n
    lo[0] = hi[0] = ground[0]
    lo[n - 1] = hi[n - 1] = ground[n - 1]
    for i in range(1, n - 1):
        lo[i] = max(H_MIN, lo[i - 1] - MAX_GRADE)
        hi[i] = min(H_MAX, hi[i - 1] + MAX_GRADE)
    for i in range(n - 2, 0, -1):
        lo[i] = max(lo[i], lo[i + 1] - MAX_GRADE)
        hi[i] = min(hi[i], hi[i + 1] + MAX_GRADE)
    return lo, hi


def brute_force_lockable(ground):
    """返回 (feasible, [set(每点可锁标高), ...])。"""
    n = len(ground)
    lo, hi = _relaxed_band(ground)
    if any(lo[i] > hi[i] for i in range(n)):
        return False, [set() for _ in range(n)]

    found = [set() for _ in range(n)]
    plan = [0] * n
    plan[0], plan[n - 1] = ground[0], ground[n - 1]

    def prefix_windows_ok(upto):
        for k in range(upto - WINDOW + 2):
            if not any(plan[k + j] == plan[k + j + 1] for j in range(WINDOW)):
                return False
        return True

    def dfs(idx):
        if idx == n:
            for i, h in enumerate(plan):
                found[i].add(h)
            return
        prev = plan[idx - 1]
        cands = (
            range(max(lo[idx], prev - MAX_GRADE), min(hi[idx], prev + MAX_GRADE) + 1)
            if idx < n - 1
            else [plan[n - 1]]
        )
        for h in cands:
            # 末点候选为固定值，仍需校验与前一点的高差
            if abs(h - prev) > MAX_GRADE:
                continue
            plan[idx] = h
            if prefix_windows_ok(idx - 1):
                dfs(idx + 1)

    dfs(1)
    return bool(found[0]), found


def assert_matches_bruteforce(ground):
    r = lockable_heights(ground)
    feasible, bf = brute_force_lockable(ground)
    assert r["feasible"] == feasible
    if not feasible:
        assert "points" not in r
        assert "无解" in r["reason"]
        return
    assert len(r["points"]) == len(ground)
    for i, p in enumerate(r["points"]):
        assert p["index"] == i + 1
        assert p["fixed"] == (i == 0 or i == len(ground) - 1)
        assert set(p["values"]) == bf[i]
        # 区间压缩：展平后恰好等于 values，且区间升序、互不相邻
        flat = []
        for lo, hi in p["intervals"]:
            assert lo <= hi
            flat.extend(range(lo, hi + 1))
        assert flat == p["values"]
        for (_, a), (b, _) in zip(p["intervals"], p["intervals"][1:]):
            assert b >= a + 2
        if p["fixed"]:
            assert p["values"] == [ground[i]]
            assert p["intervals"] == [[ground[i], ground[i]]]


# ---------- 定点用例：含空档，不能用最小/最大值填平 ----------

def test_sample_intervals_have_gaps():
    r = lockable_heights([100, 107, 114, 119, 124, 130])
    assert r["feasible"]
    p2 = r["points"][1]
    # 101～105 均不可锁定：锁定后前 5 段只能一直爬坡，凑不出水平段
    assert p2["values"] == [100, 106, 107, 108]
    assert p2["intervals"] == [[100, 100], [106, 108]]
    assert r["points"][0]["intervals"] == [[100, 100]]
    assert r["points"][-1]["intervals"] == [[130, 130]]


def test_flat_ground():
    r = lockable_heights([100] * 6)
    assert r["feasible"]
    for p in r["points"]:
        assert p["values"]
        assert 100 in p["values"]
        for h in p["values"]:
            assert 0 <= h <= H_MAX


def test_grade_only_n3_full_range():
    # n=3 不触发水平窗口：中间点只要与两端高差均 ≤ 8
    r = lockable_heights([100, 100, 108])
    vals = r["points"][1]["values"]
    assert vals == list(range(100, 109))
    assert r["points"][1]["intervals"] == [[100, 108]]


def test_boundary_zero():
    r = lockable_heights([0, 8, 0])
    assert r["feasible"]
    assert r["points"][1]["values"] == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_infeasible_strict_rise():
    r = lockable_heights([0, 8, 16, 24, 32, 40])
    assert r["feasible"] is False
    assert "无解" in r["reason"]


def test_infeasible_endpoint_gap():
    r = lockable_heights([0, 100, 100, 100, 100, 41])
    assert r["feasible"] is False
    assert "无解" in r["reason"]


# ---------- 非法输入：复用 plan_trail 的同一套校验 ----------

@pytest.mark.parametrize(
    "ground",
    [
        [],
        [1, 2],
        [1] * 17,
        [0, -1, 0],
        [0, 201, 0],
        [0, 1.5, 0],
        [0, "8", 0],
        [0, True, 0],
        "not-a-list",
    ],
)
def test_invalid_inputs(ground):
    with pytest.raises(PlanError):
        lockable_heights(ground)


# ---------- 穷举对拍 ----------

def test_bruteforce_band_cases():
    """地面限制在宽 8 的带状内（n≤6，全 0～200 值域），穷举对拍。"""
    rng = random.Random(42)
    cases = [
        [0, 0, 0, 8, 8, 0],
        [0, 2, 5, 8, 6, 4],
        [8, 8, 0, 0, 8, 8],
        [100, 107, 114, 119, 124, 130],
        [200, 192, 184, 192, 200],
        [0, 8, 0],
        [0, 0, 8],
    ]
    for ground in cases:
        assert_matches_bruteforce(ground)

    for _ in range(40):
        n = rng.randint(3, 6)
        base = rng.randint(0, H_MAX - MAX_GRADE)
        ground = [base + rng.randint(0, MAX_GRADE) for _ in range(n)]
        assert_matches_bruteforce(ground)


def test_bruteforce_small_domain_longer():
    """小值域下 n=7、8 穷举对拍。"""
    rng = random.Random(99)
    for _ in range(20):
        n = rng.randint(7, 8)
        ground = [rng.randint(0, 10) for _ in range(n)]
        assert_matches_bruteforce(ground)


# ---------- 大规模结构不变量 ----------

def test_optimal_plan_always_lockable():
    """最优方案里每点标高必属于该点可锁定集合（任意随机算例）。"""
    rng = random.Random(2026)
    for _ in range(150):
        n = rng.randint(3, N_MAX)
        ground = [rng.randint(H_MIN, H_MAX) for _ in range(n)]
        pr = plan_trail(ground)
        lr = lockable_heights(ground)
        assert pr["feasible"] == lr["feasible"]
        if pr["feasible"]:
            for i, h in enumerate(pr["plan"]):
                assert h in lr["points"][i]["values"]


def test_lock_feasibility_implies_complete_plan():
    """对报出的每个可锁定值，DP 结果应能重建一条锁定它的完整可行坡线。"""
    rng = random.Random(555)
    for _ in range(30):
        n = rng.randint(3, N_MAX)
        ground = [rng.randint(H_MIN, H_MAX) for _ in range(n)]
        r = lockable_heights(ground)
        if not r["feasible"]:
            continue
        for i, p in enumerate(r["points"]):
            # 抽查每点最小/最大/中间可锁值，构造锁定问题并验证可行
            picks = {p["values"][0], p["values"][-1]}
            if len(p["values"]) > 2:
                picks.add(p["values"][len(p["values"]) // 2])
            for h in picks:
                # 独立锁定 DP 复核该值确实能撑起一条完整可行坡线
                assert _locked_planner(ground, i, h), (ground, i, h)


def _locked_planner(ground, pos, h):
    """独立小 DP：锁定 plan[pos]=h 时是否存在完整可行坡线（布尔可达）。"""
    from app.planner import _bwd_step, _fwd_step, _MASK_RECENT

    n = len(ground)
    masks = 1 << 4
    fwd = [[0] * masks for _ in range(n)]
    fwd[0][_MASK_RECENT] = 1 << ground[0]
    for k in range(1, n):
        fwd[k] = _fwd_step(fwd[k - 1], k)
        if k == pos:
            for m in range(masks):
                fwd[k][m] &= 1 << h

    bwd = [[0] * masks for _ in range(n)]
    for m in range(masks):
        bwd[n - 1][m] = 1 << ground[-1]
    for seg in range(n - 2, -1, -1):
        bwd[seg] = _bwd_step(bwd[seg + 1], seg)
        if seg == pos:
            for m in range(masks):
                bwd[seg][m] &= 1 << h

    return any(fwd[pos][m] & bwd[pos][m] for m in range(masks)) != 0


def test_n16_runs_fast():
    import time

    ground = [10 + 11 * i for i in range(N_MAX)]
    start = time.perf_counter()
    r = lockable_heights(ground)
    assert time.perf_counter() - start < 1
    assert r["feasible"] is False
