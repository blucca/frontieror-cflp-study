"""Whole CFLP input, primal witness, and independent sparse validation.

FrontierOR / Klose & Drexl input attribution: CC BY 4.0.
All original arcs remain available; omitted solution entries mean zero.
"""
import json
import math
import numpy as np


def load(path):
    with open(path) as f:
        data = json.load(f)
    arrays(data)
    return data


def arrays(data):
    n, m = data["num_customers"], data["num_plants"]
    assert type(n) is int and type(m) is int and n > 0 and m > 0
    d = np.asarray(data["customers"]["demands"], dtype=np.float64)
    s = np.asarray(data["plants"]["capacities"], dtype=np.float64)
    f = np.asarray(data["plants"]["fixed_costs"], dtype=np.float64)
    c = np.asarray(data["shipping_costs"], dtype=np.float64)
    assert d.shape == (n,) and s.shape == f.shape == (m,) and c.shape == (n, m)
    assert all(np.isfinite(a).all() for a in (d, s, f, c))
    assert (d >= 0).all() and (s >= 0).all(), "nonnegative demand/capacity"
    assert float(s.sum()) + 1e-7 >= float(d.sum()), "complete graph capacity"
    return d, s, f, c


def verify(data, sol):
    """Check every demand, capacity, bound, VUB, aggregate, and true cost.

    Uses sparse witness and math.fsum, independently of native matrices.
    The published official checker is run separately by the controller.
    """
    n, m = data["num_customers"], data["num_plants"]
    d, s = data["customers"]["demands"], data["plants"]["capacities"]
    opened = sol["open_plants"]
    assert all(type(j) is int and 0 <= j < m for j in opened)
    assert len(opened) == len(set(opened)), "unique binary facility witness"
    opened = set(opened)
    assign = sol["assignments"]
    assert set(assign) == {str(i) for i in range(n)}, "all customer rows"
    loads = [[] for _ in range(m)]
    shipping, row_error, bound_error = [], 0.0, 0.0
    for i in range(n):
        row = assign[str(i)]
        assert isinstance(row, dict)
        for key, x in row.items():
            j = int(key)
            assert key == str(j) and 0 <= j < m
            assert type(x) in (int, float) and math.isfinite(x)
            bound_error = max(bound_error, -x, x - 1.0)
            assert -1e-9 <= x <= 1.0 + 1e-9
            assert j in opened or abs(x) <= 1e-10, "all VUBs x<=binary y"
            loads[j].append(d[i] * x)
            shipping.append(data["shipping_costs"][i][j] * x)
        err = abs(math.fsum(row.values()) - 1.0)
        row_error = max(row_error, err)
        assert err <= 1e-7, "complete demand fraction"
    cap_error = max(math.fsum(v) - (s[j] if j in opened else 0.0)
                    for j, v in enumerate(loads))
    assert cap_error <= 1e-6, "all original floating capacities"
    assert math.fsum(s[j] for j in opened) + 1e-7 >= math.fsum(d)
    objective = math.fsum(shipping) + math.fsum(data["plants"]["fixed_costs"][j]
                                             for j in opened)
    reported = sol["objective_value"]
    assert type(reported) in (int, float) and math.isfinite(reported)
    assert abs(reported - objective) <= max(1e-6, abs(objective) * 1e-10)
    return {"pass": True, "objective": objective, "open_count": len(opened),
            "arcs": sum(len(x) for x in assign.values()), "max_demand_error": row_error,
            "max_capacity_excess": cap_error, "max_bound_excess": bound_error}


def from_flows(data, q, opened=None):
    d, s, f, c = arrays(data)
    assert q.shape == c.shape and np.isfinite(q).all() and q.min() >= -1e-8
    q = np.maximum(q, 0.0)
    # Preserve every positive returned flow, including numerical tiny flows.
    used = set(int(j) for j in np.nonzero(q.sum(axis=0) > 0.0)[0])
    # Explicit binary facilities are authoritative. A SCIP tiny numerical flow
    # formerly opened an extra $178 facility in the witness; retain the flow
    # within the validator's tolerance without changing its binary decision.
    sites = used if opened is None else set(int(j) for j in opened)
    sites |= set(int(j) for j in np.nonzero(f < 0.0)[0])
    if not sites:
        # The all-zero-demand case still requires fraction assignments.
        sites.add(int(np.argmin(f + c.sum(axis=0))))
    assignments = {}
    for i, demand in enumerate(d):
        if demand == 0:
            j = min(sites, key=lambda j: (c[i, j], j))
            assignments[str(i)] = {str(j): 1.0}
        else:
            assignments[str(i)] = {str(int(j)): float(q[i, j] / demand)
                                   for j in np.nonzero(q[i] > 0.0)[0]}
    total = math.fsum(data["plants"]["fixed_costs"][j] for j in sites)
    total += math.fsum(data["shipping_costs"][int(i)][int(j)] * x
                      for i, row in assignments.items() for j, x in row.items())
    sol = {"objective_value": total, "open_plants": sorted(sites), "assignments": assignments}
    verify(data, sol)
    return sol


def greedy(data, penalty=1.0, seed=1):
    """Immediate feasible complete-graph transport; fractional capacity stays float."""
    d, s, f, c = arrays(data)
    positive = np.nonzero(d > 0)[0]
    unit = np.divide(c, d[:, None], out=np.zeros_like(c), where=d[:, None] > 0)
    rent = np.divide(np.maximum(f, 0.0), s, out=np.zeros_like(f), where=s > 0)
    score = unit + penalty * rent[None, :]
    score[:, s == 0] = np.inf
    order = np.argsort(score, axis=1, kind="stable")
    regret = score[np.arange(len(d)), order[:, min(1, len(s) - 1)]] - score[np.arange(len(d)), order[:, 0]]
    rng = np.random.default_rng(seed)
    customers = sorted(positive, key=lambda i: (-regret[i], -d[i], rng.random()))
    remaining = s.copy()
    q = np.zeros_like(c)
    for i in customers:
        needed = d[i]
        for j in order[i]:
            take = min(needed, remaining[j])
            if take > 0:
                q[i, j] = take
                remaining[j] -= take
                needed -= take
            if needed <= 1e-10:
                break
        assert needed <= 1e-7, "full graph has sufficient total capacity"
    return from_flows(data, q)


if __name__ == "__main__":
    sample = {"num_customers": 3, "num_plants": 2,
              "customers": {"demands": [2, 1, 0]},
              "plants": {"capacities": [1.5, 2.0], "fixed_costs": [1, 2]},
              "shipping_costs": [[2, 6], [4, 1], [9, 3]]}
    q = np.asarray([[1.5, .5], [0, 1], [0, 0]], dtype=float)
    assert from_flows(sample, q)["objective_value"] == 10.0
    assert verify(sample, greedy(sample))["pass"]
    numerical = {"num_customers": 1, "num_plants": 2, "customers": {"demands": [1]},
                 "plants": {"capacities": [2, 2], "fixed_costs": [1, 178]},
                 "shipping_costs": [[1, 2]]}
    tiny = from_flows(numerical, np.asarray([[1-1e-12, 1e-12]]), opened=[0])
    assert tiny["open_plants"] == [0] and abs(tiny["objective_value"] - 2) < 1e-9
    bad = from_flows(sample, q)
    bad["assignments"]["0"] = {"0": 1.0}
    try:
        verify(sample, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("floating capacity violation must fail")
    print("PASS float-capacity / splitting / zero-demand / full cost / rejection")
