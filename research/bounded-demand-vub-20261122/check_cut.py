#!/usr/bin/env python3
"""Independent finite-grid CFLP cut check; no LP/MIP solver is used.

Every binary facility set and every legal half-grid fraction allocation of
these small complete graphs is enumerated.  Mixed positive-demand flows and
capacities are multiples of .5, so the enumeration includes transport LP
optima (the scaled transportation matrix is totally unimodular).  Zero-demand
rows include every opened-site vertex as well as half/half splits.
"""
import importlib.util
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import sys


TOL = 1e-9


def compositions(total, size):
    if size == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, size - 1):
            yield (first,) + tail


def enumerate_allocations(data, sites):
    """Yield exact (shipping, positive shipping) for all finite legal x."""
    if not sites:
        return
    d = [Fraction(str(v)) for v in data["customers"]["demands"]]
    s = [Fraction(str(v)) for v in data["plants"]["capacities"]]
    c = [[Fraction(str(v)) for v in row] for row in data["shipping_costs"]]
    if sum(s[j] for j in sites) < sum(d):
        return
    choices = []
    for demand, costs in zip(d, c):
        tokens = int(demand * 2) if demand else 2
        assert not demand or tokens == demand * 2
        rows = []
        for counts in compositions(tokens, len(sites)):
            x = [Fraction(count, tokens) for count in counts]
            loads = [demand * fraction for fraction in x]
            cost = sum(costs[j] * fraction for j, fraction in zip(sites, x))
            rows.append((loads, cost, cost if demand else Fraction(0)))
        choices.append(rows)

    def visit(i, loads, total, positive):
        if i == len(d):
            yield total, positive
            return
        for row_loads, cost, positive_cost in choices[i]:
            next_loads = [a + b for a, b in zip(loads, row_loads)]
            if all(load <= s[j] for j, load in zip(sites, next_loads)):
                yield from visit(i + 1, next_loads, total + cost,
                                 positive + positive_cost)

    yield from visit(0, [Fraction(0)] * len(sites), Fraction(0), Fraction(0))


def close(actual, expected, label):
    assert math.isfinite(actual), (label, actual)
    assert abs(actual - expected) <= TOL * max(1.0, abs(expected)), (
        label, actual, expected)


def check_fixture(make_cut, data, alphas, support_cases):
    d = data["customers"]["demands"]
    s = data["plants"]["capacities"]
    f = data["plants"]["fixed_costs"]
    c = data["shipping_costs"]
    positive = [i for i, demand in enumerate(d) if demand > 0]
    zero_floor = math.fsum(min(c[i]) for i, demand in enumerate(d) if demand == 0)
    m = data["num_plants"]
    cuts = []
    for alpha in alphas:
        assert len(alpha) == len(positive)
        cut = make_cut(data, alpha)
        mu = [max([0.0] + [a - c[i][j] / d[i]
                            for i, a in zip(positive, alpha)]) for j in range(m)]
        intercept = math.fsum(d[i] * a for i, a in zip(positive, alpha)) + zero_floor
        close(float(cut["intercept"]), intercept, "intercept")
        assert len(cut["coeff"]) == len(cut["mu"]) == m
        for j in range(m):
            close(float(cut["mu"][j]), mu[j], ("all-facility mu", j))
            close(float(cut["legacy_coeff"][j]), -s[j] * mu[j], ("legacy coeff", j))
            price = float(cut["knapsack_prices"][j])
            assert price >= 0.0
            terms = [(a - c[i][j] / d[i], d[i]) for i, a in zip(positive, alpha)]
            nu = [max(0.0, beta - price) for beta, demand in terms]
            dual_value = s[j] * price + math.fsum(demand * v for (_, demand), v in zip(terms, nu))
            close(float(cut["coeff"][j]), -dual_value, ("knapsack dual coeff", j))
            assert dual_value <= s[j] * mu[j] + TOL * max(1.0, abs(s[j] * mu[j]))
            remaining, fill_value = s[j], 0.0
            for beta, demand in sorted(terms, reverse=True):
                take = min(remaining, demand) if beta > 0.0 else 0.0
                remaining -= take
                fill_value += beta * take
            close(dual_value, fill_value, ("independent knapsack dual/primal", j))
            close(float(cut["knapsack_dual_values"][j]), dual_value, ("recorded dual value", j))
            close(float(cut["knapsack_primal_values"][j]), fill_value, ("recorded fill value", j))
            for (beta, demand), v in zip(terms, nu):
                assert beta <= price + v + TOL
            for i, a in zip(positive, alpha):
                assert a - float(cut["mu"][j]) <= c[i][j] / d[i] + TOL
        cuts.append(cut)

    stats = {"binary_sets": 2 ** m, "feasible_sets": 0, "allocations": 0,
             "cuts": len(cuts), "supports": []}
    minima = {}
    for binary in itertools.product((0, 1), repeat=m):
        sites = tuple(j for j, opened in enumerate(binary) if opened)
        fixed = math.fsum(f[j] for j in sites)
        bounds = [float(cut["intercept"]) + math.fsum(float(cut["coeff"][j])
                  for j in sites) for cut in cuts]
        count, shipping_min, positive_min = 0, math.inf, math.inf
        for shipping, positive_cost in enumerate_allocations(data, sites):
            count += 1
            shipping_min = min(shipping_min, float(shipping))
            positive_min = min(positive_min, float(positive_cost))
            for bound in bounds:
                assert float(shipping) + TOL >= bound, (sites, shipping, bound)
                # Fixed fees are added separately, including negative zero-capacity fees.
                assert float(shipping) + fixed + TOL >= bound + fixed
        if count:
            stats["feasible_sets"] += 1
            stats["allocations"] += count
            minima[sites] = (shipping_min, positive_min)

    for cut_index, sites, expected_zero_slack in support_cases:
        cut = cuts[cut_index]
        shipping_min, positive_min = minima[tuple(sites)]
        bound = float(cut["intercept"]) + math.fsum(float(cut["coeff"][j])
                                                                 for j in sites)
        positive_bound = bound - zero_floor
        close(positive_bound, positive_min, "current-S positive support")
        current_zero = math.fsum(min(c[i][j] for j in sites)
                                for i, demand in enumerate(d) if demand == 0)
        close(current_zero - zero_floor, expected_zero_slack, "zero global slack")
        close(shipping_min - bound, expected_zero_slack, "current-S total slack")
        stats["supports"].append({"cut": cut_index, "S": list(sites),
                                  "positive_gap": positive_min - positive_bound,
                                  "zero_global_slack": expected_zero_slack})
    return stats, minima, cuts


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[2] / "products" / "frontieror-cflp-study"))
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("joint_checked", here / "joint.py")
    joint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joint)
    mixed = {"num_customers": 3, "num_plants": 4,
             "customers": {"demands": [1.0, .5, 0.0]},
             "plants": {"capacities": [.5, 1.0, 1.5, 0.0],
                        "fixed_costs": [1.0, 2.0, 1.0, -2.0]},
             "shipping_costs": [[-1.0, 2.0, 1.0, -100.0],
                                [2.0, -1.0, 1.0, -10.0],
                                [-3.0, -2.0, 4.0, -1.0]]}
    result, minima, cuts = check_fixture(
        joint.make_cut, mixed, [[2.0, -2.0], [1.0, -2.0], [-3.0, 4.0]],
        [(0, (0, 1, 3), 0.0), (1, (1, 2, 3), 1.0)])
    # A fixed-S zero constant carried to other S has a concrete violating witness.
    bad_bound = float(cuts[1]["intercept"]) + 1.0 + math.fsum(
        float(cuts[1]["coeff"][j]) for j in (0, 1, 3))
    assert minima[(0, 1, 3)][0] < bad_bound - TOL
    zero = {"num_customers": 2, "num_plants": 3,
            "customers": {"demands": [0.0, 0.0]},
            "plants": {"capacities": [0.0, 0.0, .25],
                       "fixed_costs": [1.0, -2.0, 0.0]},
            "shipping_costs": [[-5.0, -1.0, 4.0], [2.0, -3.0, -7.0]]}
    zero_result, _, _ = check_fixture(joint.make_cut, zero, [[]],
                                      [(0, (0, 1, 2), 0.0)])
    print(json.dumps({"status": "PASS", "mixed": result, "all_zero": zero_result,
                      "fixed_S_zero_constant_counterexample": "PASS",
                      "scope": "finite fixtures; global cut and current-S support only"},
                     allow_nan=False))


if __name__ == "__main__":
    main()
