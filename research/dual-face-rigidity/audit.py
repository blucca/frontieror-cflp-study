# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Blucca
#!/usr/bin/env python3
"""Read-only numerical audit of the stored large4 primal support and dual.

No solver, product imports, downloads, or new CFLP decisions. The optional
--instance reads an already available pinned input for edge-level checks.
"""
import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal_text(value):
    text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def main():
    began = time.monotonic()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    signal.alarm(180)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--instance", type=Path, help="Existing large_instance_4.json; never downloaded by this script")
    args = parser.parse_args()
    repo = args.repo
    paths = {
        "prior": repo / "results/cover/large4-hybrid-s1/solution.json",
        "metrics": repo / "results/archive/joint-master-conditional/gate/metrics.json",
        "provenance": repo / "results/provenance/source-large4.json",
    }
    sol, metrics = read(paths["prior"]), read(paths["metrics"])
    provenance = read(paths["provenance"])[0]
    cut, oracle = metrics["cuts"][0], metrics["initial_oracle"]
    assert provenance["case"] == "large4" and cut["phase"] == "initial"
    n, m = provenance["customers"], provenance["plants"]
    sites = set(sol["open_plants"])
    assert len(sites) == len(sol["open_plants"]) == 250
    assert set(sol["assignments"]) == {str(i) for i in range(n)}
    alpha, mu = cut["alpha"], cut["mu"]
    assert len(alpha) == n and len(mu) == m
    assert all(math.isfinite(x) for x in alpha + mu) and min(mu) >= 0
    parent = list(range(n + len(sites)))
    jnode = {j: n + k for k, j in enumerate(sorted(sites))}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    edges, fractions, row_error = [], [], 0.0
    for istr, row in sol["assignments"].items():
        i = int(istr)
        row_error = max(row_error, abs(math.fsum(row.values()) - 1))
        for jstr, x in row.items():
            j = int(jstr)
            assert math.isfinite(x) and 0 <= x <= 1 + 1e-9
            assert j in sites or x == 0
            if x > 0:
                union(i, jnode[j])
                edges.append((i, j, x))
                fractions.append(x)
    components = Counter(find(i) for i in range(len(parent)))
    assert len(components) == 1 and len(edges) == len(parent) - 1 == 1749
    assert row_error <= 1e-7
    open_mu = [mu[j] for j in sites]
    closed_mu = [mu[j] for j in range(m) if j not in sites]
    zero_open = sorted(j for j in sites if mu[j] == 0)
    assert zero_open == [610] and min(closed_mu) > 0
    assert oracle["fixed_s_model_status"] == "HighsModelStatus.kOptimal"
    assert oracle["independent"]["pass"]
    assert abs(cut["positive_support_error"]) < 1e-8
    assert cut["all_arc_max_violation"] < 1e-12
    assert provenance["capacity_min_max"][0] == provenance["capacity_min_max"][1]
    assert provenance["fixed_min_max"][0] >= 0 and provenance["demand_min_max"][0] > 0
    demand = Fraction(str(provenance["demand"]))
    capacity = Fraction(str(provenance["capacity_min_max"][0]))
    total_capacity, capacity_s = m * capacity, len(sites) * capacity
    lam = (demand / total_capacity + 1) / 2
    core_capacity = lam * total_capacity
    assert demand == capacity_s and 0 < lam < 1 and core_capacity > demand
    # 735 proposed numerical face band; distinct from 734's looser support check.
    tol = max(1e-6, abs(cut["current_shipping"]) * 1e-10)
    local_negative_shift_limit = tol / float(capacity)
    smallest_positive_mu = min(v for v in mu if v > 0)
    assert local_negative_shift_limit < smallest_positive_mu
    result = {
        "status": "PASS_COMPLETE_INPUT_NUMERICAL_AUDIT" if args.instance else "PASS_STORED_SUPPORT_AUDIT",
        "scope": "Stored initial large4 fixed-S only; exact-face theorem uses the original Optimal/primal-dual records plus audited numerical tolerances",
        "primal_support": {
            "customers": n, "open_facilities": len(sites), "nodes": len(parent),
            "strictly_positive_edges": len(edges), "components": len(components),
            "component_sizes": sorted(components.values()), "is_spanning_tree": True,
            "min_positive_fraction": min(fractions),
            "min_positive_flow_lower_bound_from_recorded_demand_min": min(fractions) * provenance["demand_min_max"][0],
            "positive_fractions_below_1e_8": sum(v < 1e-8 for v in fractions),
            "row_fraction_max_error": row_error,
        },
        "original_numerical_receipts": {
            "native_status": oracle["fixed_s_model_status"],
            "positive_shipping": cut["current_shipping"],
            "positive_support_error": cut["positive_support_error"],
            "all_arc_count": cut["all_source_arcs_checked"],
            "all_arc_max_violation": cut["all_arc_max_violation"],
            "max_capacity_excess": oracle["independent"]["max_capacity_excess"],
        },
        "dual_face": {
            "min_open_mu": min(open_mu), "zero_open_mu_facilities": zero_open,
            "min_closed_mu": min(closed_mu), "closed_mu_strictly_positive": True,
            "all_closed_mu_minimal": "By stored joint.make_cut all-arc maximum construction; independently recomputed when --instance supplied",
            "uniform_capacity_exact_decimal": decimal_text(capacity),
            "demand_exact_decimal": decimal_text(demand), "capacity_s_exact_decimal": decimal_text(capacity_s),
            "D_minus_capacityS_exact": str(demand - capacity_s),
            "total_capacity_exact_decimal": decimal_text(total_capacity),
            "recorded_numpy_total_capacity": provenance["capacity"],
            "core_lambda_exact": str(lam), "core_capacity_exact": decimal_text(core_capacity),
            "D_minus_core_capacity_exact": str(demand - core_capacity),
            "exact_optimal_face": "alpha'=alpha+t, mu'_S=mu_S+t; t>=0; minimal closed mu'=mu_closed+t",
            "core_objective_delta": "t*(D-sum_j s_j*yCore_j) = -29929*t",
            "unique_core_maximum_t": 0,
            "conclusion": "The stored initial cut already maximizes this strict-core objective on the exact optimal dual face",
        },
        "near_optimal_boundary": {
            "planned_735_face_tolerance": tol,
            "pure_negative_common_shift": "For -min_positive_mu<t<0, only open facility610 clips: gS(t)-Q=s610*t",
            "max_negative_common_shift_magnitude": local_negative_shift_limit,
            "max_core_gain_pure_common_shift": tol * float((core_capacity - demand - lam * capacity) / capacity),
            "arbitrary_near_face": "May be non-translational: sum_positive_edges(q*r)<=tol in exact arithmetic; exact-face translation certificate applies when residual budget=0",
            "scope": "No claim that all near-face points are common shifts or that other S share this tree",
        },
    }
    if args.instance:
        data = read(args.instance)
        expected = read(repo / "source/input-hashes.json")
        wanted = next(x["sha256"] for x in expected["inputs"] if x["file"] == "large_instance_4.json")
        assert sha(args.instance) == wanted, "Original pinned input only"
        d, s, c = data["customers"]["demands"], data["plants"]["capacities"], data["shipping_costs"]
        assert all(v > 0 for v in d)
        assert sum(Fraction(str(v)) for v in d) == demand
        assert sum(Fraction(str(s[j])) for j in sites) == capacity_s
        loads = [[] for _ in range(m)]
        rows = [[] for _ in range(n)]
        residuals, flows, costs = [], [], []
        for i, j, x in edges:
            q = d[i] * x
            loads[j].append(q); rows[i].append(q); flows.append(q)
            costs.append(c[i][j] * x)
            residuals.append(c[i][j] / d[i] - alpha[i] + mu[j])
        closure = [0.0] * m
        all_arc_violation = -math.inf
        for i, row in enumerate(c):
            for j, cost in enumerate(row):
                price = alpha[i] - cost / d[i]
                closure[j] = max(closure[j], price)
                all_arc_violation = max(all_arc_violation, price - mu[j])
        equality_error = max(abs(v) for v in residuals)
        assert equality_error < 1e-8 and all_arc_violation < 1e-8
        assert max(abs(a - b) for a, b in zip(closure, mu)) < 1e-8
        recomputed_shipping = math.fsum(costs)
        assert abs(recomputed_shipping - cut["current_shipping"]) < 1e-6
        objective = recomputed_shipping + math.fsum(data["plants"]["fixed_costs"][j] for j in sites)
        assert abs(objective - sol["objective_value"]) < 1e-6
        row_flow_error = max(abs(math.fsum(row) - d[i]) for i, row in enumerate(rows))
        capacity_slack = [s[j] - math.fsum(loads[j]) for j in sites]
        assert row_flow_error < 1e-6 and min(capacity_slack) > -1e-6
        result["independent_existing_input_readback"] = {
            "input_sha256": wanted,
            "strictly_positive_flow_min": min(flows),
            "positive_arc_dual_equality_max_abs_error": equality_error,
            "positive_arc_reduced_cost_min_max": [min(residuals), max(residuals)],
            "positive_arc_weighted_complementarity_error": math.fsum(q * r for q, r in zip(flows, residuals)),
            "all_arc_max_violation": all_arc_violation,
            "closed_mu_closure_max_abs_error": max(abs(closure[j] - mu[j]) for j in range(m) if j not in sites),
            "demand_flow_max_error": row_flow_error,
            "capacity_slack_min_max_on_S": [min(capacity_slack), max(capacity_slack)],
            "shipping_recomputed": recomputed_shipping,
            "shipping_recompute_error": recomputed_shipping - cut["current_shipping"],
            "whole_objective_recompute_error": objective - sol["objective_value"],
            "single_edge_near_face_reduced_cost_bound_using_min_flow": tol / min(flows),
            "native_solver_runs": 0,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["certificate_resources"] = {
        "scope": "JSON/source arithmetic only",
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "address_space_limit_bytes": 1024**3,
        "cpu_limit_seconds": 60, "wall_limit_seconds": 180,
        "wall_seconds": time.monotonic() - began,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
