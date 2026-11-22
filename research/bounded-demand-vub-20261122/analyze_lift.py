#!/usr/bin/env python3
"""Read-only demand/VUB support-function lift for stored initial large4 alpha.

No LP/MIP solver, new facility sets, source downloads, or product imports.
Uses a fractional-knapsack primal and an outward-rounded threshold dual.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upper_float(value):
    candidate = float(value)
    return math.nextafter(candidate, math.inf) if Fraction.from_float(candidate) < value else candidate


def lower_float(value):
    candidate = float(value)
    return math.nextafter(candidate, -math.inf) if Fraction.from_float(candidate) > value else candidate


def distribution(values):
    ordered = sorted(values)

    def quantile(p):
        pos = (len(ordered) - 1) * p
        left = int(pos)
        return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (pos - left)

    return {"count": len(values), "min": min(values), "p25": quantile(.25),
            "median": quantile(.5), "p75": quantile(.75), "p90": quantile(.9),
            "p95": quantile(.95), "max": max(values),
            "sum": math.fsum(values), "mean": math.fsum(values) / len(values),
            "positive_over_1e_7": sum(v > 1e-7 for v in values)}


def fill(demand, profit, capacity):
    """Primal fractional-knapsack optimizer; quantity bounds retain each d_i."""
    order = sorted((i for i, b in enumerate(profit) if b > 0), key=lambda i: (-profit[i], i))
    total_positive = math.fsum(demand[i] for i in order)
    if capacity == 0:
        return 0.0, max([0.0] + profit), 0, 0.0
    remaining, terms, used, marginal = capacity, [], 0, 0.0
    for i in order:
        q = min(demand[i], remaining)
        if q <= 0:
            break
        terms.append(q * profit[i])
        remaining -= q
        used += 1
        marginal = profit[i]
        if remaining <= 0:
            break
    threshold = 0.0 if total_positive <= capacity else marginal
    return math.fsum(terms), threshold, used, capacity - remaining


def main():
    start = time.monotonic()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    signal.alarm(180)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3] / "products/frontieror-cflp-study")
    parser.add_argument("--instance", type=Path, required=True, help="Already available pinned large_instance_4.json")
    args = parser.parse_args()
    repo = args.repo
    paths = {
        "prior": repo / "results/cover/large4-hybrid-s1/solution.json",
        "metrics": repo / "results/archive/joint-master-conditional/gate/metrics.json",
        "candidate1": repo / "results/archive/joint-master-conditional/gate/candidate-1.json",
        "candidate2": repo / "results/archive/joint-master-conditional/gate/candidate-2.json",
    }
    manifest = load(repo / "source/input-hashes.json")
    expected = next(x["sha256"] for x in manifest["inputs"] if x["file"] == "large_instance_4.json")
    assert digest(args.instance) == expected
    data, prior, metrics = load(args.instance), load(paths["prior"]), load(paths["metrics"])
    cut = metrics["cuts"][0]
    assert cut["phase"] == "initial"
    d, s, f, c = data["customers"]["demands"], data["plants"]["capacities"], data["plants"]["fixed_costs"], data["shipping_costs"]
    positive = [i for i, v in enumerate(d) if v > 0]
    dp = [d[i] for i in positive]
    alpha, stored_mu = cut["alpha"], cut["mu"]
    assert len(alpha) == len(positive) and len(stored_mu) == len(s)
    assert all(math.isfinite(v) for v in alpha) and all(v >= 0 for v in d + s)
    opened = set(prior["open_plants"])
    df = [Fraction.from_float(float(v)) for v in dp]
    exact_intercept = sum((a * Fraction.from_float(float(b)) for a, b in zip(df, alpha)), Fraction(0))
    exact_intercept += sum((Fraction.from_float(float(min(c[i]))) for i, v in enumerate(d) if v == 0), Fraction(0))
    intercept_lower = lower_float(exact_intercept)
    columns, new_c, safe_old_c = [], [], []
    max_closure_error, max_dual_violation, max_primal_dual_gap = 0.0, -math.inf, 0.0
    max_upper_rounding_change, fallback_count, arcs = 0.0, 0, 0
    for j, capacity in enumerate(s):
        beta, profit_upper = [], []
        for a, i in zip(alpha, positive):
            unit = c[i][j] / d[i]
            assert math.isfinite(unit)
            b = a - unit
            # Directed bounds on the exact quotient/subtraction of supplied binary floats.
            lower_unit = math.nextafter(unit, -math.inf)
            b_upper = math.nextafter(a - lower_unit, math.inf)
            beta.append(b); profit_upper.append(b_upper)
        arcs += len(beta)
        mu_raw, mu_upper = max([0.0] + beta), max([0.0] + profit_upper)
        max_closure_error = max(max_closure_error, abs(mu_raw - stored_mu[j]))
        primal, _, primal_used, primal_mass = fill(dp, beta, capacity)
        _, threshold, _, _ = fill(dp, profit_upper, capacity)
        assert threshold >= 0
        exact_dual = Fraction.from_float(float(capacity)) * Fraction.from_float(threshold)
        for b, weight in zip(profit_upper, df):
            if b > threshold:
                nu = math.nextafter(b - threshold, math.inf)
                assert Fraction.from_float(nu) >= Fraction.from_float(b) - Fraction.from_float(threshold)
                exact_dual += weight * Fraction.from_float(nu)
            else:
                nu = 0.0
            assert nu >= 0
            max_dual_violation = max(max_dual_violation, b - (threshold + nu))
        exact_old_upper = Fraction.from_float(float(capacity)) * Fraction.from_float(mu_upper)
        scheme = "threshold"
        if exact_dual > exact_old_upper:
            # Capacity-price dual is also a feasible knapsack dual: lambda=mu, nu=0.
            exact_dual = exact_old_upper
            scheme = "capacity_dual_rounding_tie"
            fallback_count += 1
        upper_new, upper_old = upper_float(exact_dual), upper_float(exact_old_upper)
        assert upper_new <= upper_old
        assert upper_new >= primal - max(1e-8, abs(primal) * 1e-12)
        assert abs(upper_new - primal) <= max(1e-8, abs(primal) * 1e-10)
        max_primal_dual_gap = max(max_primal_dual_gap, abs(upper_new - primal))
        max_upper_rounding_change = max(max_upper_rounding_change, upper_old - (-cut["coeff"][j]))
        new_c.append(upper_new); safe_old_c.append(upper_old)
        columns.append({
            "facility": j, "open_in_prior": j in opened,
            "old_capacity_coefficient_recorded": -cut["coeff"][j],
            "old_capacity_coefficient_upper": upper_old,
            "new_vub_coefficient_upper": upper_new,
            "knapsack_primal_value": primal,
            "threshold_lambda": threshold,
            "chosen_dual_scheme": scheme,
            "gain_vs_recorded_old": -cut["coeff"][j] - upper_new,
            "gain_vs_outward_old": upper_old - upper_new,
            "primal_items_used": primal_used,
            "primal_mass_used": primal_mass,
        })
    assert max_closure_error < 1e-9 and max_dual_violation <= 1e-12
    closed = [j for j in range(len(s)) if j not in opened]
    open_differences = [new_c[j] - (-cut["coeff"][j]) for j in sorted(opened)]
    assert max(abs(v) for v in open_differences) < 1e-7
    floor = math.fsum(min(row) for row in c)

    def new_bound(sites):
        exact = Fraction.from_float(intercept_lower) - sum((Fraction.from_float(new_c[j]) for j in sites), Fraction(0))
        return lower_float(exact)

    evaluations = []
    for label, path in [("prior", paths["prior"]), ("candidate1", paths["candidate1"]), ("candidate2", paths["candidate2"])]:
        sol = load(path)
        sites = sol["open_plants"]
        fixed = math.fsum(f[j] for j in sites)
        shipping = math.fsum(c[int(i)][int(j)] * x for i, row in sol["assignments"].items() for j, x in row.items())
        assert abs(shipping + fixed - sol["objective_value"]) < 1e-6
        old_bound = cut["intercept"] + math.fsum(cut["coeff"][j] for j in sites)
        lifted = new_bound(sites)
        assert lifted <= shipping + 1e-6
        old_envelope = max([floor] + [z["intercept"] + math.fsum(z["coeff"][j] for j in sites) for z in metrics["cuts"]])
        new_envelope = max(old_envelope, lifted)
        evaluations.append({
            "existing_witness": label, "open_facilities": len(sites),
            "outside_initial_S": len(set(sites) - opened),
            "old_initial_shipping_cut": old_bound,
            "new_initial_shipping_cut": lifted,
            "initial_cut_gain": lifted - old_bound,
            "old_initial_cut_true_shipping_gap": shipping - old_bound,
            "new_initial_cut_true_shipping_gap": shipping - lifted,
            "old_two_cut_plus_floor_envelope": old_envelope,
            "new_lift_plus_old_envelope": new_envelope,
            "envelope_gain": new_envelope - old_envelope,
            "true_shipping_recomputed_from_existing_witness": shipping,
            "fixed_cost": fixed, "recorded_whole_cost": sol["objective_value"],
        })
    prior_support_error = evaluations[0]["true_shipping_recomputed_from_existing_witness"] - evaluations[0]["new_initial_shipping_cut"]
    assert abs(prior_support_error) < 1e-6
    usage = resource.getrusage(resource.RUSAGE_SELF)
    output = {
        "status": "PASS_READ_ONLY_DEMAND_LIFT",
        "scope": "All 750 facility coefficients for stored initial alpha; evaluation on existing witnesses only",
        "input_sha256": expected,
        "source_hashes": {name: digest(path) for name, path in paths.items()},
        "facilities": len(s), "positive_customers": len(positive), "arcs_checked": arcs,
        "original_intercept": cut["intercept"], "lift_intercept_lower": intercept_lower,
        "new_price_scheme": "Fractional-knapsack threshold dual with directed profit bounds and exact-rational dual accumulation, rounded outward",
        "max_stored_mu_closure_error": max_closure_error,
        "max_all_arc_dual_violation": max_dual_violation,
        "max_knapsack_primal_dual_abs_gap": max_primal_dual_gap,
        "max_old_coefficient_outward_rounding_change": max_upper_rounding_change,
        "capacity_dual_rounding_tie_columns": fallback_count,
        "old_coefficient_sum_recorded": math.fsum(-v for v in cut["coeff"]),
        "old_coefficient_sum_outward": math.fsum(safe_old_c),
        "new_coefficient_sum_upper": math.fsum(new_c),
        "all_columns_gain_vs_recorded_old_sum": math.fsum(-cut["coeff"][j] - new_c[j] for j in range(len(s))),
        "max_domination_violation_vs_recorded_old": max(0.0, max(new_c[j] + cut["coeff"][j] for j in range(len(s)))),
        "max_domination_violation_vs_outward_old": 0.0,
        "open_coefficient_delta_new_minus_recorded_old": distribution(open_differences),
        "closed_absolute_gain_vs_recorded_old": distribution([-cut["coeff"][j] - new_c[j] for j in closed]),
        "closed_relative_gain_vs_recorded_old": distribution([(-cut["coeff"][j] - new_c[j]) / (-cut["coeff"][j]) for j in closed]),
        "initial_prior_new_support_error": prior_support_error,
        "existing_witness_evaluations": evaluations,
        "columns": columns,
        "native_solver_runs": 0, "new_facility_sets": 0, "new_true_cost_measurements": 0,
        "resource_receipt": {
            "cpu_affinity_count": len(os.sched_getaffinity(0)), "address_space_limit_bytes": 1024**3,
            "cpu_limit_seconds": 60, "wall_limit_seconds": 180,
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": usage.ru_utime + usage.ru_stime, "peak_rss_kib": usage.ru_maxrss,
        },
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
