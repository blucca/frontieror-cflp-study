"""Exact full-arc transport on a previously chosen CFLP facility set.

Fixed-set LP status concerns transportation only. Facility choices stay fixed.
A pending native solve emits the already verified prior and exits via caller.
"""
import time
import math
import numpy as np
import common


def refine(data, prior, deadline, seed, metrics, pending_emit, dual_out=None):
    common.verify(data, prior)
    d, s, f, c = common.arrays(data)
    sites = np.asarray(sorted(prior["open_plants"]), dtype=np.int32)
    positive = np.flatnonzero(d > 0)
    n, m = c.shape
    p, k = len(positive), len(sites)
    metrics.update(scope="all_arcs_to_fixed_S", fixed_open_count=k,
                   prior_objective=prior["objective_value"], native_result_available=False,
                   all_source_arcs=n * m, fixed_s_arcs=p * k,
                   zero_demand_customers=n - p)
    assert k > 0 and set(np.flatnonzero(f < 0)) <= set(sites)
    if not p:
        metrics["termination"] = "ZERO_DEMAND_EXACT_FIXED_S"
        return common.from_flows(data, np.zeros_like(c), opened=sites)
    if deadline - time.monotonic() < .5:
        metrics["termination"] = "NO_POLISH_BUDGET"
        return prior

    import highspy
    began = time.monotonic()
    unit = c[np.ix_(positive, sites)] / d[positive, None]
    warm = np.zeros((p, k))
    lookup = {int(j): col for col, j in enumerate(sites)}
    for row, i in enumerate(positive):
        for key, fraction in prior["assignments"][str(int(i))].items():
            j = int(key)
            if j in lookup:
                warm[row, lookup[j]] = d[i] * fraction
            else:
                assert abs(fraction) <= 1e-10, "closed flow stays within original tolerance"
    assert np.max(abs(warm.sum(axis=1) - d[positive])) <= 1e-6
    assert np.max(warm.sum(axis=0) - s[sites]) <= 1e-6
    constant = math.fsum(float(f[j]) for j in sites)
    if p < n:
        constant += math.fsum(float(v) for v in c[np.ix_(d == 0, sites)].min(axis=1))
    warm_cost = math.fsum((unit * warm).ravel()) + constant
    # Zero-demand customers are re-assigned exactly within S; this can already
    # improve the prior even though its positive-demand warm flow is identical.
    assert warm_cost <= prior["objective_value"] + max(1e-6, abs(warm_cost) * 1e-10)

    columns = p * k
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = columns, p + k
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = unit.ravel(), np.zeros(columns), np.full(columns, np.inf)
    lp.row_lower_, lp.row_upper_ = np.r_[d[positive], np.full(k, -np.inf)], np.r_[d[positive], s[sites]]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.arange(columns + 1, dtype=np.int32) * 2
    indices = np.empty(columns * 2, dtype=np.int32)
    indices[0::2] = np.repeat(np.arange(p, dtype=np.int32), k)
    indices[1::2] = np.tile(p + np.arange(k, dtype=np.int32), p)
    lp.a_matrix_.index_, lp.a_matrix_.value_ = indices, np.ones(columns * 2)
    h = highspy.Highs()
    # The preceding arc-pool run initializes HiGHS' global scheduler with 2.
    # Keep the same setting; this fixed-set LP uses serial dual simplex.
    for key, value in {"output_flag": False, "threads": 2, "parallel": "off",
                       "solver": "simplex", "simplex_strategy": 1,
                       "primal_feasibility_tolerance": 1e-9,
                       "dual_feasibility_tolerance": 1e-8, "random_seed": seed}.items():
        assert h.setOptionValue(key, value) == highspy.HighsStatus.kOk
    assert h.passModel(lp) == highspy.HighsStatus.kOk
    assert h.setSolution(columns, np.arange(columns, dtype=np.int32), warm.ravel()) == highspy.HighsStatus.kOk
    metrics.update(model_rows=p + k, model_columns=columns, model_nnz=columns * 2,
                   solver_version=h.version(), warm_objective=warm_cost,
                   build_seconds=time.monotonic() - began)
    budget = max(0.0, deadline - time.monotonic())
    if budget < .1:
        metrics["termination"] = "POLISH_BUILD_BUDGET"
        return prior
    assert h.setOptionValue("time_limit", max(.001, budget - .1)) == highspy.HighsStatus.kOk
    h.startSolve()
    try:
        done, run_status = h.wait(max(0.0, deadline - time.monotonic()))
    except BaseException as error:
        metrics.update(termination="POLISH_WAIT_FALLBACK", wait_error=type(error).__name__)
        pending_emit("POLISH_WAIT_FALLBACK", pending=True)
        raise AssertionError("pending emitter must exit")
    if not done:
        metrics["termination"] = "POLISH_PENDING_FALLBACK"
        pending_emit("POLISH_PENDING_FALLBACK", pending=True)
        raise AssertionError("pending emitter must exit")
    status = h.getModelStatus()
    metrics.update(native_result_available=True, fixed_s_model_status=str(status),
                   run_status=str(run_status), elapsed_seconds=time.monotonic() - began)
    native = h.getSolution()
    if not native.value_valid:
        metrics["termination"] = "NO_VALID_RETURNED_FIXED_S_FLOW"
        return prior
    flow = np.asarray(native.col_value, dtype=float).reshape(p, k)
    assert np.isfinite(flow).all() and flow.min() >= -1e-8
    q = np.zeros_like(c)
    q[np.ix_(positive, sites)] = np.maximum(flow, 0)
    candidate = common.from_flows(data, q, opened=sites)
    metrics["independent"] = common.verify(data, candidate)
    # Conditional joint-master experiment: capture only a completed native LP.
    if dual_out is not None:
        dual_out.update(alpha=np.asarray(native.row_dual[:p], dtype=float),
                        capacity_dual=np.asarray(native.row_dual[p:], dtype=float),
                        sites=sites.copy(), positive=positive.copy(),
                        valid=bool(native.dual_valid), optimal=status == highspy.HighsModelStatus.kOptimal,
                        positive_primal=math.fsum((unit * flow).ravel()))
    metrics.update(termination="RETURNED_VERIFIED_FIXED_S", fixed_s_objective=candidate["objective_value"],
                   native_fixed_s_objective=h.getInfo().objective_function_value + constant,
                   improvement=prior["objective_value"] - candidate["objective_value"],
                   facilities_unchanged=candidate["open_plants"] == list(sites))
    assert metrics["facilities_unchanged"]
    return candidate if candidate["objective_value"] < prior["objective_value"] - 1e-7 else prior



def capacity_cover(data, prior, seed=1):
    """One deterministic fee/capacity cover inside S, with full-graph routing.

    For equal capacities this picks the least fixed-cost feasible sub-set of S.
    Unequal capacities use the same greedy heuristic, not an optimality claim.
    """
    d, s, f, c = common.arrays(data)
    demand = math.fsum(float(v) for v in d)
    sites = set(prior["open_plants"])
    if demand == 0:
        return prior, {"termination": "ZERO_DEMAND_KEEP_PRIOR"}
    loads = np.zeros(len(s))
    for row, assignments in prior["assignments"].items():
        for key, fraction in assignments.items():
            loads[int(key)] += d[int(row)] * fraction
    chosen = {j for j in sites if f[j] < 0}
    order = sorted((j for j in sites - chosen if s[j] > 0),
                   key=lambda j: (f[j] / s[j], -loads[j] / s[j], j))
    for j in order:
        if math.fsum(float(s[k]) for k in chosen) >= demand:
            break
        chosen.add(j)
    assert math.fsum(float(s[j]) for j in chosen) >= demand
    if chosen == sites:
        return prior, {"termination": "S_ALREADY_FEE_CAPACITY_COVER"}
    # Masking closed capacities is enough to reuse the complete-graph greedy;
    # original costs, float capacities on S, and every negative fee stay intact.
    restricted = dict(data)
    restricted["plants"] = dict(data["plants"])
    restricted["plants"]["capacities"] = [float(s[j]) if j in chosen else 0.0 for j in range(len(s))]
    warm = common.greedy(restricted, penalty=0.0, seed=seed)
    common.verify(data, warm)
    assert set(warm["open_plants"]) <= chosen
    metrics = {"termination": "RETURNED_VERIFIED_FEE_CAPACITY_COVER",
               "parent_open_count": len(sites), "requested_open_count": len(chosen),
               "warm_open_count": len(warm["open_plants"]), "warm_objective": warm["objective_value"],
               "selected_capacity": math.fsum(float(s[j]) for j in chosen),
               "selected_fixed_cost": math.fsum(float(f[j]) for j in chosen),
               "independent": common.verify(data, warm)}
    return warm, metrics


def main():
    import argparse
    import json
    import os
    import resource
    import sys
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--prior_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=45)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    data = common.load(args.instance_path)
    with open(args.prior_path) as source:
        best = json.load(source)
    common.verify(data, best)
    metrics = {"algorithm": "fixed_S_all_arc_transport", "cold_run": False, "threads": 2}

    def emit(reason, pending=False):
        common.verify(data, best)
        metrics.update(termination=reason, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result = {key: best[key] for key in ("objective_value", "open_plants", "assignments")}
        result.update(status="Feasible", runtime=time.monotonic() - start, method_metrics=dict(metrics))
        with open(args.solution_path, "w") as out:
            json.dump(result, out, allow_nan=False)
        print(json.dumps(metrics), flush=True)
        if pending:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)
    try:
        best = refine(data, best, start + args.time_limit - .3, args.seed, metrics, emit)
        emit(metrics["termination"])
    except Exception as error:
        metrics["fallback_error"] = type(error).__name__ + ": " + str(error)[:250]
        emit("VERIFIED_POLISH_EXCEPTION_FALLBACK")


if __name__ == "__main__":
    main()
