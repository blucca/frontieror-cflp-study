"""Conditional CFLP gate: VUB-demand-lifted cuts and bounded joint masters.

Mature Benders construction. This stored-prior gate makes no cold-speed claim.
All LPs retain every arc to the chosen S; final status stays Feasible.
"""
import time
_WHOLE_BEGAN = time.monotonic()

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys

import numpy as np
import common
import polish


def make_cut(data, alpha):
    d, s, f, c = common.arrays(data)
    positive = np.flatnonzero(d > 0)
    alpha = np.asarray(alpha, dtype=float)
    assert alpha.shape == (len(positive),) and np.isfinite(alpha).all()
    unit = c[positive] / d[positive, None]
    beta = alpha[:, None] - unit
    assert np.isfinite(beta).all()
    demands, p, m = d[positive], len(positive), len(s)
    mu = np.maximum(0.0, beta.max(axis=0)) if p else np.zeros(m)
    prices, values, fills = np.zeros(m), np.zeros(m), np.zeros(m)
    demand_sum = math.fsum(float(v) for v in demands)
    violation, overfill = 0.0, 0.0
    for j in range(m):
        if not p:
            continue
        order = np.argsort(-beta[:, j], kind="stable")
        b, weights = beta[order, j], demands[order]
        cumulative = np.cumsum(weights)
        if s[j] == 0:
            price = mu[j]
        elif s[j] >= demand_sum:
            price = 0.0
        else:
            marginal = min(int(np.searchsorted(cumulative, s[j], side="left")), p - 1)
            price = max(0.0, float(b[marginal]))
        prices[j] = price
        nu = np.maximum(0.0, beta[:, j] - price)
        values[j] = math.fsum([float(s[j] * price),
                              *[float(v) for v in demands * nu]])
        # Independent feasible greedy primal, separate from the price formula.
        before = np.r_[0.0, cumulative[:-1]]
        take = np.minimum(weights, np.maximum(0.0, s[j] - before))
        take[b <= 0.0] = 0.0
        assert ((take >= 0.0) & (take <= weights)).all()
        load = math.fsum(float(v) for v in take)
        overfill = max(overfill, load - float(s[j]))
        assert load <= s[j] + max(1e-6, abs(s[j]) * 1e-10)
        fills[j] = math.fsum(float(v) for v in b * take)
        violation = max(violation, float((beta[:, j] - price - nu).max()))
    legacy_values = s * mu
    assert np.isfinite(values).all() and np.isfinite(legacy_values).all()
    equality_tol = np.maximum(1e-6, np.maximum(abs(values), abs(fills)) * 1e-10)
    dominance_tol = np.maximum(1e-6, abs(legacy_values) * 1e-10)
    assert (abs(values - fills) <= equality_tol).all(), "facility knapsack dual/primal equality"
    assert (values <= legacy_values + dominance_tol).all(), "VUB lifting dominates legacy s*mu"
    assert violation <= 1e-9 and (prices >= 0).all() and (values >= 0).all()
    legacy_violation = float((beta - mu).max()) if p else 0.0
    assert legacy_violation <= 1e-9
    zero_floor = math.fsum(float(v) for v in c[d == 0].min(axis=1)) if (d == 0).any() else 0.0
    intercept = math.fsum(float(v) for v in d[positive] * alpha) + zero_floor
    return {"intercept": intercept, "coeff": -values, "mu": mu,
            "knapsack_prices": prices, "knapsack_dual_values": values,
            "knapsack_primal_values": fills, "legacy_coeff": -legacy_values,
            "facility_lift": legacy_values - values,
            "support_value_gap": float(abs(values - fills).max()),
            "dominance_max_excess": max(0.0, float((values - legacy_values).max())),
            "max_greedy_capacity_excess": overfill, "legacy_arc_violation": legacy_violation,
            "zero_floor": zero_floor, "all_arc_violation": violation,
            "all_arc_count": p * m,
            "shipping_floor": math.fsum(float(v) for v in c.min(axis=1))}


def warm_for_sites(data, sites, seed):
    d, s, f, c = common.arrays(data)
    chosen = set(sites)
    assert chosen and set(np.flatnonzero(f < 0)) <= chosen
    assert math.fsum(float(s[j]) for j in chosen) >= math.fsum(float(v) for v in d)
    if not (d > 0).any():
        return common.from_flows(data, np.zeros_like(c), opened=sites)
    restricted = dict(data)
    restricted["plants"] = dict(data["plants"])
    restricted["plants"]["capacities"] = [float(s[j]) if j in chosen else 0.0 for j in range(len(s))]
    warm = common.greedy(restricted, penalty=0.0, seed=seed)
    assert set(warm["open_plants"]) <= chosen
    warm["open_plants"] = sorted(chosen)
    for i in np.flatnonzero(d == 0):
        j = min(chosen, key=lambda j: (c[i, j], j))
        warm["assignments"][str(int(i))] = {str(int(j)): 1.0}
    warm["objective_value"] = math.fsum(float(f[j]) for j in chosen) + math.fsum(
        float(c[int(i), int(j)]) * x for i, row in warm["assignments"].items() for j, x in row.items())
    common.verify(data, warm)
    return warm


def master(data, cuts, prior, deadline, seed, metrics, pending_emit):
    import highspy
    began = time.monotonic()
    d, s, f, c = common.arrays(data)
    m, k = len(s), len(cuts)
    shipping_floor = math.fsum(float(v) for v in c.min(axis=1))
    # capacity + nonempty set + accumulated shipping epigraph cuts.
    column_rows = np.tile(np.arange(k + 2, dtype=np.int32), m)
    column_vals = np.column_stack((s, np.ones(m), *[-cut["coeff"] for cut in cuts])).ravel()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m + 1, k + 2
    lp.col_cost_ = np.r_[f, 1.0]
    lp.col_lower_ = np.r_[(f < 0).astype(float), shipping_floor]
    lp.col_upper_ = np.r_[np.ones(m), np.inf]
    lp.integrality_ = [highspy.HighsVarType.kInteger] * m + [highspy.HighsVarType.kContinuous]
    lp.row_lower_ = np.r_[math.fsum(float(v) for v in d), 1.0, [cut["intercept"] for cut in cuts]]
    lp.row_upper_ = np.full(k + 2, np.inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.r_[np.arange(m + 1, dtype=np.int32) * (k + 2), m * (k + 2) + k]
    lp.a_matrix_.index_ = np.r_[column_rows, np.arange(2, k + 2, dtype=np.int32)]
    lp.a_matrix_.value_ = np.r_[column_vals, np.ones(k)]
    warm = np.zeros(m + 1)
    warm[prior["open_plants"]] = 1.0
    warm[-1] = prior["objective_value"] - math.fsum(float(f[j]) for j in prior["open_plants"])
    lhs = np.bincount(lp.a_matrix_.index_, weights=lp.a_matrix_.value_ * np.repeat(warm, np.diff(lp.a_matrix_.start_)), minlength=k + 2)
    assert (lhs >= np.asarray(lp.row_lower_) - 1e-6).all() and warm[-1] >= shipping_floor - 1e-6
    h = highspy.Highs()
    for key, value in {"output_flag": False, "threads": 2, "random_seed": seed,
                       "mip_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9}.items():
        assert h.setOptionValue(key, value) == highspy.HighsStatus.kOk
    # A saved third cut contains 1.47e-14: HiGHS loads it with kWarning
    # after its documented small-matrix filtering. This master only selects S;
    # every returned S still needs complete recourse and both primal checkers.
    model_load_status = h.passModel(lp)
    assert model_load_status in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning), str(model_load_status)
    assert h.setSolution(m + 1, np.arange(m + 1, dtype=np.int32), warm) == highspy.HighsStatus.kOk
    metrics.update(cuts=k, binary_facilities=m, model_load_status=str(model_load_status),
                   build_seconds=time.monotonic() - began,
                   shipping_floor=shipping_floor, warm_objective=prior["objective_value"])
    budget = deadline - time.monotonic()
    if budget < .1:
        metrics["termination"] = "MASTER_NO_BUILD_BUDGET"
        return None
    assert h.setOptionValue("time_limit", max(.001, budget - .1)) == highspy.HighsStatus.kOk
    h.startSolve()
    done, run_status = h.wait(max(0.0, deadline - time.monotonic()))
    if not done:
        metrics["termination"] = "MASTER_PENDING_FALLBACK"
        pending_emit("MASTER_PENDING_FALLBACK", pending=True)
        raise AssertionError("pending emitter exits")
    native = h.getSolution()
    metrics.update(native_status=str(h.getModelStatus()), run_status=str(run_status),
                   elapsed_seconds=time.monotonic() - began, value_valid=bool(native.value_valid))
    if not native.value_valid:
        metrics["termination"] = "MASTER_NO_RETURNED_BINARY"
        return None
    vector = np.asarray(native.col_value, dtype=float)
    assert np.isfinite(vector).all() and np.max(abs(vector[:m] - np.rint(vector[:m]))) <= 1e-7
    sites = [int(j) for j in np.flatnonzero(vector[:m] > .5)]
    assert sites and set(np.flatnonzero(f < 0)) <= set(sites)
    assert math.fsum(float(s[j]) for j in sites) >= math.fsum(float(v) for v in d)
    lower_estimate = float(np.dot(f, np.rint(vector[:m])) + vector[-1])
    metrics.update(termination="RETURNED_JOINT_MASTER", sites=sites, selected_open_count=len(sites),
                   master_incumbent_objective=lower_estimate, epigraph=float(vector[-1]))
    h.clear()
    return sites


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    began = _WHOLE_BEGAN
    deadline = began + 55.0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance_path", required=True)
    p.add_argument("--prior_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    data = common.load(args.instance_path)
    prior = json.loads(Path(args.prior_path).read_text())
    common.verify(data, prior)
    best = prior
    metrics = {"method": "vub_demand_lift_binary_joint_master", "cold_run": False,
               "start_objective": prior["objective_value"], "threads": 2,
               "whole_budget_seconds": 55.0, "initial_oracle_budget_seconds": 10.0,
               "max_different_sets": 8, "cuts": [], "trials": []}
    extra_start = None

    def log(event, **values):
        print(json.dumps({"event": event, "wall": time.monotonic() - began, **values}, allow_nan=False), flush=True)

    def emit(reason, pending=False):
        common.verify(data, best)
        metrics.update(termination=reason, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       whole_seconds=time.monotonic() - began, best_objective=best["objective_value"],
                       extra_seconds=time.monotonic() - extra_start if extra_start is not None else 0.0,
                       improvement_fraction=(prior["objective_value"] - best["objective_value"]) / abs(prior["objective_value"]))
        result = dict(best, status="Feasible", runtime=time.monotonic() - began)
        dump(out / "solution.json", result)
        dump(out / "metrics.json", metrics)
        log("checkpoint", termination=reason, objective=best["objective_value"])
        if pending:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)

    def capture(dual, sol, phase):
        assert dual["valid"] and dual["optimal"], "supporting cut uses returned Optimal LP dual"
        cut = make_cut(data, dual["alpha"])
        y = np.zeros(data["num_plants"])
        y[dual["sites"]] = 1.0
        positive_bound = cut["intercept"] - cut["zero_floor"] + float(np.dot(cut["coeff"], y))
        error = dual["positive_primal"] - positive_bound
        assert abs(error) <= max(1e-5, abs(dual["positive_primal"]) * 1e-8), "current S positive primal/dual support"
        fixed = math.fsum(float(data["plants"]["fixed_costs"][j]) for j in sol["open_plants"])
        shipping = sol["objective_value"] - fixed
        record = {"phase": phase, "all_source_arcs_checked": cut["all_arc_count"],
                  "all_arc_max_violation": cut["all_arc_violation"], "positive_support_error": error,
                  "zero_global_floor": cut["zero_floor"],
                  "zero_floor_slack": shipping - dual["positive_primal"] - cut["zero_floor"],
                  "current_shipping": shipping, "current_cut_shipping": positive_bound + cut["zero_floor"],
                  "alpha": dual["alpha"].tolist(), "intercept": cut["intercept"],
                  "coeff": cut["coeff"].tolist(), "mu": cut["mu"].tolist(),
                  "knapsack_prices": cut["knapsack_prices"].tolist(),
                  "knapsack_dual_values": cut["knapsack_dual_values"].tolist(),
                  "knapsack_primal_values": cut["knapsack_primal_values"].tolist(),
                  "legacy_coeff": cut["legacy_coeff"].tolist(),
                  "support_value_gap": cut["support_value_gap"],
                  "dominance_max_excess": cut["dominance_max_excess"],
                  "max_greedy_capacity_excess": cut["max_greedy_capacity_excess"]}
        metrics["cuts"].append(record)
        log("cut_verified", phase=phase, support_error=error, arcs=cut["all_arc_count"])
        return cut

    emit("VERIFIED_STORED_PRIOR")
    try:
        initial_metrics = metrics["initial_oracle"] = {}
        dual = {}
        initial = polish.refine(data, prior, min(deadline - 2.0, time.monotonic() + 10.0), args.seed, initial_metrics, emit, dual_out=dual)
        if not dual.get("valid") or not dual.get("optimal"):
            emit("STOP_INITIAL_ORACLE_NO_OPTIMAL_DUAL")
            return
        initial_metrics["independent_returned"] = common.verify(data, initial)
        import feasibility_check
        initial_metrics["official"] = feasibility_check.check_feasibility(data, initial)
        assert initial_metrics["official"]["feasible"]
        if initial["objective_value"] < best["objective_value"] - 1e-7:
            best = initial
        # Import/JSON/initial LP/cuts/master/warm/recourse/checks/output share 55s.
        extra_start = time.monotonic()
        cuts = [capture(dual, initial, "initial")]
        seen = {tuple(prior["open_plants"])}
        for iteration in range(8):
            if len(seen) >= 8 or deadline - time.monotonic() < 3.0:
                break
            trial = {"iteration": iteration + 1}
            metrics["trials"].append(trial)
            master_metrics = trial["master"] = {}
            sites = master(data, cuts, best, min(deadline - 2.0, time.monotonic() + 2.0), args.seed, master_metrics, emit)
            if sites is None:
                break
            if tuple(sites) in seen:
                trial["termination"] = "REPEATED_MASTER_S"
                break
            seen.add(tuple(sites))
            trial["closed_from_start"] = len(set(prior["open_plants"]) - set(sites))
            trial["opened_from_start"] = len(set(sites) - set(prior["open_plants"]))
            warm_start = time.monotonic()
            warm = warm_for_sites(data, sites, args.seed)
            trial.update(warm_seconds=time.monotonic() - warm_start, warm_objective=warm["objective_value"])
            oracle_metrics = trial["oracle"] = {}
            trial_dual = {}
            candidate = polish.refine(data, warm, deadline - 2.0, args.seed, oracle_metrics, emit, dual_out=trial_dual)
            common.verify(data, candidate)
            import feasibility_check
            checked = feasibility_check.check_feasibility(data, candidate)
            assert checked["feasible"]
            trial.update(objective=candidate["objective_value"], official=checked,
                         independent=common.verify(data, candidate),
                         termination="RETURNED_FULL_JOINT_WITNESS" if trial_dual.get("valid") and trial_dual.get("optimal") else "NO_COMPLETED_JOINT_ORACLE",
                         improvement_fraction=(prior["objective_value"] - candidate["objective_value"]) / abs(prior["objective_value"]))
            dump(out / f"candidate-{iteration + 1}.json", candidate)
            if candidate["objective_value"] < best["objective_value"] - 1e-7:
                best = candidate
            log("joint_witness", iteration=iteration + 1, objective=candidate["objective_value"], open_count=len(sites))
            emit("RETURNED_JOINT_CHECKPOINT")
            if trial_dual.get("valid") and trial_dual.get("optimal"):
                cuts.append(capture(trial_dual, candidate, f"joint_{iteration + 1}"))
            else:
                break
        emit("JOINT_GATE_COMPLETE")
    except Exception as error:
        metrics["failure"] = type(error).__name__ + ": " + str(error)[:400]
        emit("STOP_JOINT_GATE_EXCEPTION")


if __name__ == "__main__":
    main()
