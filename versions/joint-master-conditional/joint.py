"""Conditional CFLP gate: all-facility dual cuts and two joint master choices.

Mature Benders construction. This stored-prior gate makes no cold-speed claim.
All LPs retain every arc to the chosen S; final status stays Feasible.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import common
import polish


def make_cut(data, alpha):
    d, s, f, c = common.arrays(data)
    positive = np.flatnonzero(d > 0)
    alpha = np.asarray(alpha, dtype=float)
    assert alpha.shape == (len(positive),) and np.isfinite(alpha).all()
    unit = c[positive] / d[positive, None]
    mu = np.maximum(0.0, (alpha[:, None] - unit).max(axis=0)) if len(positive) else np.zeros(len(s))
    violation = float((alpha[:, None] - mu - unit).max()) if len(positive) else 0.0
    assert violation <= 1e-9 and (mu >= 0).all()
    zero_floor = math.fsum(float(v) for v in c[d == 0].min(axis=1)) if (d == 0).any() else 0.0
    intercept = math.fsum(float(v) for v in d[positive] * alpha) + zero_floor
    return {"intercept": intercept, "coeff": -s * mu, "mu": mu,
            "zero_floor": zero_floor, "all_arc_violation": violation,
            "all_arc_count": len(positive) * len(s),
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
    assert h.passModel(lp) == highspy.HighsStatus.kOk
    assert h.setSolution(m + 1, np.arange(m + 1, dtype=np.int32), warm) == highspy.HighsStatus.kOk
    metrics.update(cuts=k, binary_facilities=m, build_seconds=time.monotonic() - began,
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
    began = time.monotonic()
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
    metrics = {"method": "all_facility_cut_binary_joint_master", "cold_run": False,
               "start_objective": prior["objective_value"], "threads": 2,
               "gate_extra_budget_seconds": 20.0, "max_joint_sets": 2, "cuts": [], "trials": []}
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
                  "coeff": cut["coeff"].tolist(), "mu": cut["mu"].tolist()}
        metrics["cuts"].append(record)
        log("cut_verified", phase=phase, support_error=error, arcs=cut["all_arc_count"])
        return cut

    emit("VERIFIED_STORED_PRIOR")
    try:
        initial_metrics = metrics["initial_oracle"] = {}
        dual = {}
        initial = polish.refine(data, prior, time.monotonic() + 10.0, args.seed, initial_metrics, emit, dual_out=dual)
        if not dual:
            emit("STOP_INITIAL_ORACLE_NO_DUAL")
            return
        if initial["objective_value"] < best["objective_value"] - 1e-7:
            best = initial
        # Initial fixed-S solve is separately measured; cut/build/master/warm/
        # full recourse/checkpoint all count against the 20-second extra path.
        extra_start = time.monotonic()
        deadline = extra_start + 20.0
        cuts = [capture(dual, initial, "initial")]
        seen = {tuple(prior["open_plants"])}
        for iteration in range(2):
            if deadline - time.monotonic() < 1.0:
                break
            trial = {"iteration": iteration + 1}
            metrics["trials"].append(trial)
            master_metrics = trial["master"] = {}
            sites = master(data, cuts, best, min(deadline - .5, time.monotonic() + 2.0), args.seed, master_metrics, emit)
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
            candidate = polish.refine(data, warm, deadline - .3, args.seed, oracle_metrics, emit, dual_out=trial_dual)
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
            if trial_dual and iteration == 0:
                cuts.append(capture(trial_dual, candidate, "joint_1"))
            else:
                break
        emit("JOINT_GATE_COMPLETE")
    except Exception as error:
        metrics["failure"] = type(error).__name__ + ": " + str(error)[:400]
        emit("STOP_JOINT_GATE_EXCEPTION")


if __name__ == "__main__":
    main()
