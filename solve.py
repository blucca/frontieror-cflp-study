"""Cold CFLP: bounded arc pool, full-S transport, fee/capacity cover.

Pool MIP and each fixed-set LP have separate native optimization scopes.
The final complete original witness is feasible; no global bound is exported.
"""
import time
START = time.monotonic()
import argparse
import json
import math
import os
import resource
import sys
import numpy as np
import common


def assemble(data, fallback):
    d, s, f, c = common.arrays(data)
    n, m = c.shape
    k = min(16, m)
    unit = np.divide(c, d[:, None], out=c.copy(), where=d[:, None] > 0)
    rent = np.divide(np.maximum(f, 0), s, out=np.zeros(m), where=s > 0)
    score = unit + rent[None, :]
    score[np.ix_(d > 0, s == 0)] = np.inf
    near = np.argsort(score, axis=1, kind="stable")[:, :k]
    warmkeys, warmvals = [], []
    for i, row in fallback["assignments"].items():
        for j, x in row.items():
            if x > 0:  # Includes zero-demand fraction rows, not just q>0.
                warmkeys.append(int(i) * m + int(j))
                warmvals.append(x)
    keys = np.unique(np.r_[(np.arange(n)[:, None] * m + near).ravel(), warmkeys]).astype(np.int64)
    ii, jj = (keys // m).astype(np.int32), (keys % m).astype(np.int32)
    arcs, nr = len(keys), n + m + 1 + len(keys)
    degree = np.bincount(jj, minlength=m)
    starts = np.r_[np.arange(arcs + 1) * 3, 3 * arcs + np.cumsum(degree + 2)].astype(np.int32)
    rows, values = np.empty(4 * arcs + 2 * m, dtype=np.int32), np.ones(4 * arcs + 2 * m)
    rows[:3 * arcs:3], rows[1:3 * arcs:3] = ii, n + jj
    rows[2:3 * arcs:3], values[1:3 * arcs:3] = n + m + 1 + np.arange(arcs), d[ii]
    grouped, offsets = np.argsort(jj, kind="stable"), np.r_[0, np.cumsum(degree)]
    for j in range(m):
        p = int(starts[arcs + j])
        incident = grouped[offsets[j]:offsets[j + 1]]
        rows[p:p + 2], values[p:p + 2] = [n + j, n + m], [-s[j], s[j]]
        rows[p + 2:p + 2 + len(incident)] = n + m + 1 + incident
        values[p + 2:p + 2 + len(incident)] = -1
    lower = np.r_[np.ones(n), np.full(m, -np.inf), math.fsum(d), np.full(arcs, -np.inf)]
    upper = np.r_[np.ones(n), np.zeros(m), np.inf, np.zeros(arcs)]
    costs = np.r_[c[ii, jj], f]
    warm = np.zeros(arcs + m)
    positions = np.searchsorted(keys, warmkeys)
    assert np.array_equal(keys[positions], warmkeys), "all cold greedy arcs retained"
    warm[positions], warm[arcs + np.asarray(fallback["open_plants"], dtype=np.int32)] = warmvals, 1
    # Independent CSC coverage/index mapping and all-row warm-feasibility check.
    assert len(starts) == arcs + m + 1 and starts[0] == 0 and starts[-1] == len(rows)
    assert (np.bincount(ii, minlength=n) >= k).all() and (rows >= 0).all() and (rows < nr).all()
    lhs = np.bincount(rows, weights=values * np.repeat(warm, np.diff(starts)), minlength=nr)
    assert (lhs >= lower - 1e-7).all() and (lhs <= upper + 1e-7).all(), "every original row in CSC"
    assert abs(math.fsum(costs * warm) - fallback["objective_value"]) <= max(1e-6, abs(fallback["objective_value"]) * 1e-10)
    return ii, jj, starts, rows, values, costs, lower, upper, warm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=55)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    deadline = START + args.time_limit - 2.0
    data = common.load(args.instance_path)
    best = common.greedy(data, penalty=1.0, seed=args.seed)
    common.verify(data, best)
    metrics = {"algorithm": "fee_capacity_cover_full_S_transport", "threads": 2, "seed": args.seed,
               "certificate_scope": "full_original_feasibility_only",
               "native_optimization_scope": "arc_pool_subproblem", "native_result_available": False,
               "first_incumbent_seconds": time.monotonic() - START}

    def emit(termination, pending=False):
        common.verify(data, best)
        metrics["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(best)
        metrics["termination"] = termination
        result.update(status="Feasible", runtime=time.monotonic() - START, method_metrics=dict(metrics))
        with open(args.solution_path, "w") as out:
            json.dump(result, out, allow_nan=False)
        print(json.dumps(metrics), flush=True)
        if pending:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)

    try:
        d, s, f, c = common.arrays(data)
        n, m = c.shape
        ii, jj, starts, rows, values, costs, lower, upper, warm = assemble(data, best)
        arcs, nc = len(ii), len(costs)
        import highspy
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nc, len(lower)
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = costs, np.zeros(nc), np.ones(nc)
        lp.row_lower_, lp.row_upper_ = lower, upper
        lp.integrality_ = [highspy.HighsVarType.kContinuous] * arcs + [highspy.HighsVarType.kInteger] * m
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = starts, rows, values
        h = highspy.Highs()
        for key, value in {"output_flag": False, "threads": 2, "random_seed": args.seed,
                           "mip_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9}.items():
            assert h.setOptionValue(key, value) == highspy.HighsStatus.kOk
        assert h.passModel(lp) == highspy.HighsStatus.kOk
        assert h.setSolution(nc, np.arange(nc, dtype=np.int32), warm) == highspy.HighsStatus.kOk
        metrics.update(solver_version=h.version(), all_source_arcs=n * m, pool_arcs=arcs,
                       model_rows=len(lower), model_columns=nc, model_nnz=len(rows),
                       build_seconds=time.monotonic() - START, warm_objective=best["objective_value"])
        # The full-budget pool consumed all 53 seconds on L1/L2/L4, leaving
        # no transport recourse. Reserve most of the same whole budget for it.
        pool_deadline = min(deadline, START + 20.0)
        metrics["pool_phase_deadline_seconds"] = pool_deadline - START
        budget = max(0.0, pool_deadline - time.monotonic())
        if budget < 0.1:
            emit("BUILD_BUDGET_FALLBACK")
            return
        h.setOptionValue("time_limit", max(.001, budget - .1))
        h.startSolve()
        try:
            done, run_status = h.wait(max(0.0, pool_deadline - time.monotonic()))
        except BaseException as error:
            metrics["wait_error"] = type(error).__name__
            emit("PENDING_WAIT_FALLBACK", pending=True)
        if not done:
            emit("NATIVE_PENDING_FALLBACK", pending=True)
        metrics.update(native_result_available=True, pool_model_status=str(h.getModelStatus()),
                       run_status=str(run_status))  # No pool bound or gap is exported.
        native = h.getSolution()
        if native.value_valid:
            vector = np.asarray(native.col_value, dtype=float)
            assert vector.shape == (nc,) and np.isfinite(vector).all()
            x, y = vector[:arcs], vector[arcs:]
            assert x.min() >= -1e-9 and x.max() <= 1 + 1e-9
            assert np.max(np.abs(y - np.rint(y))) <= 1e-7
            opened = sorted(set(int(j) for j in np.flatnonzero(y > .5)) | set(int(j) for j in np.flatnonzero(f < 0)))
            assignments = {str(i): {} for i in range(n)}
            for i, j, val in zip(ii, jj, x):
                if val > 0:
                    assignments[str(int(i))][str(int(j))] = float(val)
            objective = math.fsum(c[int(i), int(j)] * float(val) for i, j, val in zip(ii, jj, x) if val > 0)
            objective += math.fsum(float(f[j]) for j in opened)
            candidate = {"objective_value": objective, "open_plants": opened, "assignments": assignments}
            metrics["independent"] = common.verify(data, candidate)
            if candidate["objective_value"] < best["objective_value"] - 1e-7:
                best = candidate
        metrics["pool_stage_seconds"] = time.monotonic() - START
        if deadline - time.monotonic() >= .5:
            import polish
            polish_metrics = metrics["polish"] = {}
            h.clear()  # Completed pool only: release native model before full-S LP.
            best = polish.refine(data, best, deadline, args.seed, polish_metrics, emit)
            if deadline - time.monotonic() >= 1.0:
                cover, cover_metrics = polish.capacity_cover(data, best, args.seed)
                metrics["capacity_cover"] = cover_metrics
                if cover["objective_value"] < best["objective_value"] - 1e-7:
                    best = cover
                if cover_metrics["termination"] == "RETURNED_VERIFIED_FEE_CAPACITY_COVER":
                    cover_lp_metrics = metrics["cover_transport"] = {}
                    candidate = polish.refine(data, cover, deadline, args.seed, cover_lp_metrics, emit)
                    if candidate["objective_value"] < best["objective_value"] - 1e-7:
                        best = candidate
        emit("RETURNED_VERIFIED_BEST")
    except Exception as error:
        metrics["fallback_error"] = type(error).__name__ + ": " + str(error)[:250]
        emit("VERIFIED_EXCEPTION_FALLBACK")


if __name__ == "__main__":
    main()
