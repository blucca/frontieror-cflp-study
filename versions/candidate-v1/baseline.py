"""Cold, full-arc CFLP MIPs: HiGHS compact/original VUB and SCIP compact.

Compact omits redundant VUB cuts for positive-demand integer solutions;
zero-demand inputs retain the original cuts. No warm starts or references.
HiGHS and SCIP are deliberately loaded in separate processes: their wheel
libraries have a real incompatible libhighs symbol when imported together.
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


def write(path, value):
    with open(path, "w") as f:
        json.dump(value, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def highs(data, deadline, seed, vub):
    import highspy
    d, s, f, c = common.arrays(data)
    n, m = c.shape
    vub = vub or bool((d == 0).any())
    nm, xc, yc = n * m, 3 if vub else 2, n + 2 if vub else 2
    xnz = nm * xc
    nnz = xnz + m * yc
    row = np.empty(nnz, dtype=np.int32)
    val = np.ones(nnz)
    row[:xnz:xc] = np.repeat(np.arange(n, dtype=np.int32), m)
    row[1:xnz:xc] = n + np.tile(np.arange(m, dtype=np.int32), n)
    val[1:xnz:xc] = np.repeat(d, m)
    if vub:
        row[2:xnz:xc] = n + m + 1 + np.arange(nm, dtype=np.int32)
    for j in range(m):
        p = xnz + j * yc
        row[p:p + 2] = [n + j, n + m]
        val[p:p + 2] = [-s[j], s[j]]
        if vub:
            row[p + 2:p + yc] = n + m + 1 + np.arange(n, dtype=np.int32) * m + j
            val[p + 2:p + yc] = -1
    starts = np.concatenate((np.arange(nm + 1, dtype=np.int32) * xc,
                             xnz + np.arange(1, m + 1, dtype=np.int32) * yc))
    nr = n + m + 1 + (nm if vub else 0)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = nm + m, nr
    lp.col_cost_ = np.concatenate((c.ravel(), f))
    lp.col_lower_, lp.col_upper_ = np.zeros(nm + m), np.ones(nm + m)
    lp.row_lower_ = np.concatenate((np.ones(n), np.full(m, -np.inf), [d.sum()],
                                   np.full(nm if vub else 0, -np.inf)))
    lp.row_upper_ = np.concatenate((np.ones(n), np.zeros(m), [np.inf],
                                   np.zeros(nm if vub else 0)))
    lp.integrality_ = ([highspy.HighsVarType.kContinuous] * nm
                       + [highspy.HighsVarType.kInteger] * m)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = starts, row, val
    h = highspy.Highs()
    for key, value in {"output_flag": False, "threads": 2, "random_seed": seed,
                       "mip_feasibility_tolerance": 1e-8}.items():
        assert h.setOptionValue(key, value) == highspy.HighsStatus.kOk
    assert h.passModel(lp) == highspy.HighsStatus.kOk
    build = time.monotonic() - START
    h.setOptionValue("time_limit", max(.001, deadline - time.monotonic() - .15))
    h.startSolve()
    done, status = h.wait(max(0, deadline - time.monotonic()))
    metrics = {"solver": "HiGHS", "version": h.version(), "model_rows": nr,
               "model_columns": nm + m, "model_nnz": nnz, "all_arcs": nm,
               "formulation": "original_vub" if vub else "compact_equivalent",
               "build_seconds": build, "native_result_available": done}
    if not done:
        metrics["status"] = "NATIVE_PENDING_NO_COLD_WITNESS"
        return None, metrics, True
    info, sol = h.getInfo(), h.getSolution()
    metrics.update(status=str(h.getModelStatus()), run_status=str(status),
                   mip_gap=float(info.mip_gap) if math.isfinite(info.mip_gap) else None,
                   bound=float(info.mip_dual_bound) if math.isfinite(info.mip_dual_bound) else None,
                   native_objective=float(info.objective_function_value) if sol.value_valid else None,
                   nodes=int(info.mip_node_count))
    if not sol.value_valid or not info.primal_solution_status == 2:
        return None, metrics, False
    x = np.asarray(sol.col_value[:nm]).reshape(n, m)
    opened = np.nonzero(np.asarray(sol.col_value[nm:]) > .5)[0]
    return common.from_flows(data, x * d[:, None], opened=opened), metrics, False


def scip(data, deadline, seed):
    from ortools.linear_solver import pywraplp
    d, s, f, c = common.arrays(data)
    n, m = c.shape
    h = pywraplp.Solver.CreateSolver("SCIP")
    assert h is not None, "existing wheel must provide SCIP"
    h.SetNumThreads(2)
    h.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {seed}\n")
    y = [h.BoolVar(f"y{j}") for j in range(m)]
    cap = [h.Constraint(-h.infinity(), 0) for _ in range(m)]
    agg = h.Constraint(float(d.sum()), h.infinity())
    for j in range(m):
        cap[j].SetCoefficient(y[j], -float(s[j]))
        agg.SetCoefficient(y[j], float(s[j]))
    objective = h.Objective()
    for j in range(m):
        objective.SetCoefficient(y[j], float(f[j]))
    x = []
    for i in range(n):
        demand = h.Constraint(1, 1)
        row = []
        for j in range(m):
            z = h.NumVar(0, 1, "")
            demand.SetCoefficient(z, 1)
            cap[j].SetCoefficient(z, float(d[i]))
            objective.SetCoefficient(z, float(c[i, j]))
            if d[i] == 0:
                vub = h.Constraint(-h.infinity(), 0)
                vub.SetCoefficient(z, 1)
                vub.SetCoefficient(y[j], -1)
            row.append(z)
        x.append(row)
    objective.SetMinimization()
    build = time.monotonic() - START
    h.SetTimeLimit(max(1, int((deadline - time.monotonic() - .15) * 1000)))
    status = h.Solve()
    metrics = {"solver": "SCIP", "version": h.SolverVersion(), "formulation": "compact_equivalent",
               "status": int(status), "model_rows": h.NumConstraints(), "model_columns": h.NumVariables(),
               "all_arcs": n * m, "build_seconds": build, "nodes": h.nodes(),
               "native_result_available": True,
               "bound": objective.BestBound() if math.isfinite(objective.BestBound()) else None}
    if status not in (h.OPTIMAL, h.FEASIBLE):
        return None, metrics, False
    values = np.asarray([[z.solution_value() for z in row] for row in x])
    opened = [j for j, z in enumerate(y) if z.solution_value() > .5]
    metrics["native_objective"] = objective.Value()
    return common.from_flows(data, values * d[:, None], opened=opened), metrics, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=float, default=55)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--solver", choices=["highs", "highs-vub", "scip"], required=True)
    a = p.parse_args()
    data = common.load(a.instance_path)
    deadline = START + a.time_limit
    try:
        if a.solver == "scip":
            sol, metrics, pending = scip(data, deadline, a.seed)
        else:
            sol, metrics, pending = highs(data, deadline, a.seed, a.solver == "highs-vub")
        if sol is not None:
            metrics["independent"] = common.verify(data, sol)
            write(a.solution_path, sol)
        metrics["whole_seconds"] = time.monotonic() - START
        metrics["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(a.solution_path + ".metrics.json", metrics)
        if pending:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(2)
        return 0 if sol is not None else 2
    except Exception as e:
        write(a.solution_path + ".metrics.json", {"status": "FAILED", "error": type(e).__name__ + ": " + str(e),
                                                    "whole_seconds": time.monotonic() - START})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
