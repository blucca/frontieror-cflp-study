"""Full continuous CFLP: dynamic slope scaling and exact transport exchanges.

Mature heuristics, not a new algorithm. Original full-demand shipping costs
are authoritative. Every fixed-set transport model retains every original arc.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    deadline = START + args.time_limit - 2.0
    data = common.load(args.instance_path)
    d, s, f, c = common.arrays(data)
    n, m = c.shape
    best = common.greedy(data, penalty=1.0, seed=args.seed)
    common.verify(data, best)
    metrics = {"algorithm": "full_q_slope_transport_exchange", "seed": args.seed,
               "threads": 1, "model_scope": "full_original", "native_runs": 0,
               "accepted_moves": 0, "first_incumbent_seconds": time.monotonic() - START}

    def remaining():
        return max(0.0, deadline - time.monotonic())

    def log(event, **values):
        print(json.dumps({"event": event, "wall": time.monotonic() - START, **values}), flush=True)

    def save(sol):
        nonlocal best
        common.verify(data, sol)
        if sol["objective_value"] < best["objective_value"] - 1e-7:
            best = sol
            log("incumbent", objective=best["objective_value"], opened=len(best["open_plants"]))

    def emit(reason, emergency=False):
        common.verify(data, best)
        metrics["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(best)
        result["runtime"] = time.monotonic() - START
        result["status"] = reason
        result["method_metrics"] = dict(metrics)
        with open(args.solution_path, "w") as out:
            json.dump(result, out, allow_nan=False)
        log("finish", reason=reason, objective=best["objective_value"])
        if emergency:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)  # A pending native daemon must never delay whole-process exit.

    log("greedy", objective=best["objective_value"], opened=len(best["open_plants"]))
    if remaining() > 5.0:
        save(common.greedy(data, penalty=0.0, seed=args.seed))
    positive = np.flatnonzero(d > 0)
    negative = set(int(j) for j in np.flatnonzero(f < 0))
    zeros = c[d == 0]
    demand = math.fsum(float(v) for v in d)

    def capacity(sites):
        return bool(sites) and math.fsum(float(s[j]) for j in sites) >= demand - 1e-8

    if not len(positive):
        q = np.zeros_like(c)
        for _ in range(20):
            if remaining() < 0.3:
                break
            sites = set(best["open_plants"])
            z = c[:, sorted(sites)].min(axis=1)
            candidates = []
            closed = np.asarray(sorted(set(range(m)) - sites), dtype=np.int32)
            if len(closed):
                delta = f[closed] + (np.minimum(z[:, None], c[:, closed]) - z[:, None]).sum(axis=0)
                k = int(closed[np.argmin(delta)])
                candidates.append(sites | {k})
            for j in sorted(sites - negative):
                if len(sites) > 1:
                    candidates.append(sites - {j})
            old = best["objective_value"]
            for trial in candidates:
                save(common.from_flows(data, q, opened=trial))
            if best["objective_value"] >= old - 1e-7:
                break
        emit("ZERO_DEMAND_FACILITY_SEARCH")
        return
    if remaining() < 6.0:
        emit("GREEDY_BUILD_BUDGET")
        return

    try:
        import highspy
        h = highspy.Highs()
        metrics["solver_version"] = h.version()
        for name, value in [("output_flag", False), ("threads", 1), ("parallel", "off"),
                            ("solver", "simplex"), ("simplex_strategy", 1),
                            ("primal_feasibility_tolerance", 1e-9),
                            ("dual_feasibility_tolerance", 1e-8), ("random_seed", args.seed)]:
            h.setOptionValue(name, value)
        dp = d[positive]
        unit = c[positive] / dp[:, None]
        rent = np.divide(np.maximum(f, 0.0), s, out=np.zeros(m), where=s > 0)
        p = len(positive)
        nv = p * m
        columns = np.arange(nv, dtype=np.int32)
        caprows = np.arange(p, p + m, dtype=np.int32)
        caplower = np.full(m, -np.inf)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nv, p + m
        lp.col_cost_ = (unit + rent[None, :]).ravel()
        lp.col_lower_, lp.col_upper_ = np.zeros(nv), np.full(nv, np.inf)
        lp.row_lower_, lp.row_upper_ = np.r_[dp, caplower], np.r_[dp, s]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.arange(nv + 1, dtype=np.int64) * 2
        indices = np.empty(2 * nv, dtype=np.int32)
        indices[0::2] = np.repeat(np.arange(p, dtype=np.int32), m)
        indices[1::2] = np.tile(caprows, p)
        lp.a_matrix_.index_, lp.a_matrix_.value_ = indices, np.ones(2 * nv)
        status = h.passModel(lp)
        if status == highspy.HighsStatus.kError:
            raise RuntimeError("transport model rejected")
        del lp, indices
        metrics.update(model_columns=nv, model_rows=p + m, model_nnz=2 * nv,
                       build_seconds=time.monotonic() - START)
        log("built", columns=nv, rows=p + m)

        def costs(penalty):
            status = h.changeColsCost(nv, columns, (unit + penalty[None, :]).ravel())
            if status == highspy.HighsStatus.kError:
                raise RuntimeError("objective update rejected")

        def bounds(sites):
            opened = np.zeros(m, dtype=bool)
            opened[list(sites)] = True
            status = h.changeRowsBounds(m, caprows, caplower, s * opened)
            if status == highspy.HighsStatus.kError:
                raise RuntimeError("capacity update rejected")

        def solve(phase, budget):
            budget = min(float(budget), remaining())
            if budget < 0.1:
                return None
            # wait is the wall authority even when a native loop ignores time_limit.
            h.setOptionValue("time_limit", h.getRunTime() + max(0.01, budget - 0.05))
            began = time.monotonic()
            h.startSolve()
            try:
                done, run_status = h.wait(max(0.0, min(budget, remaining())))
            except BaseException as error:
                metrics["pending_error"] = type(error).__name__
                emit("NATIVE_WAIT_FALLBACK", emergency=True)
            if not done:
                metrics["pending_phase"] = phase
                metrics["native_result_available"] = False
                emit("NATIVE_PENDING_FALLBACK", emergency=True)
            metrics["native_runs"] += 1
            metrics["native_result_available"] = True
            elapsed = time.monotonic() - began
            status = h.getModelStatus()
            metrics["last_model_status"] = str(status)
            log("native_returned", phase=phase, elapsed=elapsed, status=str(status), run_status=str(run_status))
            solution = h.getSolution()
            if not solution.value_valid:
                return None
            flow = np.asarray(solution.col_value, dtype=float).reshape(p, m)
            if not np.isfinite(flow).all() or flow.min() < -1e-8:
                return None
            q = np.zeros_like(c)
            q[positive] = np.maximum(flow, 0.0)
            dual = np.asarray(solution.row_dual, dtype=float)
            optimal = status == highspy.HighsModelStatus.kOptimal
            return q, dual[:p], dual[p:], optimal, elapsed

        def witness(q, sites=None):
            # Preserve every positive flow; no support or serialization cutoff.
            used = set(int(j) for j in np.flatnonzero(q.sum(axis=0) > 0.0)) | negative
            if sites is not None and used - set(sites):
                raise AssertionError("returned positive flow outside fixed facility set")
            sol = common.from_flows(data, q, opened=used if sites is None else sites)
            save(sol)
            if sites is not None and used and used != set(sites):
                compact = common.from_flows(data, q, opened=used)
                save(compact)
                if compact["objective_value"] < sol["objective_value"] - 1e-7:
                    sol = compact
            return sol

        slope_end = min(deadline - 10.0, time.monotonic() + 23.0)
        for iteration in range(3):
            budget = min(18.0 if iteration == 0 else 6.0, slope_end - time.monotonic())
            returned = solve("slope_" + str(iteration), budget)
            if returned is None:
                break
            q, pi, v, optimal, latency = returned
            try:
                witness(q)
            except AssertionError:
                break
            if not optimal or iteration == 2 or remaining() < 14.0:
                break
            loads = q.sum(axis=0)
            updated = rent.copy()
            active = loads > 1e-8
            updated[active] = np.maximum(f[active], 0.0) / loads[active]
            if np.allclose(updated, rent, rtol=1e-5, atol=1e-8):
                break
            rent = updated
            costs(rent)
            h.setOptionValue("presolve", "off")

        sites = set(best["open_plants"])
        costs(np.zeros(m))
        bounds(sites)
        returned = solve("original_cost_reflow", min(10.0, remaining()))
        if returned is None:
            emit("NO_RETURNED_ORIGINAL_REFLOW")
            return
        q, pi, v, optimal, latency = returned
        current = witness(q, sites)
        sites = set(current["open_plants"])
        bounds(sites)
        if not optimal:
            emit("FEASIBLE_ORIGINAL_REFLOW")
            return
        h.setOptionValue("presolve", "off")

        for round_id in range(4):
            if remaining() < max(2.0, latency * 1.5):
                break
            opened = np.asarray(sorted(sites), dtype=np.int32)
            closed = np.asarray(sorted(set(range(m)) - sites), dtype=np.int32)
            z = zeros[:, opened].min(axis=1) if len(zeros) else np.zeros(0)
            zd = {}
            droppable = [int(j) for j in opened if j not in negative and len(sites) > 1]
            for j in droppable:
                zd[j] = zeros[:, sorted(sites - {j})].min(axis=1) if len(zeros) else z
            drop = sorted(droppable, key=lambda j: -f[j] - s[j] * v[j] + (zd[j] - z).sum())[:6]
            rho = np.minimum(0.0, (unit[:, closed] - pi[:, None]).min(axis=0)) if len(closed) else np.zeros(0)
            az = (np.minimum(z[:, None], zeros[:, closed]) - z[:, None]).sum(axis=0) if len(zeros) else np.zeros(len(closed))
            add_score = f[closed] + s[closed] * rho + az
            add = [int(closed[k]) for k in np.argsort(add_score)[:8]]
            rho_map = dict(zip((int(k) for k in closed), rho))
            candidates = []
            for j in drop:
                lb = -f[j] - s[j] * v[j] + (zd[j] - z).sum()
                if capacity(sites - {j}):
                    candidates.append((float(lb), (j,), ()))
            for k in add:
                dz = (np.minimum(z, zeros[:, k]) - z).sum() if len(zeros) else 0.0
                candidates.append((float(f[k] + s[k] * rho_map[k] + dz), (), (k,)))
            for j in drop:
                for k in add:
                    trial = sites - {j} | {k}
                    if capacity(trial):
                        dz = (np.minimum(zd[j], zeros[:, k]) - z).sum() if len(zeros) else 0.0
                        lb = f[k] - f[j] + s[k] * rho_map[k] - s[j] * v[j] + dz
                        candidates.append((float(lb), (j,), (k,)))
            improved = False
            for lb, remove, insert in sorted(candidates)[:12]:
                if lb >= -1e-6 or remaining() < max(2.0, latency * 1.5):
                    break
                trial = (sites - set(remove)) | set(insert)
                bounds(trial)
                returned = solve("facility_move", min(4.0, remaining()))
                if returned is None:
                    bounds(sites)
                    continue
                q2, pi2, v2, optimal2, latency = returned
                try:
                    candidate = witness(q2, trial)
                except AssertionError:
                    bounds(sites)
                    continue
                if candidate["objective_value"] < current["objective_value"] - 1e-7 and optimal2:
                    current, sites, pi, v = candidate, set(candidate["open_plants"]), pi2, v2
                    bounds(sites)
                    metrics["accepted_moves"] += 1
                    improved = True
                    break
                bounds(sites)
            if not improved:
                break
        emit("VERIFIED_BEST")
    except Exception as error:
        metrics["fallback_error"] = type(error).__name__ + ": " + str(error)[:250]
        emit("VERIFIED_EXCEPTION_FALLBACK")


if __name__ == "__main__":
    main()
