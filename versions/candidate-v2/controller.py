"""Task-local serial whole-process runs, two independent full checkers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
REV = "ef42a2eac13595453f986d9df0b62e86e75ab596"
URL = f"https://huggingface.co/datasets/frontieror/FrontierOR/resolve/{REV}/tasks/klose2005/instance/"
CASES = {"tiny": ("tiny_instance.json", 100, 100),
         **{f"large{i}": (f"large_instance_{i}.json", 1500, 750) for i in range(1, 6)}}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bounded():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def source(cases):
    import common
    target = ROOT / "input"
    target.mkdir(exist_ok=True)
    receipts = []
    for name in cases:
        file, n, m = CASES[name]
        p = target / file
        if not p.exists():
            req = urllib.request.Request(URL + file + "?download=true", headers={"User-Agent": "Blucca-CFLP-study/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response, p.open("wb") as f:
                assert response.status == 200, "full source, not metadata range"
                while chunk := response.read(512 * 1024):
                    f.write(chunk)
                    assert f.tell() <= 24 * 1024**2, "one original source size cap"
            assert p.stat().st_size > 10000
        data = common.load(p)
        assert data["num_customers"] == n and data["num_plants"] == m
        d, s, fixed, cost = common.arrays(data)
        receipts.append({"case": name, "source_url": URL + file, "revision": REV,
                         "full_input_bytes": p.stat().st_size, "customers": n, "plants": m,
                         "shipping_arcs": n * m, "demand": float(d.sum()), "capacity": float(s.sum()),
                         "fixed_min_max": [float(fixed.min()), float(fixed.max())],
                         "shipping_min_max": [float(cost.min()), float(cost.max())],
                         "demand_min_max": [float(d.min()), float(d.max())],
                         "capacity_min_max": [float(s.min()), float(s.max())]})
    dump(ROOT / ("source-" + "-".join(cases) + ".json"), receipts)


def run_one(case, algorithm, seed, out):
    import common
    instance = ROOT / "input" / CASES[case][0]
    out.mkdir(parents=True, exist_ok=False)
    solution = out / "solution.json"
    entry = {"method": "method.py", "pool": "pool.py"}.get(algorithm, "baseline.py")
    command = [sys.executable, "-B", str(ROOT / entry),
               "--instance_path", str(instance), "--solution_path", str(solution),
               "--time_limit", "55", "--seed", str(seed)]
    if algorithm not in ("method", "pool"):
        command += ["--solver", "highs" if algorithm == "highs-off" else algorithm]
        if algorithm == "highs-off":
            command += ["--no_presolve"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(ROOT))
    begin = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"case": case, "algorithm": algorithm, "seed": seed, "status": "STARTED",
              "resource_limits": {"whole_algorithm_seconds": 60, "address_space_bytes": 4 * 1024**3, "cpus": 2}}
    with (out / "algorithm.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                preexec_fn=bounded, start_new_session=True)
        try:
            code = proc.wait(timeout=60)
            result["algorithm_exit"] = code
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = -signal.SIGKILL
            result["algorithm_exit"], result["failure"] = code, "WHOLE_60S_TIMEOUT"
    result["algorithm_whole_seconds"] = time.monotonic() - begin
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["cpu_seconds"] = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    metrics = Path(str(solution) + ".metrics.json")
    if metrics.exists():
        result["metrics"] = json.loads(metrics.read_text())
    if code == 0 and solution.exists():
        try:
            data, sol = common.load(instance), json.loads(solution.read_text())
            if "method_metrics" in sol:
                result["metrics"] = sol["method_metrics"]
                result["method_status"] = sol.get("status")
            result["independent"] = common.verify(data, sol)
            spec = importlib.util.spec_from_file_location("official_checker", ROOT / "feasibility_check.py")
            checker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checker)
            result["official"] = checker.check_feasibility(data, sol)
            assert result["official"]["feasible"]
            result["status"] = "PASS"
        except Exception as e:
            result["status"], result["failure"] = "FAIL", type(e).__name__ + ": " + str(e)
    else:
        result["status"] = "NO_WHOLE_WITNESS"
    result["whole_with_checkers_seconds"] = time.monotonic() - begin
    dump(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("case", "algorithm", "status", "algorithm_whole_seconds")}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="tiny,large4")
    p.add_argument("--algorithms", default="highs,highs-vub,scip")
    p.add_argument("--source_only", action="store_true")
    p.add_argument("--batch", default="first")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    cases, algorithms = a.cases.split(","), a.algorithms.split(",")
    assert all(x in CASES for x in cases) and all(x in ("highs", "highs-vub", "highs-off", "scip", "method", "pool") for x in algorithms)
    source(cases)
    if a.source_only:
        return
    out = ROOT / a.batch
    out.mkdir(exist_ok=False)
    state = {"status": "RUNNING", "pid": os.getpid(), "cases": cases, "algorithms": algorithms, "results": []}
    dump(out / "state.json", state)
    for case in cases:
        for algorithm in algorithms:
            r = run_one(case, algorithm, a.seed, out / f"{case}-{algorithm}-s{a.seed}")
            state["results"].append(r)
            dump(out / "state.json", state)
            if r["status"] == "FAIL" or algorithm in ("method", "pool") and r["status"] != "PASS":
                state["status"] = "STOP_VALIDATION_FAILURE"
                dump(out / "state.json", state)
                return
    state["status"] = "DONE"
    dump(out / "state.json", state)


if __name__ == "__main__":
    main()
