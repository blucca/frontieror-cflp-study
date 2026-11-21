# FrontierOR CFLP: capacity, then facility choices

A runnable, cold-input study of splittable capacitated facility location. The
selected heuristic emits a fully checked solution for all six pinned benchmark
cases under the recorded 2-CPU / 4-GiB-address-space / whole-60-second envelope.
The observed costs and failures below are the result, including weak cases.

## Install and solve

Tested on Linux, Python 3.13.5. Native packages are pinned to the measured versions.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python solve.py --instance_path fixtures/splittable-zero.json \
  --solution_path solution.json --time_limit 55 --seed 1
```

Supply an original instance JSON through the same `--instance_path` interface.
[Input fields](source/instance_schema.json) and [output fields](source/solution_schema.json)
are upstream copies. `open_plants` is the binary facility set; `assignments` is a
customer-to-facility map of **fractions**, with omitted arcs equal to zero.
`objective_value` is recomputed from the actual decision witness.

The original objective is `sum(c_ij*x_ij) + sum(f_j*y_j)`: c already prices the
customer's **entire** demand. Capacity uses `sum(d_i*x_ij) <= s_j*y_j` with the
original floating-point s. Fractions are continuous and may split; demand rows,
capacity, bounds, binary facilities, VUB `x_ij<=y_j`, aggregate capacity, and true
cost are all checked. Zero-demand customers still receive fractions summing to
one and their supplied c values still enter the objective. Negative fixed-cost
facilities retain the same opening semantics as the shared validator.

## Reproduce the whole-process experiment

```sh
# Original six inputs: about 86.6 MB, pinned revision plus recorded SHA-256.
python replay.py --source_only
# Creates results/run-001/{case}-{algorithm}-s1/ with JSON and algorithm logs.
python replay.py --batch run-001 --algorithms solve,highs,scip
# Optional preserved full-transport comparison:
python replay.py --batch transport-001 --algorithms transport
```

`replay.py` downloads into git-ignored `input/`, validates source hashes, serially
launches separate solver processes, restricts CPU affinity to two CPUs, applies
`RLIMIT_AS = 4 GiB`, and enforces a 60-second process deadline. Solver entry points
receive 55 seconds, including imports, JSON parsing, construction, solve, and
output; the candidate reserves write time. Peak **RSS** is reported separately.
The AS cap is virtual address space, not a resident-memory cap. Official plus
independent full checks run afterward and have separate timing fields. Running
`solve.py` directly uses its internal deadline; use `replay.py` for the measured
resource envelope.

Measured stack: NumPy 2.5.3, HiGHS 1.15.1, OR-Tools 9.15.6755 with SCIP 10.0.0 /
SoPlex 8.0.0. HiGHS and SCIP are loaded in separate processes because these installed
wheels have a libhighs symbol conflict when combined. [Environment](results/environment.json).

## Selected mechanism

`solve.py` is a fixed **K=16** arc-pool MIP. Each customer selects the 16 lowest
input-derived unit-shipping-plus-capacity-rent scores (or all sites when M<16).
Its own cold full-graph regret greedy supplies a feasible starting witness; every
positive greedy fraction is added to the pool, including zero-demand rows.
All N demand rows, all M capacity rows, all M binary facility variables, every
pool-arc VUB, the aggregate constraint, and original costs remain represented.
Every other original arc is implicitly zero. The shared validator certifies the
emitted witness against the complete original instance.

The sparse MIP performs coupled facility-set changes. Its native `kOptimal`
status certifies **only that selected arc subproblem**. Whole-solution status is
always `Feasible`, and candidate output carries no global bound or gap. Both arc
selection and initialization use only the supplied input, without reference or
baseline witnesses. If native `startSolve()/wait()` is still pending at deadline,
only an earlier completed and independently verified snapshot is written; live
native solution state is left unread and the process exits immediately.

## All six selected results

One cold run per case, seed 1. The greedy column is original-cost initialization
and time to first independently verified witness. The last column is objective
excess over the old Gurobi incumbent, not a certified MIP gap.

| Case / receipt | Greedy cost / first s | Selected cost | Open | Whole s | Peak RSS MiB | Full checks | Excess over reference |
|---|---:|---:|---:|---:|---:|---|---:|
| [tiny](results/latest/tiny-solve-s1/result.json) | 8595.24863 / 0.185 | 6153.25793 | 34 | 3.885 | 89.1 | PASS | +0.00% |
| [large1](results/latest/large1-solve-s1/result.json) | 61771.41869 / 0.402 | 40150.95332 | 250 | 53.013 | 556.4 | PASS | +0.10% |
| [large2](results/latest/large2-solve-s1/result.json) | 28490.18721 / 0.369 | 21157.21443 | 273 | 53.017 | 626.4 | PASS | +0.01% |
| [large3](results/latest/large3-solve-s1/result.json) | 165636.84792 / 0.384 | 162950.20392 | 263 | 2.888 | 316.2 | PASS | +7.48% |
| [large4](results/latest/large4-solve-s1/result.json) | 121278.69121 / 0.416 | 85458.43883 | 309 | 53.046 | 399.8 | PASS | +7.44% |
| [large5](results/latest/large5-solve-s1/result.json) | 320938.60984 / 0.395 | 312314.60313 | 322 | 6.705 | 343.0 | PASS | +56.90% |

**Quality limit:** large3 and large5 returned pool-optimal solutions quickly, yet
are +7.48% and +56.90% above the old full-model reference objectives. Fixed arc
restriction is a real quality failure on these cases. Those results remain in
the default six-case table. Large4 is also +7.44% above the reference. The large1
and large2 objectives are close to the reference incumbents; runtime/hardware
regimes differ. All six selected outputs passed both complete witness checks.

## Cold full-MIP controls

Both controls use every original shipping arc and all facilities, with no greedy
or reference warm start. The retained HiGHS default is the compact full model:
for positive demand, capacity plus nonnegative x enforce closed-site zero flow;
zero-demand rows retain explicit VUBs. SCIP uses the same integer semantics.
HiGHS uses two threads; SCIP's retained control uses one thread within the same
2-CPU allocation after the actual two-thread large4 memory failure.

| Case | HiGHS full compact outcome | Whole s / RSS MiB | SCIP full outcome | Whole s / RSS MiB |
|---|---|---:|---|---:|
| tiny | 6153.25793 | 5.188 / 117.3 | 6153.25793 | 14.381 / 227.4 |
| large1 | pending; no witness | 55.077 / 1472.2 | allocation ABNORMAL; no witness | 21.698 / 3762.1 |
| large2 | pending; no witness | 55.111 / 1472.2 | allocation ABNORMAL; no witness | 20.841 / 3762.2 |
| large3 | pending; no witness | 55.102 / 1472.2 | allocation ABNORMAL; no witness | 20.993 / 3762.2 |
| large4 | pending; no witness | 55.086 / 1476.5 | allocation ABNORMAL; no witness | 22.329 / 3762.1 |
| large5 | pending; no witness | 55.072 / 1476.6 | allocation ABNORMAL; no witness | 21.252 / 3762.0 |

The five large HiGHS processes were still pending at their whole entry deadlines;
there was no safely returned whole-solution file. SCIP's five large processes
returned allocation/ABNORMAL failures under the 4-GiB AS cap. These are measured
outcomes of this pinned model and wrapper. Formal competition ranking and broader
solver performance are separate measurements.

The original full-VUB HiGHS control gave the same tiny cost in 7.597 seconds and
no large4 whole witness in 55.111 seconds. Presolve-off large4 also remained pending
at 55.101 seconds (2527.2 MiB RSS). One- and two-thread SCIP large4 failures are
both preserved. [All original and corrected controls](results/archive/).

## Earlier full-transport method and corrections

`transport.py` keeps every original arc and reuses a continuous q=d*x transport LP,
dynamic slope scaling, original-cost fixed-site reflow, and facility exchanges.
V1 stopped after four improving rounds; V2 used remaining time and a local-stop
condition. The complete V2 results remain:

| Case | Original cost | Whole s | RSS MiB | Actual returned work |
|---|---:|---:|---:|---|
| tiny | 6188.62821 | 0.723 | 55.1 | returned full-arc LP / exchanges |
| large1 | 42464.77438 | 51.512 | 887.9 | returned full-arc LP / exchanges |
| large2 | 23008.14770 | 34.143 | 888.2 | returned full-arc LP / exchanges |
| large3 | 165636.84792 | 18.837 | 887.9 | greedy pending fallback |
| large4 | 87788.65641 | 51.467 | 888.0 | returned full-arc LP / exchanges |
| large5 | 320938.60984 | 18.837 | 887.9 | greedy pending fallback |

On large3/large5 the first slope LP reached its phase deadline while pending, so
these are explicitly the saved greedy witnesses, not optimized transportation
outputs. The first tiny SCIP formatter also added an unnecessary 178 fixed-cost
charge for a tiny closed-site numerical flow; native cost was 6153.25793 and the
old full witness cost was 6331.25793. The binary set is authoritative in the
corrected formatter. Old raw outputs and frozen source variants remain in
[results/archive](results/archive/) and [versions](versions/).

## Upstream Gurobi references

The upstream one-thread records have a different runtime/hardware regime; some
run for an hour. Their reported gaps/statuses belong to those records. They are
evaluation references, separate from the cold open-source comparison.

| Case | Reference objective | Upstream whole s | Upstream reported gap |
|---|---:|---:|---:|
| tiny | 6153.25793 | 2.02 | 0.00806% |
| large1 | 40108.89529 | 3614.82 | 0.15291% |
| large2 | 21154.35052 | 3615.21 | 0.10392% |
| large3 | 151605.78671 | 198.44 | 0.00327% |
| large4 | 79542.60878 | 3614.89 | 0.32766% |
| large5 | 199049.57606 | 669.75 | 0.00942% |

[Reference JSON / checker records](results/reference/). Algorithm entry points
read only their input path; evaluation references are outside their data flow.

## Audit and licenses

- [Machine-readable six-case summary](results/summary.json) and per-run raw JSON/logs.
- [Raw/published artifact hashes](results/archive-manifest.json); infrastructure process
  IDs and private path strings are normalized in published copies.
- [Frozen code / upstream hashes](source/code-manifest.json) and [input hashes](source/input-hashes.json).
- `python common.py` runs the existing small float/split/zero-demand/cost/rejection
  assert; recorded native synthetic and CSC checks are retained in results.
- Original code: [MIT](LICENSE). Upstream source and benchmark-derived artifacts:
  CC BY 4.0, [NOTICE](NOTICE.md). Pinned FrontierOR revision
  `ef42a2eac13595453f986d9df0b62e86e75ab596`, task `klose2005`.

This is a self-contained implementation of mature mechanisms and a bounded
benchmark study. Its public contribution is reproducible full feasibility,
resource/quality evidence, and the retained failures.
