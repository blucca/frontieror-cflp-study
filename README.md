# FrontierOR CFLP: route freely, then close the costly sites

A runnable, cold-input study of splittable capacitated facility location. The
fast path emits a fully checked solution for all six pinned benchmark cases
under the recorded 2-CPU / 4-GiB-address-space / whole-60-second envelope.
Full-arc transportation and a fee/capacity cover repair the two largest facility
cost failures. The original longer pool wins on two other cases; both paths and
all weak results are available below.

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
# Preserved longer-pool quality/time trade-off:
python replay.py --batch pool-001 --algorithms pool
# Optional terminal-supported variant and earlier slope-transport method:
python replay.py --batch terminal-001 --algorithms terminal
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

`solve.py` uses three bounded steps, all derived from its supplied input:

1. **Cold feasible transport, then a coupled arc-pool MIP.** K=16 unit-shipping +
   capacity-rent neighbors per customer, plus every positive fraction of its own
   complete-graph regret greedy. All facilities, original costs, capacity rows,
   demand rows, pool VUBs and aggregate capacity remain. The pool phase ends at
   20 seconds of the same whole entry budget.
2. **Full transportation over the selected facilities S.** `polish.py` keeps
   every positive-demand customer-to-S arc: q=d*x, cost c/d, p+|S| rows and
   2p|S| nonzeros. Closed sites are exactly eliminated. All S fixed fees remain;
   zero-demand customers choose their least original c within S.
3. **One fee/capacity cover and full recourse.** Within S, retain all negative
   fixed-fee sites; order remaining sites by f/s, breaking equal ratios by the
   actual transported load, and select enough floating capacity for all demand.
   On equal positive capacities this minimizes the cover's fixed fee within S; general
   capacities use a greedy heuristic. Re-route the complete graph on this subset,
   immediately verify its feasible witness, then optimize all its transport arcs
   using remaining whole time. Keep the lowest original-cost verified witness.

The output includes all customer fractions and their binary open-site set.
Native pool `kOptimal` concerns the selected arc subproblem; transportation LP
`kOptimal` concerns its fixed S. The full CFLP output is always **Feasible**, with
no global bound/gap exported. Initialization and all refinements use the supplied
instance; reference witnesses are outside their data flow. A pending native
`startSolve()/wait()` writes the previously completed, independently verified best
and exits immediately, leaving live native solution state unread.

The 20-second pool allocation leaves time for full recourse instead of using all
53 internal seconds on sparse MIP. It is a measured quality/time trade-off, not
an adaptive parameter search. The original 53-second pool is retained in
`versions/arc-pool-baseline/solve.py` and replay's `pool` mode.

## All six selected results

One cold run per case, seed 1. The greedy column is original-cost initialization
and time to first independently verified witness. The last column is objective
excess over the old Gurobi incumbent, not a certified MIP gap.

| Case / receipt | Greedy cost / first s | Final cost | Open | Whole s | Peak RSS MiB | Full checks | Excess over reference |
|---|---:|---:|---:|---:|---:|---|---:|
| [tiny](results/cover/tiny-hybrid-s1/result.json) | 8595.24863 / 0.066 | 6153.25793 | 34 | 3.583 | 85.5 | PASS | +0.00% |
| [large1](results/cover/large1-hybrid-s1/result.json) | 61771.41869 / 0.301 | 41609.61054 | 250 | 24.074 | 388.3 | PASS | +3.74% |
| [large2](results/cover/large2-hybrid-s1/result.json) | 28490.18721 / 0.285 | 22149.38992 | 339 | 24.684 | 406.0 | PASS | +4.70% |
| [large3](results/cover/large3-hybrid-s1/result.json) | 165636.84792 / 0.294 | 151943.14944 | 250 | 12.768 | 408.0 | PASS | +0.22% |
| [large4](results/cover/large4-hybrid-s1/result.json) | 121278.69121 / 0.290 | 84489.46641 | 250 | 22.006 | 408.7 | PASS | +6.22% |
| [large5](results/cover/large5-hybrid-s1/result.json) | 320938.60984 / 0.300 | 200612.77666 | 250 | 18.550 | 403.8 | PASS | +0.79% |

**Observed trade-off:** the selected path restores large3/large5 to about
+0.22%/+0.79% of the upstream reference incumbents, and improves large4's original
pool cost. On large1/large2 it costs +3.63%/+4.69% more than the original pool,
while taking about 24 seconds instead of 53 seconds. Every case and both costs
remain in the study; apply the complete replay to your own model before choosing
its quality/time budget.

### Preserved original 53-second arc pool

| Case / receipt | Greedy cost / first s | Original pool cost | Open | Whole s | Peak RSS MiB | Full checks | Excess over reference |
|---|---:|---:|---:|---:|---:|---|---:|
| [tiny](results/latest/tiny-solve-s1/result.json) | 8595.24863 / 0.185 | 6153.25793 | 34 | 3.885 | 89.1 | PASS | +0.00% |
| [large1](results/latest/large1-solve-s1/result.json) | 61771.41869 / 0.402 | 40150.95332 | 250 | 53.013 | 556.4 | PASS | +0.10% |
| [large2](results/latest/large2-solve-s1/result.json) | 28490.18721 / 0.369 | 21157.21443 | 273 | 53.017 | 626.4 | PASS | +0.01% |
| [large3](results/latest/large3-solve-s1/result.json) | 165636.84792 / 0.384 | 162950.20392 | 263 | 2.888 | 316.2 | PASS | +7.48% |
| [large4](results/latest/large4-solve-s1/result.json) | 121278.69121 / 0.416 | 85458.43883 | 309 | 53.046 | 399.8 | PASS | +7.44% |
| [large5](results/latest/large5-solve-s1/result.json) | 320938.60984 / 0.395 | 312314.60313 | 322 | 6.705 | 343.0 | PASS | +56.90% |

The original pool returned quickly on large3/large5 but left +7.48%/+56.90%
reference excess. Their fixed-fee excess explains 78.72%/95.35% of the original
cost difference: rent-adjusted K16 plus cold sparse support kept costly capacity
fragmented. Full-S transport unlocks re-routing; closing the unnecessary sites
then captures most of the improvement.

### Mechanism probes and terminal-pool variant

The first cold full-S probe retained 263/322 open sites and cost
160413.31037/305968.38056 on large3/large5 (9.95/15.08 seconds). Its fixed-S LPs
returned Optimal. The next capacity cover selected 250 sites in both cases,
with fixed fees 149040/188000, and complete cold costs
151943.14944/200612.77666 (13.54/19.26 seconds). These probes used the original
pool deadline; their complete records are in
[fixed-S gate](results/archive/fixed-S-gate/) and
[cover gate](results/archive/cover-gate/).

A final experimental pass expands the same K16 pool with the current full-arc
best support while retaining every original cold-greedy arc, and spends only the
remaining whole budget on a coupled MIP. It preserves the current feasible best
when native solve remains pending. Its complete six-case outcomes are retained
in [terminal-six](results/archive/terminal-six/) and summary JSON; the selected
fast entry remains the three-step version above.

| Case | Terminal cost | Whole s | Actual final work |
|---|---:|---:|---|
| tiny | 6153.25793 | 3.381 | same optimal pool; skipped |
| large1 | 41609.61054 | 53.067 | native pending; prior complete best kept |
| large2 | 22149.38992 | 53.059 | native pending; prior complete best kept |
| large3 | 151866.38634 | 18.277 | returned expanded-pool MIP |
| large4 | 84489.46641 | 53.048 | native pending; prior complete best kept |
| large5 | 200358.00993 | 21.838 | returned expanded-pool MIP |

The fast entry is selected because the terminal pass retains the same quality
on three pending cases while extending their whole time to 53 seconds. Its
small returned improvements on large3/large5 are available as the separate
`terminal` replay mode.

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
- [Baseline raw/published artifact hashes](results/archive-manifest.json). New full-S,
  cover and terminal records retain their complete witnesses; controller process
  IDs and private path strings are normalized in published copies.
- [Frozen code / upstream hashes](source/code-manifest.json) and [input hashes](source/input-hashes.json).
- `python common.py` runs the existing float/split/zero-demand/cost/rejection assert.
  `python check_polish.py` checks the new fixed-S and capacity-cover paths on one
  synthetic fixture; recorded native and CSC checks are in results/validation.
- Original code: [MIT](LICENSE). Upstream source and benchmark-derived artifacts:
  CC BY 4.0, [NOTICE](NOTICE.md). Pinned FrontierOR revision
  `ef42a2eac13595453f986d9df0b62e86e75ab596`, task `klose2005`.

This is a self-contained implementation of mature mechanisms and a bounded
benchmark study. Its public contribution is reproducible full feasibility,
resource/quality evidence, and the retained failures.

## Try it on a real task

For splittable warehouse/factory supply planning, use the same input schema and
save the complete decision witness. A useful next review is one practitioner's
replay, with the returned result JSON, machine summary, and intended use (initial
incumbent or standalone solution). Open an issue with authorized public or
anonymized model material, the current formulation, runtime budget and acceptance
criteria; any full engineering engagement is scoped to that actual task.
