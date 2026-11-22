# Demand-limited facility cuts for warehouse supply — stopped gate

2026-11-22. **The complete bounded gate finished, and its cost-improvement test
failed.** Seven new facility sets have Optimal fixed-set transportation and pass
both full witness checkers. Every new set costs more than the retained prior;
the best output remains **84,489.466411**. The closest new set costs
84,906.987584, **0.49417% higher**. Whole time including the final two checkers is
35.475 seconds. The default solver and its six cold results stay unchanged.

## The planning task

Choose which warehouses or factories to open, then allocate each customer's
demand among them. Opening a site incurs its fixed fee; shipped fractions incur
the supplied transport costs and consume site capacity. Demand may be split
across sites. The complete output contains the binary open-site set and every
customer's allocation fractions, with original cost recomputed from that witness.

This study uses the original `large4` model: **1,500 customers, 750 possible
sites and all 1,125,000 shipping arcs**. It starts from the already produced
[250-site fast solution](../../results/cover/large4-hybrid-s1/solution.json).
That solution is supplied as an input to this research gate. The gate measures a
stored-prior continuation; the [six-case cold study](../../README.md#all-six-selected-results)
measures construction of a solution from the instance alone.

The practical question is whether stronger transport estimates help a small
joint facility master choose a cheaper complete supply plan within a fixed
process budget. The [earlier two-set master](../../versions/joint-master-conditional/)
returned valid plans whose transport costs were about 30,000 above their cut
estimates. The [dual-face follow-up](../dual-face-rigidity/) closed its proposed
exact-face price optimization. Here we keep the available demand prices and
change how each facility's contribution is bounded.

## The mathematical change

For positive demand, write `q_ij = d_i*x_ij`, `u_ij = c_ij/d_i` and
`beta_ij = alpha_i-u_ij`, using demand prices `alpha` from a completed Optimal
fixed-site transportation LP. The original `c_ij` prices the customer's entire
demand, so dividing by `d_i` gives cost per shipped unit.

The earlier cut gives a facility's whole capacity its highest customer benefit:

```text
C_old_j = s_j * max(0, max_i beta_ij)
```

The original model also has the opening link `q_ij <= d_i*y_j`, often called a
variable upper bound. A customer can supply at most its own demand to a site.
Keeping that limit gives a continuous knapsack for each of the 750 sites:

```text
C_j = max sum_i beta_ij*q_i
      subject to 0 <= q_i <= d_i, sum_i q_i <= s_j

L0 = sum_{i:d_i=0} min_j c_ij
shipping >= sum_{i:d_i>0} d_i*alpha_i + L0 - sum_j C_j*y_j
```

Descending benefit order computes the knapsack fill. A threshold price
`lambda_j >= 0` and `nu_ij = max(0, beta_ij-lambda_j)` independently give its dual
value `s_j*lambda_j + sum_i d_i*nu_ij`; the implementation compares both values
and checks all original arc inequalities.

In exact arithmetic, `0 <= C_j <= C_old_j`. The new cut therefore dominates the
earlier cut at the same demand prices: the facility discount respects the amount
of demand actually available at each benefit. The cut is valid for binary site
decisions and for fractional decisions that retain the original opening links.
At its generating Optimal binary site set it retains positive-demand transport
support. Zero-demand customers use the global `L0` floor; their actual original
costs remain in each complete witness. Negative fixed-fee sites retain the
existing forced-open semantics.

Continuous knapsack, variable upper bounds and Benders decomposition are mature
methods. The contribution here is their complete implementation on this supplied
model, checked coefficient evidence and a bounded whole-method result.

## Original-input coefficient analysis

[The read-only report](results/lift-stats.json) evaluates the stored initial
demand prices over all 750 sites. It generates no new facility sets and invokes
no numerical solver.

| Analysis item | Recorded result |
|---|---:|
| Original arcs checked | 1,125,000 |
| Closed sites with coefficient reduction above `1e-7` | 500 / 500 |
| Sum of facility-discount reductions | 22,172.411334 |
| Median closed-site reduction | 38.651964 / 6.96652% |
| Maximum knapsack primal/dual absolute difference | 3.41061e-13 |
| Maximum all-arc dual violation | 0 |
| Prior positive-transport support error | 6.54836e-11 |
| Wall / CPU seconds; peak RSS | 1.129 / 1.179; 88.71 MiB |

On the two **earlier** joint sets, the initial shipping cut rises by
3,431.498348 and 3,990.537799. Their remaining differences from actual shipping
are still 26,571.899417 and 26,953.110358. These are stronger estimates on stored
decisions. Actual new-plan costs are measured by the separate gate below.

The read-only analysis uses outward-rounded prices and rational accumulation.
The executed `joint.py` uses ordinary binary floating-point arithmetic with
explicit tolerance checks. Their reports retain those separate numerical
scopes, including tiny protective-rounding changes at the 250 open sites.

## Complete stored-prior gate

Seed 1, two allocated CPUs and a **4-GiB virtual-address-space limit** are fixed.
Peak resident memory is reported separately. The algorithm shares one absolute
55-second budget across imports, input parsing, the initial transportation LP,
all pricing, masters, warm allocation, complete recourse, in-worker checks and
output. The 60-second acceptance gate includes the final independent and upstream
full checkers. Small fixture checks precede that large4 timer and have their own
receipts.

The initial LP receives at most 10 seconds. Each master receives at most
2 seconds, including its construction. The loop permits **eight distinct site
sets total: the initial set plus at most seven new sets**. Every new set is
evaluated using all positive-demand customer-to-open-site arcs. A completed
Optimal recourse dual supplies the next full-facility cut. If native work is
still pending at the application deadline, the algorithm writes its previously
verified best and exits.

Passing requires at least one new Optimal fixed-site recourse, all coefficient
and witness checks, whole time at most 60 seconds, and a best actual cost at
least **0.5% below** 84,489.46641112343. The latter test fails.

All seven recourse LPs return Optimal. All seven candidate witnesses and the
retained best pass both complete checkers. The master returns Optimal on the
first two sets and a complete binary incumbent at TimeLimit on sets 3–7.

| Set | Open sites | Actual total cost | Change from prior | Master status |
|---|---:|---:|---:|---|
| Retained prior | 250 | 84,489.466411 | 0.00000% | supplied witness |
| 1 | 250 | 102,220.276562 | +20.98582% | Optimal |
| 2 | 250 | 101,336.323879 | +19.93960% | Optimal |
| 3 | 259 | 91,413.102349 | +8.19467% | TimeLimit |
| 4 | 265 | 84,906.987584 | +0.49417% | TimeLimit |
| 5 | 262 | 88,809.798412 | +5.11346% | TimeLimit |
| 6 | 258 | 92,802.928844 | +9.83964% | TimeLimit |
| 7 | 260 | 90,444.608010 | +7.04838% | TimeLimit |

The closest set, 4, lowers shipping to 52,043.987584 but incurs fixed fees of
32,863. The prior costs 31,702 in fixed fees plus 52,787.466411 in shipping;
the new set's larger fixed fee leaves its total higher.

| Complete receipt | Value |
|---|---:|
| Algorithm worker process | 33.986328 s |
| Algorithm's internal whole-time metric | 33.879558 s |
| Whole process plus final full checkers | 35.474743 s |
| Peak worker RSS | 473,056 KiB / 461.97 MiB |
| Distinct site sets / completed new Optimal recourse LPs | 8 / 7 |
| Full-facility cuts with support, arc and knapsack checks | 8 / 8 PASS |
| Cumulative original-arc checks across the eight cuts | 9,000,000 |
| Candidate plus retained-best full double checks | 8 / 8 PASS |
| Termination / quality decision | `JOINT_GATE_COMPLETE` / **STOP** |

Across these eight cuts, maximum absolute positive-support error is
1.45519e-11, knapsack primal/dual difference 2.27374e-13, and coefficient
dominance excess 1.13687e-13. All-arc violation and greedy capacity excess are 0.

[Complete solutions, cuts, native statuses and logs](results/complete/)
retain all seven outcomes. The loop ends normally, with zero best-cost gain and
no pending solve or exception. Stronger facility prices and feasible, timely
outputs are the observed mechanism and engineering results; the required cheaper
supply plan was not found within this fixed gate.

## The first API failure and its narrow repair

The first attempted run returned the same first two costs above, then stopped
at the third master. `Highs.passModel` received 3,753 sparse entries. Three exact
zeros and one positive `1.46549e-14` entry were filtered by the recorded
`small_matrix_value = 1e-9` threshold, and the loaded model returned `kWarning`.
The old assertion required `kOk` and stopped the worker.

The repair accepts `kOk` and `kWarning`; `kError` still stops model admission.
The [native small check](check_master.py) reproduces the warning using a valid
redundant tiny-coefficient cut and passes. A separate replay of the saved
three-cut master returned a complete 259-site TimeLimit incumbent; that diagnostic
checked model loading and selection. The seven-set run above supplies the actual
recourse, feasibility, cost and whole-time result after the repair. Masters 3–7
record `kWarning` in that complete run.

[First partial-run records](results/first-api-failure/) preserve its two worse
plans and the assertion failure. The source repair and the scientific stopping
decision are recorded separately.

## Replay and evidence

From the repository root, use the existing pinned Linux/Python environment
(measured Python 3.13.5, NumPy 2.5.3, HiGHS 1.15.1) and the original large4 input.
The root [replay instructions](../../README.md#reproduce-the-whole-process-experiment)
fetch and hash-check that input.

```sh
STUDY=research/bounded-demand-vub-20261122
mkdir -p temp/bounded-demand-vub-readback
python -B "$STUDY/analyze_lift.py" --repo . \
  --instance input/large_instance_4.json \
  > temp/bounded-demand-vub-readback/lift-stats.json
python -B "$STUDY/check_cut.py"
python -B "$STUDY/check_master.py"
python -B "$STUDY/replay_gate.py" \
  --instance_path input/large_instance_4.json \
  --prior_path results/cover/large4-hybrid-s1/solution.json \
  --output_dir temp/bounded-demand-vub-gate-001
```

The output directory for the gate must be new. The portable runner provides the
recorded resource envelope and full checks; direct `joint.py` execution provides
the internal deadline. [Provenance](provenance.json) distinguishes measured
method bytes, the first failed source, the controller's path-only adaptation,
and existing input/prior/control references. Measured method files are byte-equal to their captured run. Input data and old controls are reused
at their existing repository paths.

**Stopped:** six-case cold expansion of this method, additional seeds, larger
budgets, more site sets and default integration. This result covers one complete
original instance, one stored prior and one fixed bounded loop. Further
convergence and cold start-to-output performance have their own unmeasured scope.
The original [cold full-MIP controls](../../README.md#cold-full-mip-controls)
remain the controls for the six-case default study.

Independent software is MIT. Benchmark attribution, derived witnesses and the
pinned FrontierOR `klose2005` revision follow [NOTICE](../../NOTICE.md), including
the supplied credit to Klose & Drexl (2005). This is an independent public
research study; formal FrontierOR entry and full-suite evaluation have their
separate scope.
