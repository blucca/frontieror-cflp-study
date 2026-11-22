# Full-facility cuts and two joint choices: stopped gate

This experiment takes the already produced large4 fast solution as a prior.
It captures a completed fixed-S transportation LP's demand dual, constructs
valid cuts over **all 750 original facilities**, and accumulates a second cut
after evaluating the first joint set. The complete original transportation
problem is solved on each new set, rather than estimating the saving from
independent opening/closing scores. Benders decomposition is a mature method.

## Reproduce

Use the repository's pinned environment and original source inputs. From the
repository root:

```sh
python replay.py --source_only --cases large4
python versions/joint-master-conditional/check_cut.py
python versions/joint-master-conditional/check_master.py
python versions/joint-master-conditional/replay_gate.py \
  --instance_path input/large_instance_4.json \
  --prior_path results/cover/large4-hybrid-s1/solution.json \
  --output_dir temp/joint-gate-001
```

The output directory must be new. The runner applies two-CPU affinity,
4-GiB virtual-address-space and 60-second whole-worker limits. It first runs
the small checks, then invokes the conditional experiment and checks both
candidate witnesses and the retained best using both original validators.
Its path arguments replace the original machine-specific controller paths;
the measured method files are preserved verbatim. Direct `joint.py` calls
retain internal deadlines; the wrapper supplies the external resource cap.

The gate is scoped to a positive-demand, nonzero-objective stored prior. It
allows an initial fixed-S oracle of at most 10 seconds, then 20 seconds for
cut construction, master, warm routing, at most two new sets, full recourse
and witness checkpoints. Passing would require a completed joint oracle,
at least 0.5% real cost reduction and the extra path within 20 seconds.

## Cut and master

For each positive-demand customer define `a_ij = c_ij / d_i`. Given finite
demand dual `alpha_i`, compute over **every original arc**:

```
mu_j = max(0, max_i(alpha_i - a_ij))
L0 = sum_{i:d_i=0} min_j c_ij
shipping >= sum_i(d_i * alpha_i) + L0 - sum_j(s_j * mu_j * y_j)
```

The all-arc inequalities are checked. At the generating S, the positive-demand
cut is checked against actual primal transport cost. A zero-demand minimum
within that S belongs to that S only; the global `L0` works across arbitrary
choices, including negative supplied shipping costs.

The master minimizes original `f*y + theta`, keeps capacity and a nonempty
binary set, fixes negative-fee facilities open, applies accumulated shipping
cuts and the valid unconditional `theta >= sum_i min_j c_ij` floor. Complete
greedy routing supplies a warm flow on the chosen S, followed by the preserved
full-S LP. Native Optimal statuses belong to their LPs or the cut master;
the full CFLP output remains Feasible.

## Outcome and first failure

The prior costs **84489.466411**. Two genuinely joint 250-site choices replace
77 / 86 sites and cost **99466.862943 / 101222.707071**, increases of
**17.73% / 19.81%**. Both complete witnesses pass both checkers. Two all-arc
checks each cover 1125000 arcs; generating positive primal/cut errors are
`-7.28e-12 / 4.37e-11`.

The extra path is **8.787 seconds**, whole process **11.858 seconds**, peak
RSS **376.8 MiB**. Selected-set size and master runtime are acceptable; cut
strength at unseen sets is the failure. Their fixed-fee savings are 1378 / 570,
while shipping increases 16355 / 17303. The saved prior remains the final output.

The first run completes its initial oracle, then fails a warm-matrix assertion
expression because `HighsLp.row_lower_` returns a Python list. The common
boundary is corrected with `np.asarray`; a small native two-cut master check
catches that interface and validates the joint choice. The original failure
source, logs, checks and retained prior are preserved alongside the corrected
complete gate in [results](../../results/archive/joint-master-conditional/).

The finite independent cut check covers 24 binary sets, 17 feasible sets,
352 legal allocations and four cuts, including split flows, float/zero capacity,
zero demand and negative costs. It also retains a concrete counterexample to
carrying a fixed-S zero-demand constant into a global cut.

**Stop this route.** Additional cold cases, parameter rescue, stabilization,
new controls and integration into the fast default are outside this completed
gate. Existing cold results and the terminal/pool alternatives remain available.
