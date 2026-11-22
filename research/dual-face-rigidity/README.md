# The retained transport cut already wins its exact dual-face test

**Decision: stop the proposed dual-face optimizer before running it.** The
retained `large4` transport witness has a connected positive-flow support. Its
optimal duals have one common-shift direction, and the existing cut chooses the
core-point maximum along that direction. A new read-only audit covers all
1,125,000 original arcs. The current solver and all six cold results stay intact.

This is a classical Magnanti–Wong-style dual-face/core-point construction.
The numerical audit and the exact-face argument are distinct parts of the study.
[Audited result](result.json) / [stdlib audit](audit.py).

## The point being tested

The [earlier two-set experiment](../../versions/joint-master-conditional/)
produced costs 99,466.86294 and 101,222.70707 against the retained
84,489.46641. The all-facility cuts were valid and supported their generating
facility sets, but their binary master underestimated the new sets' actual
transport costs by about 30,000. Before implementing another optimizer we tested
whether the current dual face had unused directions that could strengthen its
first cut.

This study concerns the **initial retained fixed-S transport point**, with
1,500 positive-demand customers and 250 open facilities out of 750.
Its positive shipping cost is 52,787.46641112343. Current CFLP status remains
`Feasible`; the original native `Optimal` status concerns fixed-S transportation.

| Read-only original-input audit | Result |
|---|---:|
| Positive-flow graph | 1,750 vertices, 1,749 edges, one component |
| Smallest strictly positive transported amount | 0.011999999999957822 |
| Largest positive-edge dual-equality residual | 1.7763568394e-15 |
| Largest violation over all 1,125,000 arcs | 0 |
| Closed-facility minimum-price recomputation error | 0 |
| Demand-row flow error | 0 |
| Open-capacity slack range | [−1.5063505998e-12, 0] |
| Original objective recomputation error | 0 |

The existing dual has minimum open-facility price `0`, attained only at site
`610`; all 500 closed-facility prices are strictly positive, with minimum
`0.2185547794117646`. Source capacities use the canonical decimal `119.716`.
Thus `D=29,929`, total capacity `T=89,787`, and the 250-site capacity equals `D`
in canonical-decimal arithmetic. The original binary-floating capacity residuals
are reported separately in the table and JSON.

## Exact-face argument

For fixed sites `S`, write `q_ij=d_i*x_ij` and `u_ij=c_ij/d_i`. Positive-demand
transportation has dual

```text
maximize  sum_i d_i*alpha_i - sum_{j in S} s_j*mu_j
subject to alpha_i - mu_j <= u_ij,  mu_j >= 0; alpha free.
```

Assume an optimal primal and an optimal dual of this transport LP. Every arc
with `q_ij>0` satisfies `alpha_i-mu_j=u_ij` by complementary slackness.
Consequently, differences between any two optimal duals obey
`delta_alpha_i=delta_mu_j` on every positive-flow edge. Connectivity of the
bipartite support makes every difference the same scalar `t`:

```text
alpha'(t) = alpha + t
mu'_S(t) = mu_S + t.
```

The audited support is a spanning tree. Connectivity is the property used by
the argument. Since the reference open minimum is zero, dual nonnegativity
requires `t>=0`. For each closed site the all-arc extension is the smallest
feasible price,

```text
mu_j(t) = max(0, max_i(alpha_i+t-u_ij)) = mu_j(0)+t,
```

where the equality holds here because every reference closed price is positive.
Larger closed prices lower the core objective. With the single strict fractional
core `yCore_j=lambda=(D/T+1)/2=2/3`, the best all-facility core value therefore is

```text
gCore(t) - gCore(0) = t*(D - sum_j s_j*yCore_j) = -29,929*t.
```

Its maximum is `t=0`: the existing minimal all-arc extension. In exact-face
arithmetic this proposed first core-point optimization replays the current cut.
For a fixed set with spare total capacity, the same common shift also changes
its own dual objective; equality with the optimum then fixes `t=0` directly.

The script audits the stored numerical primal/dual point and reports its
residuals. The argument's premise is exact optimality and complementary
slackness; its application here uses the original `Optimal` record together with
the observed positive support and numerical residuals. The data table supplies
the floating-point verification scope.

## The near-optimal band has its own scope

The unexecuted proposal used `Q +/- tau`, with
`tau=max(1e-6, abs(Q)*1e-10)=5.278746641112343e-6`.
Within a near-optimal band, positive arcs may carry reduced-cost residuals;
the weighted residual budget permits directions beyond common translation.
Those directions were not optimized in this study.

For a *pure common shift* slightly below zero, only the zero-price site `610`
clips before the next strictly positive price. Its fixed-S loss is
`119.716*abs(t)`. The planned band permits
`abs(t)<=4.4093910931808136e-8`, and its corresponding core improvement is at most
`(250-2/3)*tau=0.0013161674958506776`.
This bound applies to that common-shift path. Other near-face directions and
other facility sets retain their separate, unmeasured scope.

**Stopped:** new Pareto LPs, additional master/recourse iterations, six-case cold
expansion and new defaults. New native solver runs: **0**. The read-only original
input audit took 1.171 s wall, 1.116 CPU s and 79.73 MiB peak RSS under its recorded
2-CPU / 1-GiB-address-space limits. No new facility decision was generated.

## Reproduce the audit

From the repository root on Linux, Python 3.13:

```sh
# Use the already supplied pinned large_instance_4.json.
# replay.py --source_only is the existing way to fetch the six original inputs.
python research/dual-face-rigidity/audit.py --repo . \
  --instance input/large_instance_4.json > audit-result.json
```

The full command independently reads the original input, fixed witness and dual
record, checks all positive edges and all arcs, recomputes the original cost and
capacity residuals, and prints the report. It never invokes a solver. Omitting
`--instance` audits the stored support and existing receipts only, with the
separate `PASS_STORED_SUPPORT_AUDIT` label.
The source manifest's original input digest is retained as the benchmark's
pinned-input check. The script uses standard-library modules only, restricts
itself to two CPUs and 1 GiB address space, and stops at 60 CPU / 180 wall seconds.

Audit code: MIT. Original FrontierOR benchmark: CC BY 4.0 with the attribution
and pinned revision in the repository [NOTICE](../../NOTICE.md). All prior decision
witnesses, solver source, original failed sets and cold baselines remain present.
