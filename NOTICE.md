# Licenses and source attribution

Original solver, validator, runner, and documentation code is MIT licensed (LICENSE).

The upstream model implementation, descriptive schemas, official feasibility checker,
reference solutions, and benchmark-derived artifacts are attributed to the FrontierOR
authors and distributed under Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/ and
https://creativecommons.org/licenses/by/4.0/legalcode .

Upstream dataset: https://huggingface.co/datasets/frontieror/FrontierOR
Pinned revision: `ef42a2eac13595453f986d9df0b62e86e75ab596`; task `klose2005`.
Task source: https://huggingface.co/datasets/frontieror/FrontierOR/resolve/ef42a2eac13595453f986d9df0b62e86e75ab596/tasks/klose2005/

The upstream task credits Klose & Drexl (2005), “Lower Bounds for the Capacitated
Facility Location Problem Based on Column Generation”, Management Science
51(11):1689–1705. That credit is retained as supplied by the task.

`feasibility_check.py`, `source/gurobi_code.py`, and the two `source/*schema.json`
files are unchanged upstream bytes. File URLs and SHA-256 values are recorded in
`source/code-manifest.json`. Upstream reference witnesses in `results/reference/`
are evaluation artifacts; solver entry points read only the supplied instance.
The approximately 86.6 MB instance set is obtained by pinned URL and SHA-256
through `replay.py`, and is excluded from this repository.

Public result JSON is portably formatted and omits infrastructure process IDs.
Logs retain measured failures; private filesystem paths are normalized if present.
Raw and published content hashes are recorded in `results/archive-manifest.json`.
No paper PDFs are redistributed. This is an independent implementation of mature
heuristics and a benchmark study.
