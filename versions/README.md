# Preserved source checkpoints

`candidate-v1/` and `candidate-v2/` retain the frozen original module bytes.
V1 stopped after four exchange rounds; V2 used remaining time and an actual
local-stop condition. The old SCIP formatter converted a tiny closed-site flow
into an extra opening charge; its original output and corrected control remain
in `results/archive/`. Each archived source has a SHA-256 record in
`source/code-manifest.json`. The current public runner is `replay.py` at repo root.

`joint-master-conditional/` preserves a separate stored-prior large4 gate:
full-facility dual cuts, a binary joint master, two complete transport evaluations,
small checks and bounded replay. Both joint choices cost more; the route stopped
and the cold default remains unchanged. Its complete results include the first
native-API construction failure and the corrected gate.
