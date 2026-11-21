# Preserved source checkpoints

`candidate-v1/` and `candidate-v2/` retain the frozen original module bytes.
V1 stopped after four exchange rounds; V2 used remaining time and an actual
local-stop condition. The old SCIP formatter converted a tiny closed-site flow
into an extra opening charge; its original output and corrected control remain
in `results/archive/`. Each archived source has a SHA-256 record in
`source/code-manifest.json`. The current public runner is `replay.py` at repo root.
