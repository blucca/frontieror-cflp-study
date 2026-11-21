# Pinned upstream source

FrontierOR `klose2005`, revision `ef42a2eac13595453f986d9df0b62e86e75ab596`. See [NOTICE](../NOTICE.md).

- [Instance fields](instance_schema.json); c contains the cost of an entire customer's demand.
- [Solution fields](solution_schema.json); assignments are continuous fractions, omitted arcs zero.
- [Official checker](../feasibility_check.py); [original Gurobi model](gurobi_code.py).
- [Input SHA-256/size receipts](input-hashes.json), recorded from the actual six benchmark inputs.
- [Code and upstream byte hashes](code-manifest.json).

`replay.py --source_only` obtains original inputs into the git-ignored `input/`
directory. Reference witnesses are in `results/reference/` for evaluation.
