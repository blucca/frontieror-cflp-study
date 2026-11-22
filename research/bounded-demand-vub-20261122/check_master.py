"""One native master smoke for the actual highspy array/list boundary."""
import json
import time
import numpy as np
import joint
import common

def no_pending(reason, pending=False):
    raise AssertionError(reason)

data = {"num_customers": 1, "num_plants": 3,
        "customers": {"demands": [1]},
        "plants": {"capacities": [.5, .5, 1], "fixed_costs": [1, 1, 3]},
        "shipping_costs": [[1, 2, 10]]}
prior = joint.warm_for_sites(data, [2], 1)
cut = joint.make_cut(data, [10])
metrics = {}
# Valid redundant cut reproduces the actual positive-tiny matrix warning.
redundant = {"intercept": 0.0, "coeff": np.asarray([0.0, -1e-14, 0.0])}
sites = joint.master(data, [cut, joint.make_cut(data, [2]), redundant], prior, time.monotonic() + 2, 1, metrics, no_pending)
assert metrics["model_load_status"] == "HighsStatus.kWarning"
assert sites == [0, 1] and abs(metrics["master_incumbent_objective"] - 3.5) < 1e-8
warm = joint.warm_for_sites(data, sites, 1)
assert common.verify(data, warm)["pass"] and abs(warm["objective_value"] - 3.5) < 1e-8
zero = {"num_customers": 1, "num_plants": 2, "customers": {"demands": [0]},
        "plants": {"capacities": [0, 0], "fixed_costs": [1, 2]},
        "shipping_costs": [[100, 0]]}
z = joint.warm_for_sites(zero, [0], 1)
assert z["open_plants"] == [0] and z["objective_value"] == 101
print(json.dumps({"status": "PASS", "joint_sites": sites, "objective": warm["objective_value"],
                  "all_zero_requested_S": z["open_plants"], "native": metrics["native_status"]}))
