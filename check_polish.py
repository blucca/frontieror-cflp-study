import json
import time
import numpy as np
import common
import polish

sample = {"num_customers": 3, "num_plants": 3,
          "customers": {"demands": [2, 1, 0]},
          "plants": {"capacities": [1.5, 2.0, 0], "fixed_costs": [1, 2, -.5]},
          "shipping_costs": [[2, 6, 99], [4, 1, 99], [9, 3, 1]]}
prior = common.from_flows(sample, np.asarray([[0, 2, 0], [1, 0, 0], [0, 0, 0.]]), opened=[0, 1, 2])
# A valid but deliberately expensive zero-demand row exercises its constant.
prior["assignments"]["2"] = {"0": 1.0}
prior["objective_value"] += 8
common.verify(sample, prior)

def pending(reason, **values):
    raise AssertionError("fixture solve returned pending: " + reason)

metrics = {}
sol = polish.refine(sample, prior, time.monotonic() + 10, 1, metrics, pending)
assert sol["open_plants"] == [0, 1, 2] and abs(sol["objective_value"] - 7.5) < 1e-7
assert metrics["fixed_s_model_status"] == "HighsModelStatus.kOptimal"
assert metrics["model_columns"] == 6 and metrics["model_rows"] == 5 and metrics["model_nnz"] == 12
assert sol["assignments"]["2"] == {"2": 1.0}
assert common.verify(sample, sol)["pass"]
empty = {}
assert polish.refine(sample, prior, time.monotonic() - 1, 1, empty, pending) is prior
assert empty["termination"] == "NO_POLISH_BUDGET"
print("PASS full S arcs / float capacity / split / original c/d / zero-demand cost / negative fixed / no-budget prior", flush=True)
print(json.dumps(metrics), flush=True)
cover_data = dict(sample)
cover_data["num_plants"] = 4
cover_data["plants"] = {"capacities": [1.5, 2, 0, 3], "fixed_costs": [1, 2, -.5, 20]}
cover_data["shipping_costs"] = [row + [99] for row in sample["shipping_costs"]]
cover_prior = dict(prior)
cover_prior["open_plants"] = [0, 1, 2, 3]
cover_prior["objective_value"] += 20
common.verify(cover_data, cover_prior)
cover, cover_metrics = polish.capacity_cover(cover_data, cover_prior)
assert cover["open_plants"] == [0, 1, 2] and abs(cover["objective_value"] - 7.5) < 1e-7
assert cover_metrics["selected_capacity"] == 3.5 and cover_metrics["selected_fixed_cost"] == 2.5
assert common.verify(cover_data, cover)["pass"]
assert polish.capacity_cover(sample, prior)[1]["termination"] == "S_ALREADY_FEE_CAPACITY_COVER"
print("PASS capacity-cover / expensive unused closure / negative fee retained / float capacities / true cost / already-cover",flush=True)
