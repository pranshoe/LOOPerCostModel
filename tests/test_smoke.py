"""End-to-end smoke test.

Runs the small bundled data slice through the pipeline in memory and checks that feature
tensors have the expected fixed shapes and that the model produces exactly one finite
predicted speedup per datapoint.

Usage:  python tests/test_smoke.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from looper_cost_model.constants import MAX_EXPR_LEN, MAX_NUM_TRANSFORMATIONS, MAX_TAGS
from looper_cost_model.data import features as F
from looper_cost_model.data.looperset import normalize_schedule, stream_jsonl_gz
from looper_cost_model.model import LOOPerCostModel
from looper_cost_model.predict import _schedule_input


def run(mini_path):
    model = LOOPerCostModel().eval()
    n_datapoints = 0
    for program in stream_jsonl_gz(mini_path):
        if F.drop_program(program):
            continue
        program_json = program["program_annotation"]
        try:
            template = F.build_representation_template(program)
        except Exception:
            continue

        for raw in program["schedules_list"]:
            if F.drop_schedule(raw):
                continue
            try:
                schedule_json = normalize_schedule(program_json, raw)
                comps, loops, expr = F.fill_schedule_representation(program_json, schedule_json, template)
            except Exception:
                continue

            # Shape checks.
            assert comps.shape[0] == 1 and comps.shape[2] == 33 + MAX_TAGS * MAX_NUM_TRANSFORMATIONS + 813
            assert expr.shape[0] == 1 and expr.shape[2] == MAX_EXPR_LEN and expr.shape[3] == 11
            assert loops.shape[0] == 1 and loops.shape[2] == 8

            # Forward pass: exactly one finite predicted speedup (LeakyReLU keeps it from
            # collapsing to zero but may be marginally negative by design).
            with torch.no_grad():
                pred = model(_schedule_input(program_json, schedule_json, template))
            assert pred.numel() == 1 and torch.isfinite(pred).all()
            n_datapoints += 1
            if n_datapoints >= 200:
                print(f"SMOKE OK: {n_datapoints} datapoints, shapes and forward pass valid.")
                return
    assert n_datapoints > 0
    print(f"SMOKE OK: {n_datapoints} datapoints, shapes and forward pass valid.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, ".testdata", "mini.jsonl.gz")
    run(path)
