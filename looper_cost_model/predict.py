"""Inference for LOOPerCostModel: predict schedule speedups for LOOPerSet programs.

Given a trained checkpoint and a LOOPerSet ``*.jsonl.gz`` file, predicts the speedup of
every schedule of every program (features are extracted on the fly, no caching):

    python -m looper_cost_model.predict --config configs/default.yaml \
        eval.weights_path <checkpoint.pt> data.valid_file <programs.jsonl.gz>
"""

import torch

from .config import load_config
from .data import features as F
from .data.looperset import ScheduleTooComplexException, normalize_schedule, stream_jsonl_gz
from .constants import (
    ExpressionTooLongException,
    LoopsDepthException,
    NbAccessException,
    NbTransformationException,
)
from .model import LOOPerCostModel


def _schedule_input(program_json, schedule_json, template):
    """Build the single-datapoint model input tuple for one schedule."""
    comps_tensor, loops_tensor, expr_tensor = F.fill_schedule_representation(
        program_json, schedule_json, template
    )
    _, num_comps, _ = comps_tensor.shape
    first_part, vectors, third_part = F.split_computation_vector(
        comps_tensor.view(num_comps, -1)
    )
    return (
        template.tree,
        first_part.view(1, num_comps, -1),
        vectors,
        third_part.view(1, num_comps, -1),
        loops_tensor,
        expr_tensor,
    )


@torch.no_grad()
def predict_program(model, program, device):
    """Yield ``(schedule_index, schedule_str, predicted_speedup)`` for each schedule."""
    model.eval()
    program_json = program["program_annotation"]
    try:
        template = F.build_representation_template(program)
    except (LoopsDepthException, NbAccessException, ExpressionTooLongException):
        return

    for schedule_index, raw_schedule in enumerate(program["schedules_list"]):
        try:
            schedule_json = normalize_schedule(program_json, raw_schedule)
            inputs = _schedule_input(program_json, schedule_json, template)
        except (NbTransformationException, ScheduleTooComplexException):
            continue
        inputs = (inputs[0],) + tuple(t.to(device) for t in inputs[1:])
        prediction = model(inputs).item()
        yield schedule_index, raw_schedule.get("schedule_str", ""), prediction


def run(cfg, max_programs=None):
    device = cfg.training.device
    model = LOOPerCostModel(
        input_size=cfg.model.input_size,
        comp_embed_layer_sizes=cfg.model.comp_embed_layer_sizes,
        drops=cfg.model.drops,
    )
    if not cfg.eval.weights_path:
        raise ValueError("eval.weights_path must point at a trained checkpoint.")
    model.load_state_dict(torch.load(cfg.eval.weights_path, map_location=device))
    model = model.to(device)

    for i, program in enumerate(stream_jsonl_gz(cfg.data.valid_file)):
        if max_programs is not None and i >= max_programs:
            break
        print(f"# {program['program_name']}")
        for schedule_index, sched_str, prediction in predict_program(model, program, device):
            print(f"  schedule {schedule_index:>4}  pred_speedup={prediction:.4f}  {sched_str}")


def main():
    run(load_config())


if __name__ == "__main__":
    main()
