# Model–search interface

This is the inference bridge between LOOPer's C++ polyhedral search and the Python cost
model. The search process launches `main.py` as a long-lived subprocess and, for every
candidate schedule it wants scored, writes three JSON lines to the process's stdin and
reads one predicted speedup back from stdout.

It is **self-contained**: it does not import the training repository. The only third-party
dependencies are `torch`, `numpy`, and `sympy`.

## Files

| File | Role |
| --- | --- |
| `main.py` | Entry point: loads the model, then loops reading schedules from stdin and printing predicted speedups. |
| `json_to_tensor.py` | Turns the search module's program/schedule JSON into the model's fixed-size input tensors. |
| `model.py` | The cost-model network (`LOOPerCostModel`, a recursive-LSTM regressor). |

## Protocol

Per candidate, the search writes three single-line JSON documents to stdin:

1. `program_json` — the program annotation (`memory_size`, `iterators`, `computations`).
2. `no_sched_json` — the unscheduled program (used to build the base loop-nest tree); must
   carry `tree_structure` and no fusions.
3. `sched_json` — the candidate schedule: per-computation `tiling`, `unrolling_factor`,
   `parallelized_dim`, `shiftings`, and a `transformations_list` of 16-element affine tags
   (interchange / reversal / skewing), plus program-level `fusions` and `tree_structure`.

`main.py` prints one line per candidate: the predicted speedup as a float.

## Configuration

Set via environment variables:

- `MODEL_WEIGHTS_PATH` — path to the trained checkpoint. Use the weights produced by
  training the cost model (e.g. `best_model_looperset_pact25_compact.pt`). If the path
  ends in `part0.pt` / `part_0.pt` an untrained model is used (search bootstrapping).
- `TIRAMISU_ROOT` — Tiramisu installation root (read by the search integration).

The network hyperparameters are fixed to match the trained model: `input_size=846`,
computation-embedding layers `600 350 200 180`, dropout `0.05`. They can be overridden
with the `layers` / `dropouts` environment variables but must match the checkpoint.

## Example

```bash
export MODEL_WEIGHTS_PATH=/path/to/best_model_looperset_pact25_compact.pt
export TIRAMISU_ROOT=/path/to/tiramisu
printf '%s\n%s\n%s\n' "$PROGRAM_JSON" "$NO_SCHED_JSON" "$SCHED_JSON" | python main.py
```

The tensors this interface builds are identical to those produced by the training feature
extractor, so the checkpoint scores schedules exactly as it was trained to.
