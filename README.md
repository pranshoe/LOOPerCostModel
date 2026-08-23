# LOOPerCostModel

LOOPerCostModel is the learned cost model of the [LOOPer](https://ieeexplore.ieee.org/document/11282943)
polyhedral autoscheduler. Given a program and a candidate sequence of loop
transformations (a *schedule*), it predicts the **speedup** that schedule would achieve,
allowing the autoscheduler's search to rank transformations without compiling and running
each candidate.

The model is a tree-structured neural network that mirrors a program's loop nest. It reads
two kinds of features:

- **Program features** — the untransformed program's structure: the loop-nest tree, each
  computation's iteration domain (as an affine-constraint matrix), its memory-access
  matrices, and its arithmetic expression tree.
- **Schedule features** — the applied transformations: interchange, reversal, skewing,
  tiling, unrolling, parallelization, shifting, and fusion.

These are embedded bottom-up — sequences of affine transformations and expression
operations by LSTMs, then computations and loops by a recursive network that follows the
loop-nest hierarchy — and regressed to a single predicted speedup. The model is trained to
minimize mean absolute percentage error (MAPE) against measured speedups, defined as
`initial_execution_time / min(execution_times)`.

It trains directly on the [LOOPerSet](https://huggingface.co/datasets/Mascinissa/LOOPerSet)
dataset with no manual preprocessing.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. A GPU is optional — the code runs unchanged on
CPU or GPU (see [Configuration](#configuration)).

## Data

LOOPerCostModel reads LOOPerSet's gzipped JSON-Lines splits directly. Download the
`pact25` compact split (the one used to train the published model) from the
[dataset card](https://huggingface.co/datasets/Mascinissa/LOOPerSet):

```python
from huggingface_hub import hf_hub_download
for name in ["looperset_v2_pact_train_compact.jsonl.gz",
             "looperset_v2_pact_validation_compact.jsonl.gz"]:
    hf_hub_download(repo_id="Mascinissa/LOOPerSet", repo_type="dataset",
                    filename=f"data/pact25/{name}", revision="v2.1",
                    local_dir="LOOPerSet")
```

Point `data.train_file` / `data.valid_file` at any LOOPerSet split (e.g. the larger `full`
split, or the standard variants that retain raw source and IR).

## Usage

Every command reads `configs/default.yaml` and accepts `key=value` overrides on the command
line. All artifacts — the feature cache, batched tensors, checkpoints, and logs — are
written under `experiment.base_path`.

Prepare the batched dataset (feature extraction + batching; run once per split):

```bash
python -m looper_cost_model.prepare_data --config configs/default.yaml
```

Train (any split not yet prepared is prepared automatically):

```bash
python -m looper_cost_model.train --config configs/default.yaml training.device=cuda:0
```

Evaluate a checkpoint on the validation split:

```bash
python -m looper_cost_model.evaluate --config configs/default.yaml \
    eval.weights_path runs/pact25/weights/best_model_looperset_pact25_compact.pt
```

Predict schedule speedups for the programs in a LOOPerSet file:

```bash
python -m looper_cost_model.predict --config configs/default.yaml \
    eval.weights_path <checkpoint.pt> data.valid_file <programs.jsonl.gz>
```

### A quick run on CPU

A 60-program slice is bundled under `.testdata/` for a fast local check:

```bash
python -m looper_cost_model.train \
    experiment.base_path=./runs/mini \
    data.train_file=.testdata/mini.jsonl.gz \
    data.valid_file=.testdata/mini.jsonl.gz \
    data.batch_size=256 data.min_functions_per_footprint=0 \
    training.device=cpu training.max_epochs=5
```

## Configuration

Configuration is a set of typed dataclasses (see `looper_cost_model/config.py`) with
defaults in `configs/default.yaml`, layered as *defaults → `--config` file → CLI
overrides*. The most useful options:

| Key | Description |
| --- | --- |
| `experiment.base_path` | Output root for cache, batches, checkpoints, and logs |
| `data.train_file`, `data.valid_file` | LOOPerSet `*.jsonl.gz` splits |
| `data.batch_size` | Datapoints per batch |
| `data.nb_workers` | Parallel feature-extraction workers |
| `data.min_functions_per_footprint` | Drop rare loop-nest shapes with too few programs (`0` keeps all) |
| `training.device` | `cpu` or `cuda:N` — the only device switch |
| `training.lr`, `training.max_epochs` | Constant learning rate and epoch count |

**Device handling.** `training.device` is the single switch; there is no separate GPU code
path. Cached tensors are stored on CPU and, at load time, as many batches as fit within a
GPU-memory budget are placed on the device while the rest stream from host memory. Moving a
workflow from CPU to GPU requires no code change.

## Project layout

```
looper_cost_model/
  constants.py        Fixed input dimensions and drop-condition exceptions
  config.py           Typed configuration (YAML + CLI overrides)
  data/
    looperset.py      LOOPerSet schema adapter: streaming, tree building, schedule normalization
    features.py       Program + schedule -> fixed-size input tensors
    dataset.py        Parallel feature extraction, caching, and batching
  model.py            The recursive-LSTM cost model
  train.py            Training loop (MAPE objective, constant LR, checkpointing)
  evaluate.py         Per-program metrics (MAPE, Spearman, nDCG)
  predict.py          Schedule-speedup inference
  prepare_data.py     Offline dataset preparation
configs/default.yaml  Default configuration
tests/test_smoke.py   End-to-end smoke test on the bundled slice
```

## Expected results

Trained on the full `pact25` split (100 epochs on an A100 80 GB, ~7 hours), the model
reaches **27.8% MAPE**, **0.77 Spearman rank correlation**, and **0.96 nDCG** on the
held-out validation programs, matching the numbers reported in the LOOPer paper. Short
runs on small slices reproduce the trends — decreasing validation loss and nDCG well above
a constant baseline — but not the absolute figures.

## Citation

If you use this cost model, please cite the LOOPer paper and the LOOPerSet dataset:

```bibtex
@INPROCEEDINGS{looper,
  author={Merouani, Massinissa and Boudaoud, Afif and Aouadj, Iheb Nassim and Tchoulak, Nassim and Bernou, Islem Kara and Benyamina, Hamza and Tayeb, Fatima Benbouzid-Si and Benatchba, Karima and Leather, Hugh and Baghdadi, Riyadh},
  booktitle={2025 34th International Conference on Parallel Architectures and Compilation Techniques (PACT)},
  title={LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers},
  year={2025}, pages={201-215}, doi={10.1109/PACT65351.2025.00028}}

@misc{looperset,
  title={LOOPerSet: A Large-Scale Dataset for Data-Driven Polyhedral Compiler Optimization},
  author={Massinissa Merouani and Afif Boudaoud and Riyadh Baghdadi},
  year={2025}, eprint={2510.10209}, archivePrefix={arXiv}, primaryClass={cs.PL},
  url={https://arxiv.org/abs/2510.10209}}
```

## License

Released under the MIT License (see `LICENSE`). The LOOPerSet dataset is distributed under
CC-BY-4.0.
