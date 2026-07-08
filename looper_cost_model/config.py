"""Typed configuration for LOOPerCostModel.

Configuration is a set of nested dataclasses with sensible defaults, layered as:

    defaults  <  YAML file (``--config``)  <  CLI dotlist overrides

Example::

    python -m looper_cost_model.train --config configs/default.yaml \
        training.device=cuda:0 data.batch_size=2048

The single ``training.device`` knob selects CPU or GPU; there is no separate CUDA code
path. Artifacts are written under ``experiment.base_path`` (feature cache, batched
tensors, checkpoints, logs), so no paths are hard-coded in the code.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf


@dataclass
class ExperimentConfig:
    name: str = "looperset_pact25_compact"
    # Root under which cache/, batched/, weights/ and logs/ are written.
    base_path: str = "./runs/pact25"


@dataclass
class DataConfig:
    # LOOPerSet gzipped JSON-Lines splits (paths relative to the working directory).
    # Defaults: the pact25 compact split, as downloaded by the README instructions.
    train_file: str = "LOOPerSet/data/pact25/looperset_v2_pact_train_compact.jsonl.gz"
    valid_file: str = "LOOPerSet/data/pact25/looperset_v2_pact_validation_compact.jsonl.gz"
    batch_size: int = 2048
    nb_workers: int = 8
    # Drop footprints with fewer than this many functions (0 = keep all; use 0 for small
    # samples and overfit tests).
    min_functions_per_footprint: int = 2


@dataclass
class ModelConfig:
    input_size: int = 846
    comp_embed_layer_sizes: List[int] = field(default_factory=lambda: [600, 350, 200, 180])
    drops: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.05])


@dataclass
class TrainingConfig:
    device: str = "cpu"
    lr: float = 0.00004               # constant learning rate
    weight_decay: float = 0.15e-1
    max_epochs: int = 100
    seed: int = 42
    # Optional checkpoint to resume from.
    continue_from: Optional[str] = None
    log_every: int = 1


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "looper_cost_model"


@dataclass
class EvalConfig:
    # Checkpoint to evaluate / predict with.
    weights_path: Optional[str] = None


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(argv=None):
    """Build a config from defaults, an optional ``--config <yaml>``, and CLI overrides.

    Any argument of the form ``key.subkey=value`` is applied as a dotlist override.
    Returns a validated OmegaConf object.
    """
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    base = OmegaConf.structured(Config)

    yaml_path = None
    overrides = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            yaml_path = argv[i + 1]
            i += 2
        elif argv[i].startswith("--config="):
            yaml_path = argv[i].split("=", 1)[1]
            i += 1
        else:
            overrides.append(argv[i])
            i += 1

    cfg = base
    if yaml_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(yaml_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
