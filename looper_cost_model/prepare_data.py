"""Offline data preparation for LOOPerCostModel.

Extracts features and builds batched tensors for the train and validation splits, writing
a feature cache and batched artifacts under ``experiment.base_path``. Run this once before
training on a large split (it is the memory- and time-intensive step); training will also
prepare any missing split automatically.

    python -m looper_cost_model.prepare_data --config configs/default.yaml
"""

import os

from .config import load_config
from .data.dataset import prepare_dataset


def run(cfg):
    base = cfg.experiment.base_path
    batched_dir = os.path.join(base, "batched")
    for split, dataset_file in (("train", cfg.data.train_file), ("valid", cfg.data.valid_file)):
        print(f"=== Preparing '{split}' split from {dataset_file} ===")
        prepare_dataset(
            dataset_file,
            cache_dir=os.path.join(base, "cache", split),
            batched_dir=batched_dir,
            split_name=split,
            batch_size=cfg.data.batch_size,
            nb_workers=cfg.data.nb_workers,
            min_functions_per_footprint=cfg.data.min_functions_per_footprint,
            seed=cfg.training.seed,
        )
    print(f"Batched data written to {batched_dir}")


def main():
    run(load_config())


if __name__ == "__main__":
    main()
