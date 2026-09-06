"""Evaluation for LOOPerCostModel.

Runs a trained model over a prepared split and reports the paper's cost-model metrics,
computed per program then averaged: MAPE, Spearman rank correlation, and nDCG (full,
@5, @1).

    python -m looper_cost_model.evaluate --config configs/default.yaml \
        eval.weights_path runs/pact25/weights/best_model_looperset_pact25_compact.pt
"""

import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from .config import load_config
from .data.dataset import load_batches, prepare_dataset
from .model import LOOPerCostModel


@torch.no_grad()
def predictions_dataframe(model, prepared, device):
    """Run ``model`` over ``prepared`` and return a per-datapoint dataframe with
    predictions, targets, absolute percentage error, and datapoint metadata."""
    model.eval()
    device = torch.device(device)
    rows = {k: [] for k in ("name", "sched_name", "sched_str", "exec_time",
                            "memory_use", "node_name", "tree_footprint")}
    all_preds, all_targets = [], []

    for batch_index, (inputs, labels) in enumerate(tqdm(prepared, desc="evaluating")):
        tree = inputs[0]
        inputs = (tree,) + tuple(t.to(device) for t in inputs[1:])
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_preds.append(outputs.cpu())
        all_targets.append(labels.cpu())

        attrs = list(zip(*prepared.attributes[batch_index]))
        rows["name"].extend(attrs[0])
        rows["sched_name"].extend(attrs[1])
        rows["sched_str"].extend(attrs[2])
        rows["exec_time"].extend(attrs[3])
        rows["memory_use"].extend(attrs[4])
        rows["node_name"].extend(attrs[5])
        rows["tree_footprint"].extend(attrs[6])

    preds = np.around(torch.cat(all_preds).numpy().reshape(-1), decimals=6)
    targets = np.around(torch.cat(all_targets).numpy().reshape(-1), decimals=6)
    df = pd.DataFrame(rows)
    df["memory_use"] = df["memory_use"].astype(float)
    df["prediction"] = preds
    df["target"] = targets
    df["APE"] = np.abs(df.target - df.prediction) / df.target * 100
    return df


def _ndcg(group, k):
    if len(group["target"]) < 2:
        return np.nan
    return ndcg_score([group["target"].tolist()], [group["prediction"].tolist()], k=k)


def _spearman(group):
    if len(group["target"]) < 2:
        return np.nan
    return spearmanr(group["target"], group["prediction"])[0]


def score(df):
    """Aggregate per-program metrics into a dataframe (one row per program)."""
    grouped = df.groupby("name")
    scores = pd.DataFrame({
        "count": grouped.size(),
        "nDCG": grouped.apply(lambda g: _ndcg(g, None)),
        "nDCG@5": grouped.apply(lambda g: _ndcg(g, 5)),
        "nDCG@1": grouped.apply(lambda g: _ndcg(g, 1)),
        "Spearman": grouped.apply(_spearman),
        "MAPE": grouped["APE"].mean(),
    }).reset_index()
    return scores


def summarize(scores):
    """Mean of each metric across programs."""
    return {
        metric: float(scores[metric].mean())
        for metric in ("MAPE", "Spearman", "nDCG", "nDCG@5", "nDCG@1")
    }


def evaluate(cfg, split="valid"):
    base = cfg.experiment.base_path
    batched_dir = os.path.join(base, "batched")
    if not os.path.exists(os.path.join(batched_dir, f"{split}_meta.json")):
        dataset_file = cfg.data.valid_file if split == "valid" else cfg.data.train_file
        prepare_dataset(
            dataset_file,
            cache_dir=os.path.join(base, "cache", split),
            batched_dir=batched_dir, split_name=split,
            batch_size=cfg.data.batch_size, nb_workers=cfg.data.nb_workers,
            min_functions_per_footprint=cfg.data.min_functions_per_footprint,
            seed=cfg.training.seed,
        )

    model = LOOPerCostModel(
        input_size=cfg.model.input_size,
        comp_embed_layer_sizes=cfg.model.comp_embed_layer_sizes,
        drops=cfg.model.drops,
    )
    weights_path = cfg.eval.weights_path
    if not weights_path:
        raise ValueError("eval.weights_path must point at a trained checkpoint.")
    model.load_state_dict(torch.load(weights_path, map_location=cfg.training.device))
    model = model.to(cfg.training.device)

    prepared = load_batches(batched_dir, split, device=cfg.training.device)
    df = predictions_dataframe(model, prepared, cfg.training.device)
    return summarize(score(df))


def main():
    cfg = load_config()
    metrics = evaluate(cfg)
    print("Evaluation metrics (mean over programs):")
    for name, value in metrics.items():
        print(f"  {name:10s} {value:.4f}")


if __name__ == "__main__":
    main()
