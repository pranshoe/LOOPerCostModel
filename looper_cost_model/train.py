"""Training entry point for LOOPerCostModel.

Loads (or prepares) the batched LOOPerSet data, trains the recursive-LSTM regressor with
a MAPE objective and a constant learning rate, and checkpoints the best model by
validation MAPE.

    python -m looper_cost_model.train --config configs/default.yaml
"""

import copy
import logging
import math
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from .config import load_config
from .data.dataset import load_batches, prepare_dataset
from .model import LOOPerCostModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mape_criterion(inputs, targets):
    """Mean absolute percentage error (in percent)."""
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))


def _move_inputs(inputs, device):
    tree = inputs[0]
    return (tree,) + tuple(t.to(device) for t in inputs[1:])


def train(model, batches, num_epochs, lr, weight_decay, device, batch_size,
          checkpoint_path, log_every=1, logger=None, wandb_run=None):
    """Train ``model`` on ``batches['train']`` / ``batches['val']``.

    Returns ``(history, best_state_dict)`` where ``history`` is a list of
    ``(train_mape, val_mape)`` per epoch. The best checkpoint (lowest validation MAPE) is
    saved to ``checkpoint_path``.
    """
    logger = logger or logging.getLogger(__name__)
    device = torch.device(device)
    model = model.to(device)
    # A small constant learning rate (no schedule) reproduces the published cost model:
    # ~28% MAPE, 0.77 Spearman and 0.96 nDCG on the pact25 validation split.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_datapoints = {
        phase: sum(y.shape[0] for y in prepared.Y) for phase, prepared in batches.items()
    }

    history = []
    best_mape = math.inf
    best_state = None
    start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_mape = 0.0
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss = 0.0
            pbar = tqdm(batches[phase], desc=f"epoch {epoch + 1} [{phase}]")
            for inputs, labels in pbar:
                inputs = _move_inputs(inputs, device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    assert outputs.shape == labels.shape
                    batch_mape = mape_criterion(outputs, labels)
                    if phase == "train":
                        # Size-weight the gradient so partial batches count proportionally.
                        (batch_mape * labels.shape[0] / batch_size).backward()
                        optimizer.step()
                running_loss += batch_mape.item() * labels.shape[0]
                pbar.set_description(f"epoch {epoch + 1} [{phase}] MAPE {batch_mape.item():.3f}")

            epoch_mape = running_loss / n_datapoints[phase]
            if phase == "train":
                train_mape = epoch_mape
            else:
                history.append((train_mape, epoch_mape))
                if epoch_mape <= best_mape:
                    best_mape = epoch_mape
                    best_state = copy.deepcopy(model.state_dict())
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(best_state, checkpoint_path)
                    logger.debug(f"Saved checkpoint to {checkpoint_path}")
                if wandb_run is not None:
                    wandb_run.log({
                        "train_mape": train_mape, "val_mape": epoch_mape,
                        "best_mape": best_mape, "epoch": epoch,
                    })
                message = (
                    f"Epoch {epoch + 1}/{num_epochs}: train MAPE {train_mape:.4f}  "
                    f"val MAPE {epoch_mape:.4f}  best {best_mape:.4f}  "
                    f"time {time.time() - epoch_start:.1f}s"
                )
                print(message)
                if epoch % log_every == 0:
                    logger.info(message)

    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed / 60:.1f}m; best val MAPE {best_mape:.4f}")
    print(f"Training complete in {elapsed / 60:.1f}m; best val MAPE {best_mape:.4f}")
    return history, best_state


def run(cfg):
    base = cfg.experiment.base_path
    cache_dir = os.path.join(base, "cache")
    batched_dir = os.path.join(base, "batched")
    weights_dir = os.path.join(base, "weights")
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logs_dir, f"{cfg.experiment.name}.log"),
        filemode="a", level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment {cfg.experiment.name}")

    set_seed(cfg.training.seed)

    # Prepare batched data if it is not already on disk.
    for split, dataset_file in (("train", cfg.data.train_file), ("valid", cfg.data.valid_file)):
        meta = os.path.join(batched_dir, f"{split}_meta.json")
        if not os.path.exists(meta):
            print(f"Preparing '{split}' split from {dataset_file}")
            prepare_dataset(
                dataset_file,
                cache_dir=os.path.join(cache_dir, split),
                batched_dir=batched_dir,
                split_name=split,
                batch_size=cfg.data.batch_size,
                nb_workers=cfg.data.nb_workers,
                min_functions_per_footprint=cfg.data.min_functions_per_footprint,
                seed=cfg.training.seed,
            )

    print("Loading batches...")
    batches = {
        "train": load_batches(batched_dir, "train", device=cfg.training.device,
                              shuffle=True, seed=cfg.training.seed),
        "val": load_batches(batched_dir, "valid", device=cfg.training.device),
    }

    model = LOOPerCostModel(
        input_size=cfg.model.input_size,
        comp_embed_layer_sizes=cfg.model.comp_embed_layer_sizes,
        drops=cfg.model.drops,
    )
    if cfg.training.continue_from:
        print(f"Resuming from {cfg.training.continue_from}")
        model.load_state_dict(torch.load(cfg.training.continue_from, map_location=cfg.training.device))

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(name=cfg.experiment.name, project=cfg.wandb.project)
        wandb.watch(model)

    checkpoint_path = os.path.join(weights_dir, f"best_model_{cfg.experiment.name}.pt")
    train(
        model, batches,
        num_epochs=cfg.training.max_epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        device=cfg.training.device,
        batch_size=cfg.data.batch_size,
        checkpoint_path=checkpoint_path,
        log_every=cfg.training.log_every,
        logger=logger,
        wandb_run=wandb_run,
    )
    print(f"Best checkpoint: {checkpoint_path}")


def main():
    run(load_config())


if __name__ == "__main__":
    main()
