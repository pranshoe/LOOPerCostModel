"""Dataset pipeline: LOOPerSet records -> cached feature shards -> batched tensors.

Three clearly separated stages, each resumable from disk:

1. :func:`extract_features` -- parallel workers stream the LOOPerSet ``*.jsonl.gz``,
   extract the per-(program, schedule) tensors, and write them to numbered pickle
   *shards* in a cache directory. Workers flush incrementally so peak memory stays
   bounded regardless of split size.
2. :func:`build_batches` -- reads the shards, groups datapoints by program
   *tree footprint* (so each batch has a uniform tensor shape for the recursive model),
   and produces fixed-size batches of ``(inputs, labels)``. All tensors are built on
   CPU; the choice of compute device is deferred to load time.
3. :func:`save_batches` / :func:`load_batches` -- persist batches as CPU shards and,
   at load time, move as many batches onto the compute device as fit within a memory
   budget (the rest stream from CPU). On a CPU-only run everything simply stays on CPU.

This deferral of device placement is what keeps the pipeline device-agnostic: CPU vs
GPU is a runtime choice, not a property baked into the cached artifacts.
"""

import gc
import json
import multiprocessing
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..constants import (
    ExpressionTooLongException,
    LoopsDepthException,
    NbAccessException,
    NbTransformationException,
)
from . import features as F
from .looperset import ScheduleTooComplexException, normalize_schedule, stream_jsonl_gz

# Number of functions each worker buffers before flushing a shard to disk.
_FLUSH_EVERY = 500
# Fraction of GPU memory to fill with resident batches before spilling to CPU. Kept well
# below 1.0 so the remainder is available for the forward/backward pass; when a split is
# larger than GPU memory the un-resident batches simply stream from CPU RAM per step.
_GPU_MEMORY_FRACTION = 0.6
# Batches per saved shard file.
_BATCHES_PER_SHARD = 256


# --------------------------------------------------------------------------------------
# Stage 1: parallel feature extraction -> shards
# --------------------------------------------------------------------------------------

def _extract_worker(input_q, output_q):
    """Worker: extract the representation for a chunk of programs, flushing pickle shards.

    Each shard is a list of ``(function_name, nb_dropped, tree_footprint,
    function_data)`` tuples, where ``function_data`` holds the per-schedule tensors.
    """
    worker_id, programs_dict, cache_dir = input_q.get()
    buffer = []
    shard_count = 0

    def flush():
        nonlocal shard_count, buffer
        if not buffer:
            return
        path = Path(cache_dir) / f"shard_{worker_id}_{shard_count}.pkl"
        with open(path, "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_count += 1
        buffer = []
        gc.collect()

    for function_name in tqdm(list(programs_dict.keys()), desc=f"worker {worker_id}"):
        program = programs_dict[function_name]
        if F.drop_program(program):
            continue

        program_json = program["program_annotation"]
        try:
            template = F.build_representation_template(program)
        except (LoopsDepthException, NbAccessException, ExpressionTooLongException):
            continue

        init_exec_time = program["initial_execution_time"]
        if not init_exec_time or init_exec_time <= 0:
            continue

        footprint = F.tree_footprint(template.tree)
        function_data = {
            "tree": template.tree,
            "comps_tensor_list": [],
            "loops_tensor_list": [],
            "comps_expr_tree_list": [],
            "datapoint_attributes_list": [],
            "speedups_list": [],
        }
        nb_dropped = 0

        for schedule_index, raw_schedule in enumerate(program["schedules_list"]):
            if F.drop_schedule(raw_schedule):
                nb_dropped += 1
                continue
            speedup = F.speedup_clip(init_exec_time / np.min(raw_schedule["execution_times"]))
            try:
                schedule_json = normalize_schedule(program_json, raw_schedule)
                comps_tensor, loops_tensor, expr_tensor = F.fill_schedule_representation(
                    program_json, schedule_json, template
                )
            except (NbTransformationException, ScheduleTooComplexException):
                nb_dropped += 1
                continue

            function_data["comps_tensor_list"].append(comps_tensor)
            function_data["loops_tensor_list"].append(loops_tensor)
            function_data["comps_expr_tree_list"].append(expr_tensor)
            function_data["speedups_list"].append(speedup)
            function_data["datapoint_attributes_list"].append(
                F.datapoint_attributes(function_name, program, schedule_index, footprint)
            )

        buffer.append((function_name, nb_dropped, footprint, function_data))
        if len(buffer) >= _FLUSH_EVERY:
            flush()

    flush()
    output_q.put(worker_id)


def _load_programs(dataset_file):
    """Stream a LOOPerSet split into ``{program_name: record}``."""
    programs = {}
    for program in stream_jsonl_gz(dataset_file):
        programs[program["program_name"]] = program
    return programs


def extract_features(dataset_file, cache_dir, nb_workers=8, seed=42):
    """Extract features for a split into pickle shards under ``cache_dir``.

    If ``cache_dir`` already contains shards, extraction is skipped (idempotent /
    resumable). Uses ``nb_workers`` processes; the parent frees each program record as it
    is dispatched so it never holds the whole raw split alongside the workers' output.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(cache_dir.glob("shard_*.pkl")):
        print(f"Feature cache already present in {cache_dir}; skipping extraction.")
        return

    programs_dict = _load_programs(dataset_file)
    function_list = list(programs_dict.keys())
    random.Random(seed).shuffle(function_list)
    chunk_size = (len(function_list) // nb_workers) + 1
    print(f"Extracting {len(function_list)} programs with {nb_workers} workers "
          f"(~{chunk_size} programs/worker).")

    # Use spawn so workers start with a fresh address space and receive their program
    # chunk only via the queue -- with fork they would copy-on-write inherit the entire
    # raw split (tens of GB) held by the parent at process-start time.
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()
    workers = [
        ctx.Process(target=_extract_worker, args=(input_queue, output_queue))
        for _ in range(nb_workers)
    ]
    for w in workers:
        w.start()

    for i in range(nb_workers):
        names = function_list[i * chunk_size: (i + 1) * chunk_size]
        chunk = {name: programs_dict.pop(name) for name in names}
        input_queue.put((i, chunk, str(cache_dir)))
        del chunk

    del programs_dict, function_list
    gc.collect()

    for _ in range(nb_workers):
        output_queue.get()
    for w in workers:
        w.join()


# --------------------------------------------------------------------------------------
# Stage 2: shards -> batches (grouped by tree footprint)
# --------------------------------------------------------------------------------------

@dataclass
class PreparedBatches:
    """Batched inputs/labels plus per-batch datapoint attributes and drop statistics."""

    X: list = field(default_factory=list)
    Y: list = field(default_factory=list)
    attributes: list = field(default_factory=list)
    nb_datapoints: int = 0
    nb_dropped: int = 0

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def _iter_shards(cache_dir):
    for shard_path in sorted(Path(cache_dir).glob("shard_*.pkl")):
        with open(shard_path, "rb") as f:
            yield from pickle.load(f)


def build_batches(cache_dir, batch_size=2048, min_functions_per_footprint=0, seed=None):
    """Group cached datapoints by tree footprint and split them into fixed-size batches.

    Footprints with fewer than ``min_functions_per_footprint`` functions (and fewer than
    ``100 * min_functions_per_footprint`` datapoints) are dropped to avoid tiny,
    low-diversity batches. All tensors are produced on CPU.
    """
    rng = random.Random(seed)

    # Regroup datapoints by footprint.
    groups = {}
    footprint_stats = {}
    nb_datapoints = nb_dropped = 0
    print("Assembling datapoints by tree footprint...")
    for function_name, fn_dropped, footprint, function_data in _iter_shards(cache_dir):
        group = groups.setdefault(footprint, {
            "tree": function_data["tree"],
            "comps_tensor_list": [], "loops_tensor_list": [],
            "comps_expr_tree_list": [], "datapoint_attributes_list": [], "speedups_list": [],
        })
        for key in ("comps_tensor_list", "loops_tensor_list", "comps_expr_tree_list",
                    "datapoint_attributes_list", "speedups_list"):
            group[key].extend(function_data[key])

        n = len(function_data["speedups_list"])
        nb_dropped += fn_dropped
        nb_datapoints += n
        stats = footprint_stats.setdefault(footprint, {"nb_funcs": 0, "nb_dps": 0})
        stats["nb_funcs"] += 1
        stats["nb_dps"] += n

    prepared = PreparedBatches(nb_datapoints=nb_datapoints, nb_dropped=nb_dropped)

    print("Building batches...")
    for footprint, group in tqdm(groups.items()):
        stats = footprint_stats[footprint]
        if stats["nb_funcs"] < min_functions_per_footprint and \
                stats["nb_dps"] < 100 * min_functions_per_footprint:
            prepared.nb_datapoints -= stats["nb_dps"]
            continue

        for node in group["tree"]["roots"]:
            F.move_tree_to_device(node, "cpu")

        # Shuffle within the footprint so batches aren't dominated by one program.
        zipped = list(zip(
            group["datapoint_attributes_list"], group["comps_tensor_list"],
            group["comps_expr_tree_list"], group["loops_tensor_list"], group["speedups_list"],
        ))
        rng.shuffle(zipped)
        attributes, comps, exprs, loops, speedups = zip(*zipped)

        for start in range(0, len(speedups), batch_size):
            end = start + batch_size
            comps_batch = torch.cat(comps[start:end], 0)
            batch_n, num_comps, _ = comps_batch.shape
            first_part, vectors, third_part = F.split_computation_vector(
                comps_batch.view(batch_n * num_comps, -1)
            )
            prepared.X.append((
                group["tree"],
                first_part.view(batch_n, num_comps, -1),
                vectors,
                third_part.view(batch_n, num_comps, -1),
                torch.cat(loops[start:end], 0),
                torch.cat(exprs[start:end], 0),
            ))
            prepared.Y.append(torch.FloatTensor(speedups[start:end]))
            prepared.attributes.append(attributes[start:end])

    print(f"Datapoints: {prepared.nb_datapoints}  Batches: {len(prepared)}  "
          f"Dropped: {prepared.nb_dropped}")
    return prepared


# --------------------------------------------------------------------------------------
# Stage 3: save / load batches (device placement deferred to load time)
# --------------------------------------------------------------------------------------

def save_batches(prepared, out_dir, split_name):
    """Persist batches as CPU shards plus a small metadata file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = 0
    for shard_idx, start in enumerate(range(0, len(prepared), _BATCHES_PER_SHARD)):
        end = start + _BATCHES_PER_SHARD
        shard = {
            "X": prepared.X[start:end],
            "Y": prepared.Y[start:end],
            "attributes": prepared.attributes[start:end],
        }
        torch.save(shard, out_dir / f"{split_name}_shard_{shard_idx:04d}.pt")
        n_shards = shard_idx + 1
    meta = {
        "split_name": split_name, "n_shards": n_shards,
        "nb_datapoints": prepared.nb_datapoints, "nb_dropped": prepared.nb_dropped,
        "nb_batches": len(prepared),
    }
    with open(out_dir / f"{split_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(prepared)} batches to {out_dir} ({n_shards} shards).")


def _fits_on_gpu(device):
    if device.type != "cuda":
        return False
    allocated = torch.cuda.memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    return allocated / total < _GPU_MEMORY_FRACTION


def load_batches(out_dir, split_name, device="cpu", shuffle=False, seed=None):
    """Load saved batches, moving as many as fit onto ``device`` (rest stay on CPU).

    Returns a :class:`PreparedBatches`. On a CPU device everything stays on CPU. On a
    CUDA device, batches are moved to the GPU until the memory budget is reached, after
    which remaining batches are kept on CPU and moved per-iteration during training.
    """
    out_dir = Path(out_dir)
    device = torch.device(device)
    with open(out_dir / f"{split_name}_meta.json") as f:
        meta = json.load(f)

    prepared = PreparedBatches(
        nb_datapoints=meta["nb_datapoints"], nb_dropped=meta["nb_dropped"]
    )
    shard_paths = sorted(out_dir.glob(f"{split_name}_shard_*.pt"))
    spilled = False
    for shard_path in tqdm(shard_paths, desc=f"loading {split_name}"):
        # weights_only=False: these are our own trusted artifacts and carry Python/NumPy
        # datapoint metadata alongside the tensors, not just weights.
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for x, y, attrs in zip(shard["X"], shard["Y"], shard["attributes"]):
            target = device if (not spilled and _fits_on_gpu(device)) else torch.device("cpu")
            if target.type == "cuda":
                x = _batch_to_device(x, target)
                y = y.to(target)
            else:
                spilled = device.type == "cuda"
            prepared.X.append(x)
            prepared.Y.append(y)
            prepared.attributes.append(attrs)

    if shuffle:
        order = list(range(len(prepared)))
        random.Random(seed).shuffle(order)
        prepared.X = [prepared.X[i] for i in order]
        prepared.Y = [prepared.Y[i] for i in order]
        prepared.attributes = [prepared.attributes[i] for i in order]
    return prepared


def _batch_to_device(x, device):
    """Move a batch's input tuple to ``device`` (the tree at ``x[0]`` stays as-is; the
    model indexes its per-node tensors onto the device during the forward pass)."""
    tree, first_part, vectors, third_part, loops_tensor, expr_tensor = x
    return (
        tree,
        first_part.to(device), vectors.to(device), third_part.to(device),
        loops_tensor.to(device), expr_tensor.to(device),
    )


def prepare_dataset(dataset_file, cache_dir, batched_dir, split_name,
                    batch_size=2048, nb_workers=8, min_functions_per_footprint=0, seed=42):
    """End-to-end preparation of one split: extract -> batch -> save."""
    extract_features(dataset_file, cache_dir, nb_workers=nb_workers, seed=seed)
    prepared = build_batches(
        cache_dir, batch_size=batch_size,
        min_functions_per_footprint=min_functions_per_footprint, seed=seed,
    )
    save_batches(prepared, batched_dir, split_name)
    return prepared
