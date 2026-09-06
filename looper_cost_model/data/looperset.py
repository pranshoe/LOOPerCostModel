"""LOOPerSet (v2) schema adapter.

This module is the single place that understands the LOOPerSet record format
(https://huggingface.co/datasets/Mascinissa/LOOPerSet) and translates it into the
structures the feature extractor consumes. It provides:

  * ``stream_jsonl_gz``     -- memory-friendly streaming reader for ``*.jsonl.gz``.
  * ``build_program_tree``  -- build the loop-nest tree from ``program_annotation``.
  * ``normalize_schedule``  -- turn one ``schedules_list`` entry (a flat, typed
                               ``transformations_list``) into the per-computation schedule
                               structure used to build the model input.
  * ``schedule_to_str``     -- reconstruct the standardized ``schedule_str`` (a parsing
                               self-check against the dataset's own string).

Key properties of the format:

  * ``transformations_list`` is in true application order; each transformation's
    ``loop_levels`` are expressed in the loop frame at the moment it is applied.
  * Skewing parameters ``(a, b)`` define the 2x2 submatrix ``[[a, b], [c, d]]`` where
    ``(c, d)`` is the unimodular completion computed by ``linear_diophantine_default``.
  * The speedup label is ``initial_execution_time / min(execution_times)``.
"""

import gzip
import json

import numpy as np

# Transformation-tag layout shared with the model (see ``constants`` / ``data.features``).
from ..constants import MAX_TAGS

# Canonical loop-transformation type codes used inside the affine tag vector.
_TAG_INTERCHANGE = 1
_TAG_REVERSAL = 2
_TAG_SKEWING = 3

# Transformation "type" strings as they appear in LOOPerSet transformations_list.
AFFINE_TYPES = ("interchange", "reversal", "skewing")


class ScheduleTooComplexException(Exception):
    """Raised for the rare schedules the fixed model input cannot represent
    (e.g. a computation tiled more than once, or a transformation targeting a
    tiling-created sub-loop). Such schedules are skipped during extraction."""
    pass


def stream_jsonl_gz(file_path):
    """Yield one program dict at a time from a gzipped JSON-Lines file."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def linear_diophantine_default(f_i, f_j):
    """Basic solution (gamma, sigma) of f_i*sigma - f_j*gamma = 1.

    Mirrors utils.data_utils.linear_diophantine_default; used to complete a skewing
    2x2 submatrix to a unimodular (det = 1) matrix from its first row (f_i, f_j).
    """
    n1, n2 = abs(f_i), abs(f_j)
    while n1 != n2:
        if n1 > n2:
            n1 -= n2
        else:
            n2 -= n1
    f_i = f_i / n1
    f_j = f_j / n1
    gamma, sigma = 0, 1
    if (f_j == 1) or (f_i == 1):
        gamma = f_i - 1
        sigma = 1
    elif (f_j == -1) and (f_i > 1):
        gamma = 1
        sigma = 0
    else:
        found = False
        i = 0
        while (i < 100) and (not found):
            if ((sigma * f_i) % abs(f_j)) == 1:
                found = True
            else:
                sigma += 1
                i += 1
        if not found:
            raise ValueError(f"No diophantine solution for skewing factors ({f_i}, {f_j})")
        gamma = ((sigma * f_i) - 1) / f_j
    return gamma, sigma


def skewing_submatrix(a, b):
    """Return the 2x2 skewing submatrix (as flat (m00, m01, m10, m11)) for factors (a, b)."""
    gamma, sigma = linear_diophantine_default(a, b)
    return a, b, int(gamma), int(sigma)


def build_program_tree(program_annotation):
    """Rebuild the untransformed loop-nest tree from the iterators hierarchy.

    Returns a dict of the same shape the model expects:
        {"roots": [ {loop_name, computations_list, child_list}, ... ]}
    """
    iterators = program_annotation["iterators"]

    def make_node(loop_name):
        node = iterators[loop_name]
        return {
            "loop_name": loop_name,
            "computations_list": list(node.get("computations_list", [])),
            "child_list": [make_node(child) for child in node.get("child_iterators", [])],
        }

    roots = [name for name, node in iterators.items() if node.get("parent_iterator") is None]
    return {"roots": [make_node(name) for name in roots]}


def _is_virtual_loop(name):
    """True if ``name`` is a tiling-created sub-loop (not an original program iterator)."""
    return name.endswith("_outer") or name.endswith("_inner")


def _affine_tag(transformation):
    """Convert one affine transformation dict into a fixed-size 16-element tag vector."""
    tag = [0] * MAX_TAGS
    ttype = transformation["type"]
    levels = transformation["loop_levels"]
    if ttype == "interchange":
        tag[0] = _TAG_INTERCHANGE
        tag[1], tag[2] = levels[0], levels[1]
    elif ttype == "reversal":
        tag[0] = _TAG_REVERSAL
        tag[3] = levels[0]
    elif ttype == "skewing":
        tag[0] = _TAG_SKEWING
        tag[4], tag[5] = levels[0], levels[1]
        m00, m01, m10, m11 = skewing_submatrix(transformation["parameters"][0],
                                               transformation["parameters"][1])
        tag[7], tag[8], tag[9], tag[10] = m00, m01, m10, m11
    else:
        raise ValueError(f"Not an affine transformation: {ttype}")
    return tag


def normalize_schedule(program_annotation, schedule):
    """Normalize one LOOPerSet schedule into the per-computation structure the feature
    extractor consumes.

    Returns a dict::

        {
          <comp_name>: {
              "transformations_list": [ <16-elem affine tag>, ... ],  # in application order
              "parallelized_dim": <iterator name> | None,
              "tiling": {} | {"tiling_depth", "tiling_dims"(names), "tiling_factors"},
              "unrolling_factor": <int> | None,
              "shiftings": [[<iterator name>, factor], ...] | None,
          },
          ...
          "fusions": [[compA, compB, level], ...] | None,
          "transformed_iterators": {<comp_name>: [iterator names after fusion], ...},
          "execution_times": [...],
        }

    Loop levels for parallelization / tiling / shifting are resolved back to the
    program's declared iterator names by replaying the loop-nest reordering (interchange)
    that precedes them.
    """
    computations = program_annotation["computations"]
    comp_names = list(computations.keys())

    # Per-comp accumulators, seeded to the "no transformation" state.
    result = {
        name: {
            "transformations_list": [],
            "parallelized_dim": None,
            "tiling": {},
            "unrolling_factor": None,
            "shiftings": None,
        }
        for name in comp_names
    }
    fusions = []

    # A working loop nest per computation, used to resolve loop levels -> iterator names
    # as transformations are applied (interchange permutes it; tiling splits it).
    working_nest = {name: list(computations[name]["iterators"]) for name in comp_names}

    for transformation in schedule.get("transformations_list", []):
        ttype = transformation["type"]
        levels = transformation["loop_levels"]
        params = transformation["parameters"]
        # LOOPerSet occasionally repeats a computation in the affected list; a
        # transformation applies to each computation once, so de-duplicate (order-preserving).
        affected = list(dict.fromkeys(transformation["computations"]))

        if ttype == "fusion":
            # Fusion merges two computations at the given loop level. Record it once and
            # make the second computation adopt the first computation's loop nest.
            comp_a, comp_b = affected[0], affected[1]
            fusions.append([comp_a, comp_b, levels[0]])
            working_nest[comp_b] = list(working_nest[comp_a])
            continue

        for comp in affected:
            nest = working_nest[comp]
            # A transformation may list computations whose (possibly shallower) loop nest
            # does not contain the targeted loop levels -- e.g. a skewing of L1,L2 listed
            # for a 2-deep computation. Such a computation cannot undergo the
            # transformation at those levels, so skip it (physically a no-op for it).
            if levels and max(levels) >= len(nest):
                continue
            if ttype in AFFINE_TYPES:
                result[comp]["transformations_list"].append(_affine_tag(transformation))
                if ttype == "interchange":
                    i, j = levels
                    nest[i], nest[j] = nest[j], nest[i]
                # reversal / skewing do not reorder the loop nest.
            elif ttype == "parallelization":
                dim = nest[levels[0]]
                if _is_virtual_loop(dim):
                    raise ScheduleTooComplexException
                result[comp]["parallelized_dim"] = dim
            elif ttype == "shifting":
                dim = nest[levels[0]]
                if _is_virtual_loop(dim):
                    raise ScheduleTooComplexException
                shift = [dim, params[0] if params else 0]
                if result[comp]["shiftings"] is None:
                    result[comp]["shiftings"] = []
                result[comp]["shiftings"].append(shift)
            elif ttype == "tiling":
                dims = [nest[l] for l in levels]
                # The model represents at most one tiling per computation over original
                # loops; a second tiling or a tiling of already-tiled sub-loops cannot be
                # represented, so drop these (rare) schedules.
                if result[comp]["tiling"] or any(_is_virtual_loop(d) for d in dims):
                    raise ScheduleTooComplexException
                result[comp]["tiling"] = {
                    "tiling_depth": len(levels),
                    "tiling_dims": dims,
                    "tiling_factors": [int(p) for p in params],
                }
                # Grow the working nest so any later transformation sees tiled sub-loops.
                _apply_tiling_to_nest(nest, dims)
            elif ttype == "unrolling":
                result[comp]["unrolling_factor"] = int(params[0])
            else:
                raise ValueError(f"Unknown transformation type: {ttype}")

    result["fusions"] = fusions or None
    result["transformed_iterators"] = {name: working_nest_original(program_annotation, name, fusions)
                                       for name in comp_names}
    result["execution_times"] = schedule.get("execution_times")
    return result


def working_nest_original(program_annotation, comp_name, fusions):
    """Return the (fusion-adjusted) list of original iterator names for a computation.

    Fusion makes a computation share the loop nest of the computation it was fused into,
    which can drop some of its own iterators -- the feature extractor uses this to zero
    out the removed loops.
    """
    if fusions:
        for comp_a, comp_b, _level in fusions:
            if comp_name == comp_b:
                return list(program_annotation["computations"][comp_a]["iterators"])
    return list(program_annotation["computations"][comp_name]["iterators"])


def _apply_tiling_to_nest(nest, dims):
    """Replace tiled dims in ``nest`` with outer/inner virtual loops (in place),
    mirroring the loop-nest bookkeeping of the standardized schedule string."""
    if len(dims) == 2:
        d0, d1 = dims
        i = nest.index(d0)
        nest[i:i + 1] = [d0 + "_outer", d1 + "_outer"]
        i = nest.index(d1)
        nest[i:i + 1] = [d0 + "_inner", d1 + "_inner"]
    elif len(dims) == 3:
        d0, d1, d2 = dims
        i = nest.index(d0)
        nest[i:i + 1] = [d0 + "_outer", d1 + "_outer", d2 + "_outer"]
        i = nest.index(d1)
        nest[i:i + 1] = [d0 + "_inner", d1 + "_inner", d2 + "_inner"]
        nest.remove(d2)


# --- schedule_str reconstruction (self-check only) --------------------------------

def _comps_suffix(comps):
    return "comps=" + str(list(comps))


def schedule_to_str(schedule):
    """Rebuild the standardized ``schedule_str`` from a raw LOOPerSet schedule entry.

    Used purely to validate that our understanding of the schema matches the dataset:
    ``schedule_to_str(s) == s["schedule_str"]`` should hold for every record.
    """
    parts = []
    for t in schedule.get("transformations_list", []):
        ty = t["type"]
        ll = t["loop_levels"]
        pr = t["parameters"]
        c = _comps_suffix(t["computations"])
        if ty == "interchange":
            parts.append(f"I(L{ll[0]},L{ll[1]},{c})")
        elif ty == "reversal":
            parts.append(f"R(L{ll[0]},{c})")
        elif ty == "skewing":
            parts.append(f"S(L{ll[0]},L{ll[1]},{pr[0]},{pr[1]},{c})")
        elif ty == "parallelization":
            parts.append(f"P(L{ll[0]},{c})")
        elif ty == "tiling":
            if len(ll) == 2:
                parts.append(f"T2(L{ll[0]},L{ll[1]},{pr[0]},{pr[1]},{c})")
            else:
                parts.append(f"T3(L{ll[0]},L{ll[1]},L{ll[2]},{pr[0]},{pr[1]},{pr[2]},{c})")
        elif ty == "unrolling":
            parts.append(f"U(L{ll[0]},{pr[0]},{c})")
        elif ty == "fusion":
            # Fusion uses a bare, unquoted comps list in the standardized string.
            parts.append(f"F(L{ll[0]},comps=[{','.join(t['computations'])}])")
        elif ty == "shifting":
            parts.append(f"Sh(L{ll[0]},{pr[0] if pr else 0},{c})")
        else:
            raise ValueError(f"Unknown transformation type: {ty}")
    return "|".join(parts)
