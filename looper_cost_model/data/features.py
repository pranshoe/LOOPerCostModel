"""Cost-model input representation: turning a LOOPerSet program + schedule into tensors.

The model consumes, per (program, schedule) pair, a fixed-layout set of tensors:

* a **program tree** mirroring the loop-nest hierarchy (built once per program),
* a **per-computation vector** (``comps`` tensor) holding the untransformed program
  features (iteration domain, access matrices, expression) interleaved with the
  schedule features (which transformations were applied and their parameters),
* a **per-loop vector** (``loops`` tensor) with the same schedule tags at loop
  granularity,
* an **expression sequence** per computation.

Extraction is two-phase, matching the model's split between static and schedule
features:

1. :func:`build_representation_template` builds, once per program, a template of the
   per-computation / per-loop vectors in which the schedule-dependent slots are string
   placeholders. It also records, in ``placeholders``/``loop_placeholders``, the
   ``(row, col)`` position of every placeholder.
2. :func:`fill_schedule_representation` copies the template and overwrites the
   placeholders with the concrete schedule features, yielding the numeric tensors.

The layout constants live in :mod:`looper_cost_model.constants`.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import sympy
import torch

from ..constants import (
    MAX_ACCESSES,
    MAX_DEPTH,
    MAX_EXPR_LEN,
    MAX_NUM_TRANSFORMATIONS,
    MAX_TAGS,
    ExpressionTooLongException,
    LoopsDepthException,
    NbAccessException,
    NbTransformationException,
)
from .looperset import build_program_tree


# --------------------------------------------------------------------------------------
# Program tree
# --------------------------------------------------------------------------------------

def annotate_tree(node, loops_indices, comps_indices):
    """Attach ``loop_index`` / ``computations_indices`` tensors and a ``has_comps`` flag
    to every node of the loop-nest tree, so the model can index into the batched
    per-loop / per-computation tensors during its recursive forward pass. In place;
    returns ``node``. Tensors are built on CPU and moved to the store device at batch time.
    """
    if "roots" in node:
        for root in node["roots"]:
            annotate_tree(root, loops_indices, comps_indices)
        return node

    node["loop_index"] = torch.tensor(loops_indices[node["loop_name"]])
    if node["computations_list"]:
        node["computations_indices"] = torch.tensor(
            [comps_indices[name] for name in node["computations_list"]]
        )
        node["has_comps"] = True
    else:
        node["has_comps"] = False
    for child in node["child_list"]:
        annotate_tree(child, loops_indices, comps_indices)
    return node


def tree_footprint(tree):
    """Return a string uniquely encoding the tree's topology (loop indices + which
    computations sit at each level). Programs sharing a footprint have identical tensor
    shapes and are batched together so the recursive model can run one forward pass.
    """
    if "roots" in tree:
        footprint = ""
        for root in tree["roots"]:
            footprint += "<R>" + tree_footprint(root)
        return footprint + "</R>"
    footprint = "<L" + str(int(tree["loop_index"])) + ">"
    if tree["has_comps"]:
        footprint += "[" + "".join("C" + str(int(i)) for i in tree["computations_indices"]) + "]"
    for child in tree["child_list"]:
        footprint += tree_footprint(child)
    return footprint + "</L" + str(int(tree["loop_index"])) + ">"


def move_tree_to_device(node, device):
    """Move a program tree's index tensors to ``device`` in place."""
    node["loop_index"] = node["loop_index"].to(device, non_blocking=True)
    if "computations_indices" in node:
        node["computations_indices"] = node["computations_indices"].to(device, non_blocking=True)
    for child in node["child_list"]:
        move_tree_to_device(child, device)


# --------------------------------------------------------------------------------------
# Static program features -> template with schedule placeholders
# --------------------------------------------------------------------------------------

@dataclass
class RepresentationTemplate:
    """Per-program, schedule-independent extraction template.

    ``comps_templates`` / ``loops_templates`` are the per-computation / per-loop feature
    rows with schedule-dependent slots left as string placeholders; ``placeholders`` /
    ``loop_placeholders`` map each placeholder to its ``(row, col)`` position.
    ``initial_domains`` caches each computation's (schedule-independent) iteration-domain
    matrix so the ``sympy`` bound parsing runs once per program, not once per schedule;
    ``expr_repr`` similarly caches the (schedule-independent) padded expression sequences.
    """

    tree: dict
    comps_templates: list
    loops_templates: list
    placeholders: dict
    loop_placeholders: dict
    initial_domains: dict = field(default_factory=dict)
    expr_repr: list = field(default_factory=list)


def build_representation_template(program_dict):
    """Build the schedule-independent :class:`RepresentationTemplate` for one program.

    Schedule-dependent slots are left as string placeholders (filled per schedule by
    :func:`fill_schedule_representation`); schedule-independent slots -- notably the
    original iteration-domain constraints -- are filled here, once.

    Raises :class:`NbAccessException` / :class:`LoopsDepthException` for programs whose
    computations exceed the fixed access/depth limits.
    """
    program_json = program_dict["program_annotation"]
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        computations_dict.keys(),
        key=lambda name: computations_dict[name]["absolute_order"],
    )

    comps_templates = []
    placeholders = {}

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        if len(comp_dict["accesses"]) > MAX_ACCESSES:
            raise NbAccessException
        if len(comp_dict["iterators"]) > MAX_DEPTH:
            raise LoopsDepthException

        comp_repr = [+comp_dict["comp_is_reduction"]]

        # Per-loop schedule placeholders (parallelization, tiling, fusion, shifting),
        # padded up to MAX_DEPTH loops, then per-computation unrolling placeholders.
        c_code = "C" + str(comp_index)
        iterators_repr = []
        for iter_i in range(len(comp_dict["iterators"])):
            l_code = c_code + "-L" + str(iter_i)
            iterators_repr.extend([
                l_code + "Parallelized",
                l_code + "Tiled",
                l_code + "TileFactor",
                l_code + "Fused",
                l_code + "Shifted",
                l_code + "ShiftFactor",
            ])
        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        iterators_repr.extend([0] * iterator_repr_size * (MAX_DEPTH - len(comp_dict["iterators"])))
        iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

        # Affine transformation tags (interchange / reversal / skewing), fixed length.
        iterators_repr.append(c_code + "-TransformationTagsStart")
        iterators_repr.extend(["M"] * (MAX_TAGS * MAX_NUM_TRANSFORMATIONS - 2))
        iterators_repr.append(c_code + "-TransformationTagsEnd")

        # Original iteration-domain constraint matrix + constants vector.
        iterators_repr.append(c_code + "-OgConstraintMatrixStart")
        iterators_repr.extend(["OgC"] * ((MAX_DEPTH * MAX_DEPTH * 2) - 2))
        iterators_repr.append(c_code + "-OgConstraintMatrixEnd")
        iterators_repr.append(c_code + "-OgConstraintVectorStart")
        iterators_repr.extend(["V"] * (MAX_DEPTH * 2 - 2))
        iterators_repr.append(c_code + "-OgConstraintVectorEnd")

        # Transformed iteration-domain constraint matrix.
        iterators_repr.append(c_code + "-ConstraintMatrixStart")
        iterators_repr.extend(["C"] * ((MAX_DEPTH * MAX_DEPTH * 2) - 2))
        iterators_repr.append(c_code + "-ConstraintMatrixEnd")

        comp_repr.extend(iterators_repr)

        # Write access matrix (buffer id + padded access matrix).
        padded_write_matrix = pad_access_matrix(isl_to_write_matrix(comp_dict["write_access_relation"]))
        comp_repr.extend([comp_dict["write_buffer_id"] + 1] + padded_write_matrix.flatten().tolist())

        # Read access matrices, padded up to MAX_ACCESSES.
        read_accesses_repr = []
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(read_access_dict["access_matrix"])
            read_accesses_repr.extend(
                [+read_access_dict["access_is_reduction"], read_access_dict["buffer_id"] + 1]
                + read_access_matrix.flatten().tolist()
            )
        access_repr_len = (MAX_DEPTH + 1) * (MAX_DEPTH + 2) + 1 + 1
        read_accesses_repr.extend([0] * access_repr_len * (MAX_ACCESSES - len(comp_dict["accesses"])))
        comp_repr.extend(read_accesses_repr)

        comps_templates.append(comp_repr)
        for col, element in enumerate(comp_repr):
            if isinstance(element, str):
                placeholders[element] = (comp_index, col)

    # Per-loop representation, independent of computations.
    loops_templates = []
    loop_placeholders = {}
    for loop_index, loop_name in enumerate(program_json["iterators"]):
        l_code = "L" + loop_name
        loop_repr = [
            l_code + "Parallelized",
            l_code + "Tiled",
            l_code + "TileFactor",
            l_code + "Fused",
            l_code + "Unrolled",
            l_code + "UnrollFactor",
            l_code + "Shifted",
            l_code + "ShiftFactor",
        ]
        loops_templates.append(loop_repr)
        for col, element in enumerate(loop_repr):
            if isinstance(element, str):
                loop_placeholders[element] = (loop_index, col)

    loops_indices = {name: i for i, name in enumerate(program_json["iterators"])}
    comps_indices = {name: i for i, name in enumerate(ordered_comp_list)}
    prog_tree = annotate_tree(build_program_tree(program_json), loops_indices, comps_indices)

    # Precompute the schedule-independent per-computation features once per program:
    #   * the original iteration domain (written into the template's constraint slots,
    #     with the unpadded matrix kept for the per-schedule transformed domain), and
    #   * the padded expression sequence.
    initial_domains = {}
    expr_repr = []
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        coeff_mat, constants_col = initial_iteration_domain(program_json, comp_name)
        initial_domains[comp_name] = coeff_mat
        c_code = "C" + str(comp_index)
        ogc_start, ogc_end = placeholders[c_code + "-OgConstraintMatrixStart"], placeholders[c_code + "-OgConstraintMatrixEnd"]
        comps_templates[ogc_start[0]][ogc_start[1]: ogc_end[1] + 1] = pad_domain_matrix(coeff_mat).flatten().tolist()
        ogv_start, ogv_end = placeholders[c_code + "-OgConstraintVectorStart"], placeholders[c_code + "-OgConstraintVectorEnd"]
        comps_templates[ogv_start[0]][ogv_start[1]: ogv_end[1] + 1] = pad_domain_constants(constants_col).tolist()

        expression = expr_tree_repr(comp_dict["expression_representation"], comp_dict["data_type"])
        if len(expression) > MAX_EXPR_LEN:
            raise ExpressionTooLongException
        expression.extend([[0] * 11] * (MAX_EXPR_LEN - len(expression)))
        expr_repr.append(expression)

    return RepresentationTemplate(
        tree=prog_tree,
        comps_templates=comps_templates,
        loops_templates=loops_templates,
        placeholders=placeholders,
        loop_placeholders=loop_placeholders,
        initial_domains=initial_domains,
        expr_repr=expr_repr,
    )


# --------------------------------------------------------------------------------------
# Schedule features -> filled numeric tensors
# --------------------------------------------------------------------------------------

def fill_schedule_representation(program_json, schedule_json, template):
    """Fill a program's :class:`RepresentationTemplate` with one schedule's features.

    Returns ``(comps_tensor, loops_tensor, comps_expr_tensor)``, each with a leading
    batch dimension of 1.
    """
    placeholders = template.placeholders
    loop_placeholders = template.loop_placeholders
    # Copy the templates (rows only; elements are immutable ints/strings) so concurrent
    # schedules of the same program don't clobber each other.
    comps_repr = [list(row) for row in template.comps_templates]
    loops_repr = [list(row) for row in template.loops_templates]

    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        computations_dict.keys(),
        key=lambda name: computations_dict[name]["absolute_order"],
    )

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        comp_schedule_dict = schedule_json[comp_name]

        fused_levels = []
        if schedule_json.get("fusions"):
            for fusion in schedule_json["fusions"]:
                if comp_name in fusion:
                    fused_levels.append(fusion[2])

        c_code = "C" + str(comp_index)
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            l_code = c_code + "-L" + str(iter_i)

            parallelized = int(iterator_name == comp_schedule_dict["parallelized_dim"])
            _set(comps_repr, placeholders[l_code + "Parallelized"], parallelized)

            tiled, tile_factor = 0, 0
            if comp_schedule_dict["tiling"] and iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]:
                tiled = 1
                idx = comp_schedule_dict["tiling"]["tiling_dims"].index(iterator_name)
                tile_factor = int(comp_schedule_dict["tiling"]["tiling_factors"][idx])
            _set(comps_repr, placeholders[l_code + "Tiled"], tiled)
            _set(comps_repr, placeholders[l_code + "TileFactor"], tile_factor)

            _set(comps_repr, placeholders[l_code + "Fused"], int(iter_i in fused_levels))

            shifted, shift_factor = 0, 0
            if comp_schedule_dict["shiftings"]:
                for shifting in comp_schedule_dict["shiftings"]:
                    if iterator_name.startswith(shifting[0]):
                        shifted, shift_factor = 1, shifting[1]
                        break
            _set(comps_repr, placeholders[l_code + "Shifted"], shifted)
            _set(comps_repr, placeholders[l_code + "ShiftFactor"], shift_factor)

        unrolled, unroll_factor = 0, 0
        if comp_schedule_dict["unrolling_factor"]:
            unrolled = 1
            unroll_factor = int(comp_schedule_dict["unrolling_factor"])
        _set(comps_repr, placeholders[c_code + "-Unrolled"], unrolled)
        _set(comps_repr, placeholders[c_code + "-UnrollFactor"], unroll_factor)

        # Affine transformation tags.
        padded_tags = padded_transformation_tags(comp_schedule_dict)
        tags_start = placeholders[c_code + "-TransformationTagsStart"]
        tags_end = placeholders[c_code + "-TransformationTagsEnd"]
        assert len(padded_tags) == tags_end[1] - tags_start[1] + 1
        comps_repr[tags_start[0]][tags_start[1]: tags_end[1] + 1] = padded_tags

        # Transformed iteration domain (the original domain is precomputed in the
        # template; only the schedule-dependent transformation is applied here).
        c_start = placeholders[c_code + "-ConstraintMatrixStart"]
        c_end = placeholders[c_code + "-ConstraintMatrixEnd"]
        assert MAX_DEPTH * MAX_DEPTH * 2 == c_end[1] - c_start[1] + 1
        transformed = transformed_iteration_domain(
            program_json, schedule_json, comp_name, template.initial_domains[comp_name]
        )
        comps_repr[c_start[0]][c_start[1]: c_end[1] + 1] = transformed.flatten().tolist()

    _fill_loop_representation(
        program_json, schedule_json, computations_dict, ordered_comp_list,
        loops_repr, loop_placeholders,
    )

    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)
    comps_expr_tensor = torch.tensor([template.expr_repr]).float()
    return computations_tensor, loops_tensor, comps_expr_tensor


def _set(repr_list, index, value):
    repr_list[index[0]][index[1]] = value


def _fill_loop_representation(
    program_json, schedule_json, computations_dict, ordered_comp_list,
    loops_repr, loop_placeholders,
):
    """Aggregate per-computation schedule features to loop granularity and write them
    into ``loops_repr`` (in place)."""
    loop_schedules = {
        loop_name: {
            "tiled": 0, "tile_factor": 0, "unrolled": 0, "unroll_factor": 0,
            "shifted": 0, "shift_factor": 0, "parallelized": 0, "fused": 0,
        }
        for loop_name in program_json["iterators"]
    }

    for comp_name in ordered_comp_list:
        comp_schedule_dict = schedule_json[comp_name]
        # A fused computation shares the loop nest (and schedule) of the computation it
        # was fused into.
        if schedule_json.get("fusions"):
            for fusion in schedule_json["fusions"]:
                if comp_name in fusion:
                    comp_schedule_dict = schedule_json[fusion[0]]

        if comp_schedule_dict["tiling"]:
            for i, tiled_loop in enumerate(comp_schedule_dict["tiling"]["tiling_dims"]):
                loop_schedules[tiled_loop]["tiled"] = 1
                factor = int(comp_schedule_dict["tiling"]["tiling_factors"][i])
                assert loop_schedules[tiled_loop]["tile_factor"] in (0, factor)
                loop_schedules[tiled_loop]["tile_factor"] = factor

        if comp_schedule_dict["unrolling_factor"]:
            innermost = schedule_json["transformed_iterators"][comp_name][-1]
            loop_schedules[innermost]["unrolled"] = 1
            factor = int(comp_schedule_dict["unrolling_factor"])
            assert loop_schedules[innermost]["unroll_factor"] in (0, factor)
            loop_schedules[innermost]["unroll_factor"] = factor

        if comp_schedule_dict["parallelized_dim"]:
            loop_schedules[comp_schedule_dict["parallelized_dim"]]["parallelized"] = 1

        if comp_schedule_dict["shiftings"]:
            for shifting in comp_schedule_dict["shiftings"]:
                loop_schedules[shifting[0]]["shifted"] = 1
                loop_schedules[shifting[0]]["shift_factor"] = shifting[1]

    if schedule_json.get("fusions"):
        for fusion in schedule_json["fusions"]:
            loop_schedules[computations_dict[fusion[0]]["iterators"][fusion[2]]]["fused"] = 1
            loop_schedules[computations_dict[fusion[1]]["iterators"][fusion[2]]]["fused"] = 1

    loop_suffixes = (
        "Parallelized", "Tiled", "TileFactor", "Unrolled",
        "UnrollFactor", "Shifted", "ShiftFactor", "Fused",
    )
    schedule_keys = (
        "parallelized", "tiled", "tile_factor", "unrolled",
        "unroll_factor", "shifted", "shift_factor", "fused",
    )
    for loop_name in program_json["iterators"]:
        l_code = "L" + loop_name
        s = loop_schedules[loop_name]
        for suffix, key in zip(loop_suffixes, schedule_keys):
            _set(loops_repr, loop_placeholders[l_code + suffix], s[key])

    # Fusion can drop some of a computation's iterators; zero out any loop absent from
    # the last computation's fusion-adjusted iterator list.
    last_comp = ordered_comp_list[-1]
    transformed_iterators = schedule_json["transformed_iterators"][last_comp]
    if len(program_json["iterators"]) > len(transformed_iterators):
        for loop_name in set(program_json["iterators"]) - set(transformed_iterators):
            l_code = "L" + loop_name
            for suffix in loop_suffixes:
                _set(loops_repr, loop_placeholders[l_code + suffix], 0)


# --------------------------------------------------------------------------------------
# Affine transformation tags & matrices
# --------------------------------------------------------------------------------------

def padded_transformation_tags(comp_schedule_dict):
    """Flatten a computation's affine transformation tags into a fixed-length vector.

    The tag layout (16 elements) is::

        [type, interchange_i, interchange_j, reversed_loop,
         skew_i, skew_j, skew_k, skew_p1..skew_p9]

    where ``type`` is 0 (none), 1 (interchange), 2 (reversal) or 3 (skewing). Raises
    :class:`NbTransformationException` for sequences longer than
    ``MAX_NUM_TRANSFORMATIONS`` (kept out of the fixed-size model input).
    """
    tags = comp_schedule_dict["transformations_list"]
    if len(tags) > MAX_NUM_TRANSFORMATIONS:
        raise NbTransformationException
    flat = [element for tag in tags for element in tag]
    flat += [0] * (MAX_NUM_TRANSFORMATIONS * MAX_TAGS - len(flat))
    return flat


def transformation_matrix_from_tag(transformation, matrix_size):
    """Convert a 16-element affine tag into its ``matrix_size`` transformation matrix."""
    matrix = np.identity(matrix_size)
    assert len(transformation) == MAX_TAGS
    if transformation[0] == 1:  # interchange
        assert transformation[1] < matrix_size and transformation[2] < matrix_size
        matrix[transformation[1], transformation[1]] = 0
        matrix[transformation[2], transformation[2]] = 0
        matrix[transformation[1], transformation[2]] = 1
        matrix[transformation[2], transformation[1]] = 1
    elif transformation[0] == 2:  # reversal
        assert transformation[3] < matrix_size
        matrix[transformation[3], transformation[3]] = -1
    elif transformation[0] == 3:  # skewing
        if transformation[6] == 0:  # 2D
            assert transformation[4] < matrix_size and transformation[5] < matrix_size
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[5], transformation[4]] = transformation[9]
            matrix[transformation[5], transformation[5]] = transformation[10]
        else:  # 3D
            i, j, k = transformation[4], transformation[5], transformation[6]
            assert i < matrix_size and j < matrix_size and k < matrix_size
            matrix[i, i], matrix[i, j], matrix[i, k] = transformation[7], transformation[8], transformation[9]
            matrix[j, i], matrix[j, j], matrix[j, k] = transformation[10], transformation[11], transformation[12]
            matrix[k, i], matrix[k, j], matrix[k, k] = transformation[13], transformation[14], transformation[15]
    return matrix


def net_transformation_matrix(program_json, schedule_json, comp_name):
    """Net affine transformation matrix = product of the computation's tags in order."""
    nb_iterators = len(program_json["computations"][comp_name]["iterators"])
    result = np.identity(nb_iterators)
    for transformation in schedule_json[comp_name]["transformations_list"]:
        result = np.matmul(transformation_matrix_from_tag(transformation, nb_iterators), result)
    return result


# --------------------------------------------------------------------------------------
# Iteration-domain constraints
# --------------------------------------------------------------------------------------

def initial_iteration_domain(program_json, comp_name):
    """Unpadded constraint matrix ``A`` and constants vector ``b`` for a computation's
    original iteration domain (``D = {x : A x <= b}``).

    This is schedule-independent and uses ``sympy`` to parse the loop bounds, so it is
    computed once per program (in :func:`build_representation_template`) and reused for
    every schedule rather than recomputed each time.

    Supports bounds of the form ``+/- i +/- j ... +/- const`` and ``max(const, iter)``.
    """
    comp_dict = program_json["computations"][comp_name]
    nb_dims = len(comp_dict["iterators"])
    coeff_mat = np.zeros((nb_dims * 2, nb_dims), int)
    constants_col = np.zeros((nb_dims * 2), int)

    for i, iterator_name_row in enumerate(comp_dict["iterators"]):
        coeff_mat[i * 2, i] = -1
        coeff_mat[i * 2 + 1, i] = 1

        iterator_row_dict = program_json["iterators"][iterator_name_row]
        upper_bound = str(sympy.simplify(iterator_row_dict["upper_bound"])).replace(" ", "")
        lower_bound = str(sympy.simplify(iterator_row_dict["lower_bound"])).replace(" ", "")

        # Special case: lower bound expressed as max(const, iter); take the constant.
        if "max" in lower_bound or "Max" in lower_bound:
            lower_bound = re.findall(r"[mM]ax\(.+,(.+)\)", lower_bound)[0]

        constants_in_upper = re.findall(r"(?:^|\+|-)\d+", upper_bound) or [0]
        constants_in_lower = re.findall(r"(?:^|\+|-)\d+", lower_bound) or [0]
        assert len(constants_in_upper) == 1
        assert len(constants_in_lower) == 1

        for iter_name in re.findall(r"[a-zA-Z]\w*", upper_bound):
            col_idx = comp_dict["iterators"].index(iter_name)
            coeff_mat[i * 2 + 1, col_idx] = 1 if "-" + iter_name in upper_bound else -1
        # -1 because we represent a non-strict inequality (i < ub  ->  i <= ub - 1).
        constants_col[i * 2 + 1] = int(constants_in_upper[0]) - 1

        for iter_name in re.findall(r"[a-zA-Z]\w*", lower_bound):
            col_idx = comp_dict["iterators"].index(iter_name)
            coeff_mat[i * 2, col_idx] = -1 if "-" + iter_name in lower_bound else 1
        constants_col[i * 2] = -int(constants_in_lower[0])

    return coeff_mat, constants_col


def pad_domain_matrix(matrix):
    """Zero-pad an iteration-domain matrix to ``(MAX_DEPTH * 2, MAX_DEPTH)``."""
    return np.pad(
        matrix, [(0, MAX_DEPTH * 2 - matrix.shape[0]), (0, MAX_DEPTH - matrix.shape[1])],
        mode="constant", constant_values=0,
    )


def pad_domain_constants(constants):
    """Zero-pad an iteration-domain constants vector to ``MAX_DEPTH * 2``."""
    return np.pad(constants, [(0, MAX_DEPTH * 2 - constants.shape[0])], mode="constant", constant_values=0)


def transformed_iteration_domain(program_json, schedule_json, comp_name, initial_matrix):
    """Padded iteration-domain constraint matrix after applying the schedule's affine
    transformations: ``A @ inv(T)`` where ``A`` is the (precomputed) original constraint
    matrix and ``T`` is the schedule's net transformation matrix."""
    transformation_matrix = net_transformation_matrix(program_json, schedule_json, comp_name)
    return pad_domain_matrix(np.matmul(initial_matrix, np.linalg.inv(transformation_matrix)))


# --------------------------------------------------------------------------------------
# Access matrices
# --------------------------------------------------------------------------------------

def pad_access_matrix(access_matrix):
    """Prepend an all-ones row/column and zero-pad to ``(MAX_DEPTH+1, MAX_DEPTH+2)``."""
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    padded[: access_matrix.shape[0], : access_matrix.shape[1] - 1] = access_matrix[:, :-1]
    padded[: access_matrix.shape[0], -1] = access_matrix[:, -1]
    return padded


def isl_to_write_matrix(isl_map):
    """Parse an ISL write-access relation string into an access matrix."""
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix


# --------------------------------------------------------------------------------------
# Expression representation
# --------------------------------------------------------------------------------------

_EXPR_TYPES = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sqrt": 4, "min": 5, "max": 6}
_DATA_TYPES = {"int32": 0, "float32": 1, "float64": 2}


def expr_op_repr(expr, comp_type):
    """One-hot encode one expression operation (8 dims) + data type (3 dims)."""
    expr_vector = [0] * 8
    expr_vector[_EXPR_TYPES.get(expr, 7)] = 1
    comp_type_vector = [0] * 3
    if comp_type in _DATA_TYPES:
        comp_type_vector[_DATA_TYPES[comp_type]] = 1
    return expr_vector + comp_type_vector


def expr_tree_repr(node, comp_type):
    """Post-order sequence of operation vectors for a computation's expression tree."""
    expr_tensor = []
    for child_node in node["children"]:
        expr_tensor.extend(expr_tree_repr(child_node, comp_type))
    expr_tensor.append(expr_op_repr(node["expr_type"], comp_type))
    return expr_tensor


# --------------------------------------------------------------------------------------
# Labels, dropping, batching helper
# --------------------------------------------------------------------------------------

def speedup_clip(speedup):
    """Lower-bound the speedup to stabilize training."""
    return max(speedup, 0.01)


def drop_program(prog_dict):
    """Drop programs with fewer than two explored schedules."""
    return len(prog_dict["schedules_list"]) < 2


def drop_schedule(schedule_json):
    """Drop schedules with empty or non-positive execution times."""
    times = schedule_json["execution_times"]
    return (not times) or min(times) < 0


def datapoint_attributes(func_name, program_dict, schedule_index, footprint):
    """Metadata carried alongside each datapoint for analysis / evaluation."""
    schedule_json = program_dict["schedules_list"][schedule_index]
    exec_time = np.min(schedule_json["execution_times"])
    return (
        func_name,
        str(schedule_index).zfill(4),
        schedule_json.get("schedule_str", ""),
        exec_time,
        program_dict["program_annotation"]["memory_size"],
        program_dict.get("node_name", "unknown"),
        footprint,
        program_dict["initial_execution_time"] / exec_time,
    )


def split_computation_vector(x, num_transformations=MAX_NUM_TRANSFORMATIONS):
    """Split the flat computation vector into (static prefix, affine-tag vectors, static
    suffix). The middle block is reshaped to ``(batch, num_transformations, MAX_TAGS)``
    for the transformation-embedding LSTM.
    """
    first_part = x[:, :33]
    tags_block = x[:, 33: 33 + MAX_TAGS * num_transformations]
    third_part = x[:, 33 + MAX_TAGS * num_transformations:]
    batch_size = x.shape[0]
    vectors = tags_block.reshape(batch_size, num_transformations, MAX_TAGS)
    return first_part, vectors, third_part
