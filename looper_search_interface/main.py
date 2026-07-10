from os import environ
import json
import numpy as np
from model import LOOPerCostModel
from json_to_tensor import *

environ["MKL_THREADING_LAYER"] = "GNU"
TIRAMISU_ROOT = environ["TIRAMISU_ROOT"]
model_path = environ["MODEL_WEIGHTS_PATH"]

# The cost model is a small recursive network run on CPU, one program at a time —
# thousands of tiny tensor ops. With PyTorch's default intra-op parallelism (one
# thread per core) the search spends far more time synchronizing threads than
# computing: on a 28-core host a single forward pass took ~3.1 s at 28 threads
# versus ~6 ms at 1 (a ~500x difference). Cap the thread count so per-candidate
# inference is fast — which is the entire point of model-guided search. Override
# with LOOPER_TORCH_THREADS if a larger model or batch benefits from more threads.
try:
    torch.set_num_threads(int(environ.get("LOOPER_TORCH_THREADS", "1")))
except (ValueError, RuntimeError):
    pass

MAX_DEPTH = 5

with torch.no_grad():
    device = "cpu"
    torch.device("cpu")

    environ["layers"] = "600 350 200 180"
    environ["dropouts"] = "0.05 " * 4

    input_size = 846 #1056
    output_size = 1

    layers_sizes = list(map(int, environ.get("layers", "600 350 200 180").split()))
    drops = list(map(float, environ.get("dropouts", "0.05 0.05 0.05 0.05 0.05").split()))

    model = LOOPerCostModel(
        input_size=input_size,
        comp_embed_layer_sizes=layers_sizes,
        drops=drops,
        loops_tensor_size=8,
    )
    if not (model_path.endswith("part0.pt") or model_path.endswith("part_0.pt")): # if part ==0 use an untrained model
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            while True:
                
                
                prog_json = input()
                no_sched_json = input()
                sched_json = input()

                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                no_sched_json = json.loads(no_sched_json)
                (
                prog_tree,
                comps_repr_templates_list,
                loops_repr_templates_list,
                comps_placeholders_indices_dict,
                loops_placeholders_indices_dict
                )  = get_representation_template(program_json, no_sched_json)
                comps_tensor, loops_tensor, comps_expr_repr = get_schedule_representation(
                    program_json,
                    sched_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict
                )
                
                x = comps_tensor
                batch_size, num_comps, __dict__ = x.shape
                
                x = x.view(batch_size * num_comps, -1)
                
                (first_part, vectors, third_part) = seperate_vector(
                        x, num_transformations=4, pad=False
                    )
                
                first_part = first_part.view(batch_size, num_comps, -1)
                
                third_part = third_part.view(batch_size, num_comps, -1)
                
                tree_tensor = (prog_tree, first_part, vectors, third_part, loops_tensor, comps_expr_repr)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))
                
        except EOFError:
            exit()