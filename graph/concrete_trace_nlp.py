import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from _collections_abc import MutableMapping

import torch
from torch.utils._pytree import tree_map
from transformers import AutoConfig, AutoModel, AutoTokenizer

from fxgraph_to_seq import Sequence
from concrete_trace_utils import concrete_trace
from concrete_trace_utils.passes.kwargs_shape_prop import KwargsShapeProp

def create_concrete_args(model_name):
    text = "Huggingface is a really excellent project!"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    concrete_args = tokenizer(text, return_tensors="pt")
    if isinstance(concrete_args, MutableMapping):
        concrete_args = dict(concrete_args)
    return concrete_args

def build_model(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    return model

def concrete_trace_wrap(model, concrete_args):
    use_cpu = False
    if torch.cuda.is_available():
        print("trying using gpu to trace...")
        try:
            model.cuda()
            to_cuda = lambda x: x.cuda() if isinstance(x, torch.Tensor) else x
            concrete_args = tree_map(to_cuda, concrete_args)
            traced_gm = concrete_trace(model, concrete_args)
        except:
            use_cpu = True
            print("failed to trace with gpu, trying cpu...")
        else:
            print("successfully traced with gpu")
            return traced_gm
    else:
        print("failed to trace with gpu, trying cpu...")
        use_cpu = True
    if use_cpu:
        try:
            model = model.cpu()
            to_cpu = lambda x: x.cpu() if isinstance(x, torch.Tensor) else x
            concrete_args = tree_map(to_cpu, concrete_args)
            traced_gm = concrete_trace(model, concrete_args)
        except:
            print("failed to trace with cpu")
            return None
        else:
            print("successfully traced with cpu")
            return traced_gm

def test_forward(traced_gm, concrete_args):
    device = next(traced_gm.parameters()).device
    concrete_args = tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
        concrete_args
    )
    try:
        traced_gm(**concrete_args)
    except:
        raise
        return False
    else:
        return True

if __name__ == "__main__":
    model_name_set_path = "model_name_set_nlp_test"
    with open(model_name_set_path, 'r') as f:
        model_name_set = eval(f.read())

    error_save_dir = "error"
    xml_save_dir = "save"
    failed_models = set(os.listdir(error_save_dir))
    failed_models = set([model_name.replace("--", "/") for model_name in failed_models])
    collected_models = set(os.listdir(xml_save_dir))
    collected_models = set([model_name.replace("--", "/") for model_name in collected_models])
    model_name_set = model_name_set - collected_models
    model_name_set = model_name_set - failed_models
    print(f"#model: {len(model_name_set)}")

    total = 0
    traced_count = 0
    for model_name in model_name_set:
        total += 1
        if total % 100 == 0:
            print(f"all: {len(model_name_set)}, tried: {total}, traced: {traced_count}")

        print("trying to trace", model_name)
        try:
            concrete_args = create_concrete_args(model_name)
            model = build_model(model_name)
            traced_gm = concrete_trace_wrap(model, concrete_args)
            assert test_forward(traced_gm, concrete_args), "traced forward failed"
            KwargsShapeProp(traced_gm).propagate(concrete_args)
            seq = Sequence(traced_gm, model)
            with open(os.path.join(xml_save_dir, model_name.replace('/', "--")), 'w') as f:
                f.write(str(seq))
            traced_count += 1
            print(f"{model_name} traced successfully")
        except:
            print(f"{model_name} failed")
            with open(os.path.join(error_save_dir, model_name.replace('/', "--")), 'w') as f:
                f.write(traceback.format_exc())
            continue

    print("concrete trace nlp done!")
