import os
import pickle
import sys

import torch
from torch.utils._pytree import tree_map

from transformers import (
    AutoConfig,
    AutoModel,
    AutoFeatureExtractor,
)

sys.path.append("/home/v-junliang/DNNGen/nni")
sys.path.append("/home/v-junliang/DNNGen/'auto_model_dev/concrete_trace/huggingface/concrete_trace_subtype")
sys.path.append("/home/v-junliang/DNNGen/auto_model_dev")
from fxgraph_to_seq import Sequence, fold_seq

from nni.common.concrete_trace_utils import concrete_trace
from nni.common.concrete_trace_utils.passes.kwargs_shape_prop import KwargsShapeProp


def check_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, keys_b = set(a.keys()), set(b.keys())
        if keys_a != keys_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return a == b

def create_dummy_input(model_name):
    with open("img.pkl", "rb") as f:
        img = pickle.load(f) 
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        dummy_input = feature_extractor(images=img, return_tensors="pt")
        return dummy_input
    except:
        return None

def build_model(model_name):
    build_apis = [
        AutoModel,
    ]

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        built = False
        model = None
        for api in build_apis:
            if not built:
                try:
                    model = api.from_config(config, trust_remote_code=True)
                except:
                    continue
                else:
                    built = True
                    break
        if built:
            model.eval()
            return model
    except:
        return None

if __name__ == "__main__":
    with open("model_name_set", 'r') as f:
        model_name_set = eval(f.read())
    print(f"#model: {len(model_name_set)}")

    failed_model_names_file_path = "/data/data0/v-junliang/DNNGen/auto_model_dev/concrete_trace_info/huggingface/cv/failed_model_names"
    try:
        with open(failed_model_names_file_path, 'r') as f:
            failed_model_names = eval(f.read())
    except:
        failed_model_names = set()

    xml_save_dir = "/data/data0/v-junliang/DNNGen/auto_model_dev/xml/huggingface/cv"
    total = 0
    traced_count = 0
    for model_name in model_name_set:
        total += 1
        if total % 100 == 0:
            print(f"all: {len(model_name_set)}, tried: {total}, traced: {traced_count}")

        if os.path.exists(os.path.join(xml_save_dir, model_name)):
            traced_count += 1
            continue
        elif model_name in failed_model_names:
            continue

        try:
            dummy_input = create_dummy_input(model_name)
            if dummy_input is None:
                continue
            model = build_model(model_name)
            if model is None:
                continue
            model = model.to("cuda:1")
            dummy_input = tree_map(lambda x: x.to("cuda:1") if isinstance(x, torch.Tensor) else x, dummy_input)
            output_origin = model(**dummy_input)
            traced_gm = concrete_trace(
                model,
                dummy_input,
                autowrap_leaf_class={
                    torch.finfo: ([], False),
                    type(output_origin): ([], False),
                },
                dce=True,
            )
            output_traced = traced_gm(**dummy_input)
            assert check_equal(output_origin, output_traced), "check_eqaul failed"
            KwargsShapeProp(traced_gm).propagate(dummy_input)
            seq = Sequence(traced_gm, model)
            with open(os.path.join(xml_save_dir, model_name.replace('/', "--")), 'w') as f:
                print(fold_seq(seq), file=f)
            traced_count += 1
        except:
            failed_model_names.add(model_name)
            with open(failed_model_names_file_path, 'w') as f:
                print(failed_model_names, file=f)

    print("concrete trace cv done!")
