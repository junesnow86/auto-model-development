import os
import pickle
import sys
import traceback

import torch
from torch.utils._pytree import tree_map

from transformers import (
    AutoConfig,
    AutoModel,
    AutoFeatureExtractor,
)

sys.path.append("/home/yileiyang/workspace/DNNGen/nni")
sys.path.append("/home/yileiyang/workspace/DNNGen/'auto_model_dev/concrete_trace/huggingface/concrete_trace_subtype")
sys.path.append("/home/yileiyang/workspace/DNNGen/auto_model_dev")
from fxgraph_to_seq import Sequence, fold_seq

from concrete_trace_utils import concrete_trace
# from nni.common.concrete_trace_utils.passes.kwargs_shape_prop import KwargsShapeProp


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
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    with open(os.path.join(current_folder, "img.pkl"), "rb") as f:
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
    with open("/home/yileiyang/workspace/auto-model-compiler/graph/huggingface_model_names/model_name_set_cv", 'r') as f:
        model_name_set = eval(f.read())
    print(f"#model: {len(model_name_set)}")

    failed_model_names_file_path = "/home/yileiyang/workspace/auto-model-compiler/graph/cv/failed_model_names"
    try:
        with open(failed_model_names_file_path, 'r') as f:
            failed_model_names = eval(f.read())
    except:
        failed_model_names = set()

    xml_save_dir = "/home/yileiyang/workspace/auto-model-compiler/graph/cv"
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)
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
            model = model.to("cuda:0")
            dummy_input = tree_map(lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x, dummy_input)
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
            # KwargsShapeProp(traced_gm).propagate(dummy_input)
            seq = Sequence(traced_gm, model)
            with open(os.path.join(xml_save_dir, model_name.replace('/', "--")), 'w') as f:
                print(fold_seq(seq), file=f)
            traced_count += 1
        except Exception as e:
            print(f"{model_name} failed")
            traceback.print_exc()
            failed_model_names.add(model_name)
            with open(failed_model_names_file_path, 'w') as f:
                print(failed_model_names, file=f)

    print("concrete trace cv done!")
