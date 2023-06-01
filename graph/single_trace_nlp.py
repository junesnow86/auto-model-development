import os
import pickle

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from concrete_trace_utils import concrete_trace
from concrete_trace_utils.passes.kwargs_shape_prop import KwargsShapeProp
from fxgraph_to_seq import Sequence, fold_seq


def create_dummy_input(model_name):
    text = "Huggingface is a really excellent project!"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dummy_input = tokenizer(text, return_tensors="pt")
        return dummy_input
    except:
        return None

def build_model(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        return model
    except:
        return None

if __name__ == "__main__":
    model_name = "gpt2"
    dummy_input = create_dummy_input(model_name)
    model = build_model(model_name)
    if dummy_input is not None and model is not None:
        traced_gm = concrete_trace(
            model,
            dummy_input,
            autowrap_leaf_class={
                torch.finfo: ([], False),
            },
            dce=True,
        )
        KwargsShapeProp(traced_gm).propagate(dummy_input)
        seq = Sequence(traced_gm, model)
        with open("sequence", "w") as f:
            print(seq, file=f)
    print("task done")
