import torch
from transformers import AutoFeatureExtractor, AutoConfig, AutoModel
import pickle
import sys
sys.path.append("/home/yileiyang/workspace/DNNGen/concrete_trace_test/nni/")
sys.path.append("/home/yileiyang/workspace/DNNGen/concrete_trace_test/concrete_trace/huggingface/concrete_trace_subtype")
sys.path.append("/home/yileiyang/workspace/DNNGen/concrete_trace_test/concrete_trace/graph2seq")
from nni.common.concrete_trace_utils import concrete_trace

with open('/home/yileiyang/workspace/DNNGen/concrete_trace_test/concrete_trace/huggingface/concrete_trace_subtype/cv/img.pkl', 'rb') as f:
    raw_input = pickle.load(f)

model_name = 'hustvl/yolos-tiny'
extractor = AutoFeatureExtractor.from_pretrained(model_name)
dummy_input = extractor(images=raw_input, return_tensors='pt')
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_config(config)
model.eval()

with torch.no_grad():
    output_origin = model(**dummy_input)

traced_gm = concrete_trace(
    model,
    dummy_input,
    use_operator_patch=True,
    autowrap_leaf_class={
        torch.finfo: ((), False),
        type(output_origin): ((), False),
    },
    dce=True
)

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
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return a == b
    
with torch.no_grad():
    output_traced = traced_gm(**dummy_input)

assert check_equal(output_origin, output_traced), 'check equal failed'
print('trace succeeded!')
