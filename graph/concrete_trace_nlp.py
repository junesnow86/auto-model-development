import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from _collections_abc import MutableMapping

import torch
from torch.utils._pytree import tree_map
from transformers import AutoConfig, AutoModel, AutoTokenizer

from cube.graph.parser.converter import to_fx_graph
# pip install sentencepiece
import psutil
import gc
import multiprocessing
import time

def print_memory_usage(prefix : str = ""):
    process = psutil.Process()
    mem_info = process.memory_info()
    print("When " + prefix + f": Current memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")


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
    if torch.cuda.is_available():
        try:
            traced_gm = to_fx_graph(model, concrete_args)
        except:
            raise Exception("Failed to trace with gpu")
        print("Successfully traced with gpu")
        return traced_gm
    else:
        raise RuntimeError("CUDA is not available")

def test_forward(traced_gm, concrete_args):

    try:
        traced_gm.eval()
        traced_gm(**concrete_args)
    except:
        raise Exception("Failed to run forward with gpu")
    return True

def check_align(before_trace, after_trace):
    for key in after_trace.keys():
        if isinstance(after_trace[key], torch.Tensor):
            if not torch.all(before_trace[key] == after_trace[key]):
                return False
    return True

def trace_single_model(model_name: str, nlp_dir: str, model_name_set: set, tried, model_loaded: list, success_traced: list, forward_aligned: list, model_failed: list, success_lock, errors_lock, fail_lock):
    try:
        tried.value += 1
        print_memory_usage(f"trying to trace {model_name}")
        # concrete_args = create_concrete_args(model_name)
        text: str = "Huggingface is a really excellent project!"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        concrete_args = tokenizer(text, return_tensors="pt")
        if isinstance(concrete_args, MutableMapping):
            concrete_args = dict(concrete_args)
        print_memory_usage("after create_concrete_args of " + model_name)
        # model = build_model(model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        print_memory_usage("after build_model of " + model_name)
        model.eval()
        model_loaded.append(f"{model_name}, {config.architectures}")
        before_trace = model(**concrete_args)
        
        traced_gm = concrete_trace_wrap(model, concrete_args)
        print_memory_usage("after concrete_trace of " + model_name)
        success_traced.append(f"{model_name}, {config.architectures}")
        traced_gm.eval()
        after_trace = traced_gm(**concrete_args)

        assert check_align(before_trace, after_trace), "Traced model does not match the original model"
        forward_aligned.append(f"{model_name}, {config.architectures}")
        with success_lock:
            with open(os.path.join(nlp_dir, "success"), 'a') as file:
                file.write(f"{model_name}\n")
        print(f"{model_name} traced successfully")
    except Exception as e:
        model_failed.append(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        error_message = traceback.format_exc()
        with errors_lock:
            with open(os.path.join(nlp_dir, "errors"), "a") as file:
                file.writelines(f"\n{model_name} failed\n")
                if 'config' in locals() and config:
                    file.writelines(f"Architectures: {config.architectures}\n")
                file.writelines(error_message)
        with fail_lock:
            with open(os.path.join(nlp_dir, "fail"), 'a') as file:
                file.write(f"{model_name}\n")
        print(f"{model_name} failed")
        print(error_message)
    finally:
        print(f"all: {len(model_name_set)}, tried: {tried.value}, model_failed: {len(model_failed)} model_loaded: {len(model_loaded)}, success_traced: {len(success_traced)}, forward_aligned: {len(forward_aligned)}")

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    model_name_set_path = os.path.join(current_folder, "huggingface_model_names/model_name_set_nlp")
    with open(model_name_set_path, 'r') as f:
        all_model = eval(f.read())

    nlp_dir = os.path.join(current_folder, "nlp")
    if not os.path.exists(nlp_dir):
        os.makedirs(nlp_dir)

    print(f"# model: {len(all_model)}")
    success_models = set()
    if os.path.exists(os.path.join(nlp_dir, "success")):
        with open(os.path.join(nlp_dir, "success"), 'r') as file:
            names = [line.strip() for line in file]
            success_models = set(names)
    model_name_set = all_model - success_models
    print(f"# already_collected: {len(success_models)}")

    fail_models = set()
    if os.path.exists(os.path.join(nlp_dir, "fail")):
        with open(os.path.join(nlp_dir, "fail"), 'r') as file:
            names = [line.strip() for line in file]
            fail_models = set(names)
    model_name_set = model_name_set - success_models
    print(f"# already_failed: {len(fail_models)}")
    print(f"# need_to_try: {len(model_name_set)}")

    mem_info = psutil.virtual_memory()
    total_memory = mem_info.total
    print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")

    


    with multiprocessing.Manager() as manager:
        tried = manager.Value('i', 0)
        model_loaded = manager.list()
        success_traced = manager.list()
        forward_aligned = manager.list()
        model_failed = manager.list()
        success_lock = manager.Lock()
        errors_lock = manager.Lock()
        fail_lock = manager.Lock()
        with multiprocessing.Pool(processes=24) as pool:
            arguments = [(model_name, nlp_dir, model_name_set, tried, model_loaded, success_traced, forward_aligned, model_failed, success_lock, errors_lock, fail_lock) for model_name in model_name_set]

            pool.starmap(trace_single_model, arguments)
        
    # for model_name in model_name_set:
        
    #     tried += 1
    #     my_process = multiprocessing.Process(target=trace_single_model, args=(model_name, nlp_dir, model_name_set, tried, model_loaded, success_traced, forward_aligned, model_failed), name="Trace " + model_name)
    #     my_process.start()
    #     my_process.join()

        # try:
        #     # concrete_args = create_concrete_args(model_name)
        #     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        #     concrete_args = tokenizer(text, return_tensors="pt")
        #     if isinstance(concrete_args, MutableMapping):
        #         concrete_args = dict(concrete_args)
        #     print_memory_usage("after create_concrete_args of " + model_name)
        #     # model = build_model(model_name)
        #     config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        #     model = AutoModel.from_config(config, trust_remote_code=True)
        #     print_memory_usage("after build_model of " + model_name)
        #     model.eval()
        #     model_loaded.append(f"{model_name}, {config.architectures}")
        #     before_trace = model(**concrete_args)
            
        #     traced_gm = concrete_trace_wrap(model, concrete_args)
        #     print_memory_usage("after concrete_trace of " + model_name)
        #     success_traced.append(f"{model_name}, {config.architectures}")
        #     traced_gm.eval()
        #     after_trace = traced_gm(**concrete_args)

        #     assert check_align(before_trace, after_trace), "Traced model does not match the original model"
        #     forward_aligned.append(f"{model_name}, {config.architectures}")
        #     with open(os.path.join(nlp_dir, "success"), 'a') as file:
        #         file.write(f"{model_name}\n")
        #     print(f"{model_name} traced successfully")
        # except Exception as e:
        #     model_failed.append(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        #     error_message = traceback.format_exc()
        #     with open(os.path.join(nlp_dir, "errors"), "a") as file:
        #         file.writelines(f"\n{model_name} failed\n")
        #         if 'config' in locals() and config:
        #             file.writelines(f"Architectures: {config.architectures}\n")
        #         file.writelines(error_message)
        #     with open(os.path.join(nlp_dir, "fail"), 'a') as file:
        #         file.write(f"{model_name}\n")
        #     print(f"{model_name} failed")
        #     print(error_message)
        #     continue
        # finally:
        #     try:
        #         del tokenizer
        #         del concrete_args
        #         del config
        #         del model
        #         del before_trace
        #         del traced_gm
        #         del after_trace
        #     except NameError:
        #         pass
            
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     print(f"all: {len(model_name_set)}, tried: {tried}, model_failed: {len(model_failed)} model_loaded: {len(model_loaded)}, success_traced: {len(success_traced)}, forward_aligned: {len(forward_aligned)}")
    with open(os.path.join(nlp_dir, "nlp_log"), 'w') as f:
        import json
        result_dict = {
            'forward_aligned': tuple(forward_aligned),
            'forward_aligned_num': len(forward_aligned),
            'success_traced': tuple(success_traced),
            'success_traced_num': len(success_traced),
            'model_loaded': tuple(model_loaded),
            'model_loaded_num': len(model_loaded),
            'model_failed': tuple(model_failed),
            'model_failed_num': len(model_failed),
            'tried': tried,
            'model_name_set': tuple(model_name_set),
            'model_name_set_num': len(model_name_set),
            'already_success': tuple(success_models),
            'already_success_num': len(success_models),
            'already_failed': tuple(fail_models),
            'already_failed_num': len(fail_models),
            'all_model': tuple(all_model),
            'all_model_num': len(all_model),
        }
        json.dump(result_dict, f, indent=4)

    print("concrete trace nlp done!")
