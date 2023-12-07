import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from _collections_abc import MutableMapping

import torch
from torch.utils._pytree import tree_map
from transformers import AutoConfig, AutoModel, AutoTokenizer
from cube.graph.parser.converter import to_fx_graph

import psutil
import multiprocessing
from concurrent_log_handler import ConcurrentRotatingFileHandler 
import logging 
import time

# pip install sentencepiece transformers fuzzywuzzy concurrent-log-handler

def setup_logger(log_file):
    logger = logging.getLogger(f'process-{multiprocessing.current_process().pid}')
    logger.setLevel(logging.DEBUG)
  
    handler = ConcurrentRotatingFileHandler(log_file, "a", 1024*1024, 8)
    formatter = logging.Formatter('%(asctime)s [PID %(process)d][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def print_memory_usage(prefix : str = ""):
    process = psutil.Process()
    mem_info = process.memory_info()
    print("When " + prefix + f": Current memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")

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

def check_align(before_trace, after_trace):
    for key in after_trace.keys():
        if isinstance(after_trace[key], torch.Tensor):
            if not torch.all(before_trace[key] == after_trace[key]):
                return False
    return True

def trace_single_model(model_name: str, nlp_dir: str, model_name_set: set, tried: list, 
                       model_loaded: list, success_traced: list,  model_failed: list, error_lock):
    try:
        logger = setup_logger(os.path.join(nlp_dir, 'all4debug.log'))
        start_time = time.time()
        # print_memory_usage(f"trying to trace {model_name}")
        logger.info(f"start trying model: {model_name}")
  
        logger.info(f"{model_name} traced successfully")
        tried.append(model_name)

        text: str = "Huggingface is a really excellent project!"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  
        logger.info("Tokenizer loaded")
        concrete_args = tokenizer(text, return_tensors="pt")
        if isinstance(concrete_args, MutableMapping):
            concrete_args = dict(concrete_args)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        logger.info("model loaded")  
        model_loaded.append(f"{model_name}, {config.architectures}")

        # model.eval()
        # before_trace = model(**concrete_args)
        
        traced_gm = concrete_trace_wrap(model, concrete_args)
        logger.info("model traced")  
        success_traced.append(f"{model_name}, {config.architectures}")

        # traced_gm.eval()
        # after_trace = traced_gm(**concrete_args)

        # assert check_align(before_trace, after_trace), "Traced model does not match the original model"
        # forward_aligned.append(f"{model_name}, {config.architectures}")
        
        # print_memory_usage("after concrete_trace of " + model_name)
        # print(f"{model_name} traced successfully")
    except Exception as e:
        logger.error(f"fail when trying model: {model_name}", exc_info=True)
        model_failed.append(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        error_message = traceback.format_exc()
        logger.error(f"Ready to write errors: {model_name}")
        with error_lock:
            with open(os.path.join(nlp_dir, "errors"), "a") as file:
                file.writelines(f"\n{model_name} failed\n")
                if 'config' in locals() and config:
                    file.writelines(f"Architectures: {config.architectures}\n")
                file.writelines(error_message)
        logger.error(f"Finish write errors: {model_name}")
        # print(f"{model_name} failed")
        # print(error_message)
    finally:
        end_time = time.time()
        logger.info(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")
        # print(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    model_name_set_path = os.path.join(current_folder, "huggingface_model_names/model_name_set_nlp") # model_name_set_nlp
    with open(model_name_set_path, 'r') as f:
        all_model = eval(f.read())
    print(f"# model: {len(all_model)}")

    nlp_dir = os.path.join(current_folder, "nlp")
    if not os.path.exists(nlp_dir):
        os.makedirs(nlp_dir)

    tried_models = set()
    if os.path.exists(os.path.join(nlp_dir, "tried")):
        with open(os.path.join(nlp_dir, "tried"), 'r') as file:
            names = [line.strip() for line in file]
            tried_models = set(names)
    model_name_set = all_model - tried_models
    print(f"# already_tried: {len(tried_models)}")
    print(f"# need_to_try: {len(model_name_set)}")

    mem_info = psutil.virtual_memory()
    total_memory = mem_info.total
    print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")

    process_num=23
    tried_list = []
    model_loaded_list = []
    success_traced_list = []
    model_failed_list = []
    with multiprocessing.Manager() as manager:
        tried = manager.list()
        model_loaded = manager.list()
        success_traced = manager.list()
        # forward_aligned = manager.list()
        model_failed = manager.list()

        error_lock = manager.Lock()

        arguments = [(model_name, nlp_dir, model_name_set, tried, model_loaded, 
                    success_traced, model_failed, error_lock) for model_name in model_name_set]

        batch_size = process_num * 4
        arg_batches = [arguments[i:i + batch_size] for i in range(0, len(arguments), batch_size)]

        for arg_batch in arg_batches:
            with multiprocessing.Pool(processes = process_num) as pool:
                results = pool.starmap(trace_single_model, arg_batch)
                print(results)

                tried_list.extend(list(tried))
                with open(os.path.join(nlp_dir, "tried"), 'a') as file:
                    for model_name in list(tried):
                        file.write(f"{model_name}\n")
                    tried[:] = []
                print(f"tried: {len(tried_list)}")
                    
                model_loaded_list.extend(list(model_loaded))
                with open(os.path.join(nlp_dir, "model_loaded"), 'a') as file:
                    for model_name in list(model_loaded):
                        file.write(f"{model_name}\n")
                    model_loaded[:] = []
                print(f"model_loaded: {len(model_loaded_list)}")

                success_traced_list.extend(list(success_traced))
                with open(os.path.join(nlp_dir, "success_traced"), 'a') as file:
                    for model_name in list(success_traced):
                        file.write(f"{model_name}\n")
                    success_traced[:] = []
                print(f"success_traced: {len(success_traced_list)}")

                model_failed_list.extend(list(model_failed))
                with open(os.path.join(nlp_dir, "model_failed"), 'a') as file:
                    for model_name in list(model_failed):
                        file.write(f"{model_name}\n")
                    model_failed[:] = []
                print(f"model_failed: {len(model_failed_list)}")
                # gc.collect()

                print(f"all: {len(model_name_set)}, tried: {len(tried_list)}, \
                        model_loaded: {len(model_loaded_list)}, success_traced: {len(success_traced_list)}, \
                        model_failed: {len(model_failed_list)}")

    print("concrete trace nlp done!")
