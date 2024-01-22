# pip install sentencepiece transformers fuzzywuzzy concurrent-log-handler psutil
from multiprocessing.managers import ListProxy
import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from _collections_abc import MutableMapping

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from cube.graph.parser.converter import to_fx_graph

import psutil
import multiprocessing
from multiprocessing.pool import ThreadPool
from concurrent_log_handler import ConcurrentRotatingFileHandler 
import logging
import time
import timeout_decorator


text: str = "Huggingface is a really excellent project!"
cache_dir: str = "/mnt/msrasrg/yileiyang/hf_cache"

torch.set_printoptions(edgeitems=2)

@timeout_decorator.timeout(120, timeout_exception=TimeoutError)
def load_model_with_timeout(config, trust_remote_code):
    torch.manual_seed(0)
    return AutoModel.from_config(config, trust_remote_code=trust_remote_code)

def setup_logger(log_file, level):
    # different process has different logger, with timestamp
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    # logger will only init once for one log_file
    if not logger.handlers: 
        handler = ConcurrentRotatingFileHandler(log_file, "a", 1024*1024, 8)
        formatter = logging.Formatter('%(asctime)s [PID %(process)d][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def setup_logger2(log_file, level):
    # different process has the same logger, without timestamp
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    if not logger.handlers: 
        handler = ConcurrentRotatingFileHandler(log_file, "a", 10*1024*1024, 8)
        logger.addHandler(handler)
    return logger

def print_memory_usage(logger, prefix : str = ""):
    import subprocess
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info("When " + prefix + f": Current memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_info = smi_output.strip().split('\n')
        for idx, mem in enumerate(memory_info):
            used, total = mem.split(', ')
            logger.info(f"GPU {idx}: used {used}MiB / total {total}MiB")
    except subprocess.CalledProcessError as e:
        logger.info("Can't execute nvidia-smi command:", e.output)
    except FileNotFoundError:
        logger.info("nvidia-smi command not found , make sure nvidia driver has been install successfully.")


def concrete_trace_wrap(model, dummy_input):
    if torch.cuda.is_available():
        try:
            traced_gm = to_fx_graph(model, dummy_input)
            
        except:
            raise Exception("Failed to trace with gpu")
        print("Successfully traced with gpu")
        return traced_gm
    else:
        raise RuntimeError("CUDA is not available")

def check_align(before_trace, after_trace):
    for key in after_trace.keys():
        if isinstance(after_trace[key], torch.Tensor):
            if not torch.allclose(before_trace[key].to(torch.cuda.current_device()) ,after_trace[key].to(torch.cuda.current_device()), rtol=1e-04, atol=1e-05):
                return False
    return True

def is_process_alive(pid):
    try:
        proc = psutil.Process(pid)
        return proc.is_running()
    except psutil.NoSuchProcess:
        return False

# get an available process number from 0 to multiprocessing.cpu_count() - 1
# one process can get different process num when called multiple times
def get_process_num(process_num_list: ListProxy):
    for i in range(len(process_num_list)):
        if process_num_list[i] == multiprocessing.current_process().pid:
            return i
        elif not process_num_list[i] or not is_process_alive(process_num_list[i]):
            process_num_list[i] = multiprocessing.current_process().pid
            return i
    raise RuntimeError(f"No available process number, current {process_num_list}")

def trace_worker(model_name: str, nlp_dir: str, process_num_list: ListProxy, model_name_list: ListProxy):
    p = multiprocessing.Process(target=trace_single_model, args=(model_name, nlp_dir, process_num_list, model_name_list))   # , daemon=True
    p.start()
    p.join()


def trace_single_model(model_name: str, nlp_dir: str, process_num_list: ListProxy, model_name_list: ListProxy):
    try:
        process_num = get_process_num(process_num_list)
        logger = setup_logger(os.path.join(nlp_dir, f'all4debug_{process_num}.log'), logging.INFO)

        model_loaded = False
        success_traced = False
        logger_tried = setup_logger2(os.path.join(nlp_dir, 'tried'), logging.INFO)
        logger_loaded = setup_logger2(os.path.join(nlp_dir, 'model_loaded'), logging.INFO)
        logger_traced = setup_logger2(os.path.join(nlp_dir, 'success_traced'), logging.INFO)
        logger_failed = setup_logger2(os.path.join(nlp_dir, 'model_failed'), logging.INFO)
        logger_errors = setup_logger2(os.path.join(nlp_dir, 'errors'), logging.INFO)

        start_time = time.time()
        logger.info(f"start trying model: {model_name}")
  
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"{model_name} Tokenizer loaded")

        dummy_input = tokenizer(text, return_tensors="pt")
        logger.info(f"{model_name} tokenized")
        if isinstance(dummy_input, MutableMapping):
            dummy_input = dict(dummy_input)

        logger.debug(f"dummy_input: {dummy_input}")

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"{model_name} config loaded")
        model = load_model_with_timeout(config, trust_remote_code=True)
        model_loaded = True
        logger.info(f"{model_name} model loaded")
        logger.debug(f"{model_name} has parameter: {sum(p.numel() for p in model.parameters())}")
        print_memory_usage(logger, f"after load model {model_name}")
        
        model.eval()
        before_trace = model(**dummy_input)
        logger.debug(f"original logit: {before_trace['last_hidden_state'][:2]}")

        traced_gm = concrete_trace_wrap(model, dummy_input)
        success_traced = True
        logger.info(f"{model_name} model traced")
        traced_gm.eval()
        after_trace = traced_gm(**dummy_input)
        logger.debug(f"traced logit: {after_trace['last_hidden_state'][:2]}")

        assert check_align(before_trace, after_trace), "Traced model does not match the original model"
        logger.info(f"aligned before and after trace: {model_name}, {config.architectures}")

    except (Exception, TimeoutError) as e:
        logger.error(f"fail when trying model: {model_name}", exc_info=False)
        logger_failed.info(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        error_message = traceback.format_exc()

        logger_errors.error(f"{model_name} failed")
        if 'config' in locals() and config:
            logger_errors.error(f"Architectures: {config.architectures}")
        logger_errors.error(error_message)
    finally:
        end_time = time.time()
        logger.info(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")

        logger_tried.info(f"{model_name}")
        if model_loaded:
            logger_loaded.info(f"{model_name}, {config.architectures}")
        if success_traced:
            logger_traced.info(f"{model_name}, {config.architectures}")
        model_name_list.remove(model_name)
        logger.info(f"Left models number: {len(model_name_list)}\n")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    model_name_set_path = os.path.join(current_folder, "huggingface_model_names/model_name_set_nlp")
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

    process_num = multiprocessing.cpu_count() - 20
    with multiprocessing.Manager() as manager:
        model_name_list = manager.list(model_name_set)
        
        process_num_list = manager.list([0] * process_num)
        arguments = [(model_name, nlp_dir, process_num_list, model_name_list) for model_name in model_name_set]
        with ThreadPool(processes = process_num) as pool:
            pool.starmap(trace_worker, arguments)

