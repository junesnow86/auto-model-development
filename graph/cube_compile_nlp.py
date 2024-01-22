# pip install sentencepiece transformers fuzzywuzzy concurrent-log-handler psutilfrom multiprocessing.managers import ListProxy
import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from _collections_abc import MutableMapping

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from cube.graph.parser.converter import to_fx_graph

import psutil
from concurrent_log_handler import ConcurrentRotatingFileHandler 
import logging
import time
import timeout_decorator
from cube.runtime.utils import microbatches
import cube
from examples.utils import get_policy
import examples.mlp.policy.gallery as gallery
from functools import partial
import inspect


text: str = "Huggingface is a really excellent project!"
cache_dir: str = "/mnt/msrasrg/yileiyang/hf_cache"

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
model_name_set_path = os.path.join(current_folder, "huggingface_model_names/nlp_test")
nlp_dir = os.path.join(current_folder, "nlp")

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

logger = setup_logger(os.path.join(nlp_dir, f'cube_compile_info.log'), logging.INFO)
error_logger = setup_logger(os.path.join(nlp_dir, f'cube_compile_error.log'), logging.WARNING)

torch.set_printoptions(edgeitems=2)
cube.init()

# get policy
policy = get_policy([gallery], "PASMegatronTP")
policy = partial(policy, nmicros=64//64, tp_size=2)

@timeout_decorator.timeout(120, timeout_exception=TimeoutError)
def load_model_with_timeout(config, trust_remote_code):
    torch.manual_seed(0)
    return AutoModel.from_config(config, trust_remote_code=trust_remote_code)

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
            if not torch.allclose(before_trace[key].to(torch.cuda.current_device()) ,
                                  after_trace[key].to(torch.cuda.current_device()), 
                                  rtol=1e-04, atol=1e-05):
                return False
    return True

def cube_compile_check(model_name: str):
    try:
        start_time = time.time()
        logger.info(f"start trying model: {model_name}")
  
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"{model_name} Tokenizer loaded")

        dummy_input = tokenizer(text, return_tensors="pt")
        if isinstance(dummy_input, MutableMapping):
            dummy_input = dict(dummy_input)
        logger.debug(f"{model_name} tokenized")
        logger.debug(f"dummy_input: {dummy_input}")

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"{model_name} config loaded")

        model = load_model_with_timeout(config, trust_remote_code=True)
        logger.info(f"{model_name} model loaded")
        logger.debug(f"{model_name} has parameter: {sum(p.numel() for p in model.parameters())}")
        print_memory_usage(logger, f"after load model {model_name}")

        model.eval()
        before_trace = model(**dummy_input)
        logger.debug(f"original logit: {before_trace['last_hidden_state'][:2]}")

        traced_gm = concrete_trace_wrap(model, dummy_input)
        logger.info(f"{model_name} model traced")
        traced_gm.eval()
        after_trace = traced_gm(**dummy_input)
        logger.debug(f"traced logit: {after_trace['last_hidden_state'][:2]}")

        if check_align(before_trace, after_trace):
            logger.info(f"aligned before and after trace: {model_name}, {config.architectures}")
        else:
            logger.error(f"not aligned before and after trace: {model_name}, {config.architectures}")
            error_logger.error(f"not aligned before and after trace: {model_name}, {config.architectures}")

        # cube compile model
        model = load_model_with_timeout(config, trust_remote_code=True)
        # dummy_input = tokenizer(text, return_tensors="pt")
        forward_signature = inspect.signature(model.forward)
        params_with_defaults = [
            v.default if k not in dummy_input else dummy_input[k].to(torch.cuda.current_device())
            for k, v in forward_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        ]
        logger.debug(f"forward_signature: {forward_signature}")
        logger.debug(f"params_with_defaults: {params_with_defaults}")

        dataloader = microbatches((params_with_defaults, ) * 2)
        @cube.compile(model, dataloader, PAS=policy)
        def train_iter(model, dataloader):
            data = next(dataloader)
            logit = model(*data)
            return logit

        smodel = cube.utils.load_model()
        smodel.eval()
        compiled_logit = train_iter(smodel, dataloader)
        logger.debug(f"compiled logit: {compiled_logit['last_hidden_state'][:2]}")

        if check_align(before_trace, compiled_logit):
            logger.info(f"aligned before trace and after compile: {model_name}, {config.architectures}")
        else:
            logger.error(f"not aligned before trace and after compile: {model_name}, {config.architectures}")
            error_logger.error(f"not aligned before trace and after compile: {model_name}, {config.architectures}")

    except (Exception, TimeoutError) as e:
        logger.error(f"fail when trying model: {model_name}", exc_info=False)
        error_logger.info(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        error_message = traceback.format_exc()

        error_logger.error(f"{model_name} failed")
        if 'config' in locals() and config:
            error_logger.error(f"Architectures: {config.architectures}")
        error_logger.error(error_message)
    finally:
        end_time = time.time()
        logger.info(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")

if __name__ == "__main__":

    with open(model_name_set_path, 'r') as f:
        all_model = eval(f.read())
    print(f"# model: {len(all_model)}")

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

    model_numbers = 0
    for model_name in model_name_set:
        cube_compile_check(model_name)
        model_numbers += 1
        logger.info(f"Process: {model_numbers} / {len(model_name_set)}, Percentage: {model_numbers / len(model_name_set) * 100:.2f}%\n")

