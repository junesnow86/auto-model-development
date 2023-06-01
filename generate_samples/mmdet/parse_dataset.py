import os
import warnings

warnings.filterwarnings("ignore")
from mmcv import Config
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_config(config_path):
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    return cfg

datasets = set()

mmdet_config_root = "/home/v-junliang/DNNGen/concrete_trace_test/mmdetection/configs"
for dir in os.listdir(mmdet_config_root):
    if dir == "_base_":
        continue
    for config_file in os.listdir(os.path.join(mmdet_config_root, dir)):
        if config_file.endswith(".py"):
            config_path = os.path.join(mmdet_config_root, dir, config_file)
            cfg = parse_config(config_path)
            datasets.add(cfg.data["train"]["type"].replace("Dataset", ""))
            datasets.add(cfg.data["val"]["type"].replace("Dataset", ""))
            datasets.add(cfg.data["test"]["type"].replace("Dataset", ""))
print(datasets)
