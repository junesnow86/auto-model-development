import os
import sys
import warnings

warnings.filterwarnings("ignore")
from mmcv import Config
from mmdet.utils import replace_cfg_vals, update_data_root

sys.path.append("/home/v-junliang/DNNGen/concrete_trace_test/prepare_data")
from utils import Sample


def parse_dataset(config_path):
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    dataset_dict = {}
    dataset_dict["train"] = cfg.data["train"]["type"].replace("Dataset", "")
    dataset_dict["val"] = cfg.data["val"]["type"].replace("Dataset", "")
    dataset_dict["test"] = cfg.data["test"]["type"].replace("Dataset", "")
    return dataset_dict

datasets = (
    "cityscapes",
    "coco",
    "deepfashion",
    "lvis",
    "obj365",
    "openimages",
    "voc0712",
    "wider_face",
)

samples = []

mmdet_configs_root = "/home/v-junliang/DNNGen/concrete_trace_test/mmdetection/configs"
count = 0
for dir in os.listdir(mmdet_configs_root):
    if dir == "_base_":
        continue
    for config_file in os.listdir(os.path.join(mmdet_configs_root, dir)):
        if config_file.endswith(".py"):
            dataset_dict = parse_dataset(os.path.join(mmdet_configs_root, dir, config_file))
            samples.append(Sample(
                Name=config_file.replace(".py", ""),
                Task="Object Detection",
                Datasets=dataset_dict,
                GraphML="",
            ))
            count += 1
            if count == 5:
                break
        if count == 5:
            break
    if count == 5:
        break

for sample in samples:
    print(sample)
