import os
import pickle
import sys
sys.path.append("/home/v-junliang/DNNGen/auto_model_dev")
from mapping.utils import Mapping
from generate_graph.utils import Sample
sys.path.append("/home/v-junliang/DNNGen/auto_model_dev/mapping")

def get_dataset_desc(dataset):
    root = "/data/data0/v-junliang/DNNGen/auto_model_dev/dataset_desc"
    with open(os.path.join(root, dataset), "rb") as f:
        ds_desc = pickle.load(f)
    return ds_desc.name, ds_desc.content

def get_task_desc(task):
    root = "/data/data0/v-junliang/DNNGen/auto_model_dev/task_desc"
    with open(os.path.join(root, task), "rb") as f:
        task_desc = pickle.load(f)
    return task_desc.name, task_desc.content

def get_graphml(model_name):
    root = "/data/data0/v-junliang/DNNGen/auto_model_dev/xml/mmdet"
    with open(os.path.join(root, model_name)) as f:
        return f.readline()

def create_sample(mapping: Mapping):
    name = mapping.model_name
    task = mapping.task
    dataset = mapping.dataset
    task_standard_name, task_description = get_task_desc(task)
    ds_standard_name, ds_description = get_dataset_desc(dataset)
    graphml = get_graphml(name)
    return Sample(name, task_standard_name, task_description, ds_standard_name, ds_description, graphml)

if __name__ == "__main__":
    sample_save_root = "/data/data0/v-junliang/DNNGen/auto_model_dev/samples/mmdet"
    mapping_root = "/data/data0/v-junliang/DNNGen/auto_model_dev/mapping/mmdet"
    for task in os.listdir(mapping_root):
        if not os.path.exists(os.path.join(sample_save_root, task)):
            os.mkdir(os.path.join(sample_save_root, task))
        for dataset in os.listdir(os.path.join(mapping_root, task)):
            if not os.path.exists(os.path.join(sample_save_root, task, dataset)):
                os.mkdir(os.path.join(sample_save_root, task, dataset))
            for mapping_file in os.listdir(os.path.join(mapping_root, task, dataset)):
                mapping_path = os.path.join(mapping_root, task, dataset, mapping_file)
                with open(mapping_path, "rb") as f:
                    mapping = pickle.load(f)
                sample = create_sample(mapping)
                if sample is None:
                    continue
                sample_path = os.path.join(sample_save_root, task, dataset, mapping_file)
                with open(sample_path, "w") as f:
                    print(sample, file=f, end='')
    print("done")
