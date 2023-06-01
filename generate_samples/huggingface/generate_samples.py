import json
import os
import sys

sys.path.append("/home/v-junliang/DNNGen/auto_model_dev/generate_samples")
from generate_utils import Sample


def get_ds_desc(ds_index_name):
    file_path = os.path.join("/data/data0/v-junliang/DNNGen/auto_model_dev/dataset_desc", ds_index_name)
    with open(file_path, "r") as f:
        ds_desc = json.load(f)
    return ds_desc["name"], ds_desc["content"]

def get_task_desc(task_index_name):
    file_path = os.path.join("/data/data0/v-junliang/DNNGen/auto_model_dev/task_desc", task_index_name)
    with open(file_path, "r") as f:
        task_desc = json.load(f)
    return task_desc["name"], task_desc["content"]

def get_graphml(model_name):
    # TODO: specify the subtype or move all graphml files to one folder
    file_path = os.path.join("/data/data0/v-junliang/DNNGen/auto_model_dev/xml/huggingface/nlp", model_name)
    with open(file_path, "r") as f:
        return f.readline()

def create_sample(model_name, task_index_name, ds_index_name):
    task_formal_name, task_desc = get_task_desc(task_index_name)
    ds_formal_name, ds_desc = get_ds_desc(ds_index_name)
    graphml = get_graphml(model_name)
    return Sample(model_name, task_formal_name, task_desc, ds_formal_name, ds_desc, graphml)

if __name__ == "__main__":
    sample_save_root = "/data/data0/v-junliang/DNNGen/auto_model_dev/samples/huggingface"
    mapping_root = "/data/data0/v-junliang/DNNGen/auto_model_dev/mapping/huggingface/triples"
    for task in os.listdir(mapping_root):
        if not os.path.exists(os.path.join(sample_save_root, task)):
            os.mkdir(os.path.join(sample_save_root, task))
        for ds in os.listdir(os.path.join(mapping_root, task)):
            if not os.path.exists(os.path.join(sample_save_root, task, ds)):
                os.mkdir(os.path.join(sample_save_root, task, ds))
            for model in os.listdir(os.path.join(mapping_root, task, ds)):
                try:
                    sample = create_sample(model, task, ds)
                except:
                    continue
                with open(os.path.join(sample_save_root, task, ds, model), "w") as f:
                    print(sample, file=f, end='')
                print(f"{task} {ds} {model} done")

    print("done")
