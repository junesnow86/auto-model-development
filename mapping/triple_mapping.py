import os
import json


def read_tasks(dataset_index_name):
    read_root = "/data/data0/v-junliang/DNNGen/auto_model_dev/ds2task"
    try:
        with open(os.path.join(read_root, dataset_index_name)) as f:
            tasks = eval(f.readline())
    except:
        # print(f"dataset {dataset_index_name} not found")
        return ()
    return tasks

if __name__ == "__main__":
    with open("model_name_set_all") as f:
        model_name_set = eval(f.read())
    print("#model:", len(model_name_set))

    model2ds_dir = "/data/data0/v-junliang/DNNGen/auto_model_dev/mapping/huggingface/model_to_datasets"
    triple_save_dir = "/data/data0/v-junliang/DNNGen/auto_model_dev/mapping/huggingface/triples"
    total = 0
    mapped = 0
    for model_name in model_name_set:
        model_name = model_name.replace("/", "--")
        total += 1
        if total % 100 == 0:
            print(f"{mapped} / {total} done, all {len(model_name_set)} models")

        if not os.path.exists(os.path.join(model2ds_dir, model_name)):
            # print(f"{model_name} not found")
            continue

        with open(os.path.join(model2ds_dir, model_name)) as f:
            dataset_index_names = eval(f.readline())
        for dataset_index_name in dataset_index_names:
            tasks = read_tasks(dataset_index_name)
            if len(tasks) == 0:
                # print(f"{model_name} {dataset_index_name} no task")
                continue
            for task in tasks:
                triple = {"model": model_name, "dataset": dataset_index_name, "task": task}
                if not os.path.exists(os.path.join(triple_save_dir, task)):
                    os.mkdir(os.path.join(triple_save_dir, task))
                if not os.path.exists(os.path.join(triple_save_dir, task, dataset_index_name)):
                    os.mkdir(os.path.join(triple_save_dir, task, dataset_index_name))
                with open(os.path.join(triple_save_dir, task, dataset_index_name, model_name), 'w') as f:
                    json.dump(triple, f)
        mapped += 1
        # print(f"{model_name} mapping done")

    print("triple mapping done")