import os
import pandas
import json
from parse_utils import *

            
def parse_task(task, task_dir, default_datasets, default_metrics, default_dataset_metric):
    count = 0

    # pass_root = "pass"
    # pass_dir = os.path.join(pass_root, task)
    # with open(os.path.join(pass_dir, "pass.txt"), "r") as f:
    #     pass_models = eval(f.readline())
    # with open(os.path.join(pass_dir, "done.txt"), "r") as f:
    #     done_models = eval(f.readline())
    # with open(os.path.join(pass_dir, "pass_developer.txt"), "r") as f:
    #     pass_developers = eval(f.readline())
    # with open(os.path.join(pass_dir, "done_developer.txt"), "r") as f:
    #     done_developers = eval(f.readline())
    pass_developers = []
    done_developers = []
    pass_models = []
    done_models = []
    for item in os.listdir(task_dir):
        if os.path.isdir(os.path.join(task_dir, item)):
            developer_dir = os.path.join(task_dir, item)
            if item in pass_developers:
                print(f"developer {item} is passed")
            elif item in done_developers:
                print(f"developer {item} is done")
                count += len(os.listdir(developer_dir))
            else:
                print(f"parsing developer: {item}")
                for csv in os.listdir(developer_dir):
                    model = os.path.join(item, csv[:-4])
                    if model in pass_models:
                        print(f"model {model} is passed")
                    elif model in done_models:
                        print(f"model {model} is done")
                        count += 1
                    else:
                        records = parse_model(model, default_datasets, default_metrics, default_dataset_metric)
                        if records is not None and len(records) > 0:
                            count += 1
                        # elif records is not None and len(records) == 0:
                            # return False
                        
          
        else:
            if item in pass_models:
                print(f"model {item} is passed")
            elif item in done_models:
                print(f"model {item} is done")
                count += 1
            else:            
                model = item[:-4]
                records = parse_model(model, default_datasets, default_metrics, default_dataset_metric)
                if records is not None and len(records) > 0: 
                    count += 1
                # elif records is not None and len(records) == 0:
                    # return False

    print(f"{count} models done")
    # return True
    return count



if __name__ == "__main__":
    # task = "nlp/sentence-similarity"
    # task = "cv/image-classification"
    count_dict = {}
    task_root = "/home/v-junliang/DNNGen/concrete_trace_test/collect_evaluations/evaluation_results"
    for subType in os.listdir(task_root):
        subType_dir = os.path.join(task_root, subType)
        print(f"parsing subType: {subType}")
        for task in os.listdir(subType_dir):
            print(f"parsing task: {task}")
            task_dir = os.path.join(subType_dir, task)
            count = parse_task(task, task_dir, default_datasets, default_metrics, default_dataset_metric)
            count_dict[task] = count
        save_dir = "statistics"
        with open(os.path.join(save_dir, subType+".json"), "w") as f:
            json.dump(count_dict, f, indent=2)
