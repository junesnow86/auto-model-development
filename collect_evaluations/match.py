import os
import re
import pandas


def match(file_path, datasets):
    try:
        data = pandas.read_csv(file_path)
    except:
        print(f"read csv error: {file_path}")
        return False
    for dataset in datasets:
        match = re.search(dataset, str(data))
        if match:
            return True
    return False
    
    

nlp_datasets = [
    "MNLI",
    "MNLI-(m/mm)",
    "MNLI-m",
    "QQP",
    "QNLI",
    "SST-2",
    "CoLA",
    "STS-B",
    "MRPC",
    "RTE",
    "SQuAD",
]

root = "evaluation_results/nlp"
for task in os.listdir(root):
    ret = None
    task_dir = os.path.join(root, task)
    for item in os.listdir(task_dir):
        item_path = os.path.join(task_dir, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                ret = match(file_path, nlp_datasets)
                if not ret:
                    break
        else:
            ret = match(file_path, nlp_datasets)
        if not ret:
            break
    if not ret:
        print(f"New dataset found: {file_path}")
        break

