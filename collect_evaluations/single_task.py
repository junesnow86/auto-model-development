import os
from task_test import *
from parse_utils import *

if __name__ == "__main__":
    # task = "cv/object-detection"
    # task = "cv/image-segmentation"
    # task = "cv/depth-estimation"
    # task = "cv/zero-shot-image-classification"
    # task = "cv/video-classification"
    task = "nlp/token-classification"
    task_root = "/home/yileiyang/workspace/DNNGen/concrete_trace_test/collect_evaluations/evaluation_results"
    task_dir = os.path.join(task_root, task)
    count = parse_task(task, task_dir, default_datasets, default_metrics, default_dataset_metric)
    print(f"task {task} parsed results count = {count}")