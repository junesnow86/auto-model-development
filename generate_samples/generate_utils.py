import os
from typing import Dict, Tuple, Union

class Sample:
    def __init__(self, 
                 name, 
                 task,
                 task_description, 
                 dataset,
                 dataset_description, 
                 graphml):
        self.name = name
        self.task = task
        self.task_desc = task_description
        self.dataset = dataset
        self.ds_desc = dataset_description
        self.graphml = graphml

    def __repr__(self) -> str:
        ret = f"Name: {self.name}\n"
        ret +=  f"Prompt: This model is used for the {self.task} task. "\
                f"{self.task_desc} "\
                f"And the dataset used in training/evaluation is {self.dataset}. "\
                f"{self.ds_desc} "\
                f"Next follows the `Expected completion`, which is the torch.fx graph structure of this model, in the xml format.\n"
        ret += f"Expected completion: \n{self.graphml}"
        return ret

def transform_task_str(origin: str) -> str:
    return origin.replace("-", " ").title()

def get_task_description(task: str) -> str:
    task_description_path = "/mnt/msrasrg/yileiyang/DNNGen/task_descriptions"
    with open(os.path.join(task_description_path, task)) as f:
        return f.readline()

def dataset_to_task_description(dataset: str) -> str:
    """
    Args:
        dataset: The name of the dataset.

    Returns:
        task_description: The description of tasks that the dataset is used for.
    """
    dataset_to_tasks_path = "/mnt/msrasrg/yileiyang/DNNGen/dataset-tasks/images"
    with open(os.path.join(dataset_to_tasks_path, dataset)) as f:
        tasks = eval(f.readline())
    description = ""
    count = 0
    for task in tasks:
        task = transform_task_str(task)
        ds_collected = get_task_description(task).rstrip(".\n")
        description += f"{task}: {ds_collected}; "
        count += 1
        if count >= 5:
            break
    return description.rstrip("; ")

def dataset_dict_to_description(dataset_dict: Dict[str, Union[Tuple[str], str]]) -> str:
    """
    Args:
        dataset_dict: The dictionary of datasets.

    Returns:
        description: The description union.
    """
    used = set()
    description = ""
    for dataset in dataset_dict.values():
        if dataset not in used:
            description += dataset_to_task_description(dataset) + "; "
            used.add(dataset)
    return description.rstrip("; ") + "."
