import os

task_set = set()
dataset_tasks_dir_path = "/data/data0/v-junliang/DNNGen/auto_model_dev/ds2task"
print("total dataset: ", len(os.listdir(dataset_tasks_dir_path)))
for file in os.listdir(dataset_tasks_dir_path):
    filepath = os.path.join(dataset_tasks_dir_path, file)
    with open(filepath) as f:
        tasks = eval(f.readline())
    task_set = task_set.union(tasks)
with open("task_set", 'w') as f:
    print(task_set, file=f)
print("task set saved, total task: ", len(task_set))
