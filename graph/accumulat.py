import os

# summarize the model names of each file under huggingface_model_names folder
folder_path = "/home/yileiyang/workspace/auto-model-compiler/graph/huggingface_model_names"

folder_sets = {}

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            folder_sets[filename] = eval(content)

for folder, folder_set in folder_sets.items():
    print(f"{folder}: {len(folder_set)}")

union_set = set.union(*folder_sets.values())

print(f"Union Set Size: {len(union_set)}")
