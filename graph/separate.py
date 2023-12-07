# read file1 with all models
def read_file1(file1_path):
    with open(file1_path, 'r') as file:
        content = file.read()
        model_set = eval(content)
        return set(model_set)

# read file2 with tried models
def read_and_deduplicate_file2(file2_path):
    with open(file2_path, 'r') as file:
        model_list = file.read().splitlines()
        return set(model_list)


def distribute_models(file1_models, file2_models, base_filename, n):
    remaining_models = list(file1_models - file2_models)
    models_per_file = len(remaining_models) // n

    for i in range(n):
        start_index = i * models_per_file
        if i == n - 1:
            end_index = len(remaining_models)
        else:
            end_index = start_index + models_per_file

        current_models = remaining_models[start_index:end_index]
        models_set_str = repr(set(current_models))
        file_name = f'{base_filename}_{i+1}'
        with open(file_name, 'w') as file:
            file.write(models_set_str)
        print(file_name, len(current_models))

def main(file1_path, file2_path, n):
    file1_models = read_file1(file1_path)
    print(file1_path, len(file1_models))
    file2_models = read_and_deduplicate_file2(file2_path)
    print(file2_path, len(file2_models))

    base_filename = file1_path.split('.')[0]
    distribute_models(file1_models, file2_models, base_filename, n)

if __name__ == "__main__":
    main('/home/yileiyang/workspace/auto-model-compiler/graph/huggingface_model_names/model_name_set_nlp', '/home/yileiyang/workspace/auto-model-compiler/graph/nlp_old/tried', 2)
