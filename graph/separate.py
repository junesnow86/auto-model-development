# 读取一号文件（模型集合）
def read_file1(file1_path):
    with open(file1_path, 'r') as file:
        content = file.read()
        model_set = eval(content)  # 将字符串转换为集合
        return set(model_set)  # 确保是集合类型

# 读取二号文件（模型列表），并去重
def read_and_deduplicate_file2(file2_path):
    with open(file2_path, 'r') as file:
        model_list = file.read().splitlines()  # 读取每一行并转换为列表
        return set(model_list)  # 转换为集合去重

# 修改后的分配模型函数
def distribute_models(file1_models, file2_models, base_filename, n):
    # 计算差集
    remaining_models = list(file1_models - file2_models)
    # 分配到n个文件
    models_per_file = len(remaining_models) // n

    for i in range(n):
        # 计算分片的起始和结束索引
        start_index = i * models_per_file
        # 如果是最后一个文件，则包含所有剩余的模型
        if i == n - 1:
            end_index = len(remaining_models)
        else:
            end_index = start_index + models_per_file

        # 获取当前分片的模型
        current_models = remaining_models[start_index:end_index]
        # 写入文件
        with open(f'{base_filename}_{i+1}', 'w') as file:
            file.writelines('\n'.join(current_models))


# 主函数
def main(file1_path, file2_path, n):
    # 读取一号文件和二号文件
    file1_models = read_file1(file1_path)
    file2_models = read_and_deduplicate_file2(file2_path)

    # 分配模型到n个文件
    base_filename = file1_path.split('.')[0]  # 假设文件名没有多个'.'，否则需要其他方法分割
    distribute_models(file1_models, file2_models, base_filename, n)

if __name__ == "__main__":
    # 假设n为2
    n = 2
    main('file1.txt', 'file2.txt', n)
