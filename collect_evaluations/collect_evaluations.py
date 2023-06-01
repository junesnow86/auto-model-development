import os
import json
import pandas


def get_table(model_name):
    url = "https://huggingface.co/" + model_name
    try:
        table = pandas.read_html(url, match="(?<![A-Za-z])([0-9]+\.[0-9]+)(?![A-Za-z])")
    except:
        table = []
    return table

def save(evaluation_results, save_file_path):
    if evaluation_results:
        if len(evaluation_results) > 1:
            for i in range(len(evaluation_results)):
                evaluation_results[i].to_csv(save_file_path, mode="a", index=False)
                if i < len(evaluation_results)-1:
                    with open(save_file_path, "a") as f:
                        f.write("\n")
        else:
            evaluation_results[0].to_csv(save_file_path, index=False)


if __name__ == "__main__":
    model_names_root = "model_names_txt"
    save_root = "evaluation_results"
    statistics_root = "statistics"

    exceptions = []

    for subType in os.listdir(model_names_root):
        print(f"-----collecting {subType}...------")

        subType_dir = os.path.join(model_names_root, subType)
        save_subType_dir = os.path.join(save_root, subType)
        if not os.path.exists(save_subType_dir):
            os.mkdir(save_subType_dir)

        statistics = {}
        for filename in os.listdir(subType_dir):
            file_path = os.path.join(subType_dir, filename)
            with open(file_path, "r") as f:
                model_names = f.readline()
            model_names = eval(model_names)

            save_task_dir = os.path.join(save_subType_dir, filename[:-4])
            if not os.path.exists(save_task_dir):
                os.mkdir(save_task_dir)

            print(f"{filename[:-4]} has {len(model_names)} models to try")
            count = 0
            for model_name in model_names:
                splited = model_name.split("/")

                developer_dir = None
                if len(splited) == 2:
                    developer = splited[-2]
                    developer_dir = os.path.join(save_task_dir, developer)
                    save_file_path = os.path.join(developer_dir, splited[-1]+".csv")
                else:
                    save_file_path = os.path.join(save_task_dir, model_name+".csv")
                
                if os.path.exists(save_file_path):
                    count += 1
                    if count > 0 and count % 10 == 0:
                        print(f"{count} models collected")
                    continue

                evaluation_results = get_table(model_name)

                if evaluation_results:
                    if developer_dir is not None and (not os.path.exists(developer_dir)):
                        os.mkdir(developer_dir)

                    if len(evaluation_results) > 1:
                        print(f"{model_name} #results > 1")
                        exceptions.append(model_name)

                    save(evaluation_results, save_file_path)
                    count += 1
                    
                    if count > 0 and count % 10 == 0:
                        print(f"{count} models collected")
                
            
            task_name = filename[:-4]
            statistics[task_name] = count
            print(f"{task_name} done.")
        
        statistics_save_path = os.path.join(statistics_root, subType+".json")
        with open(statistics_save_path, "w") as f:
            json.dump(statistics, f, indent=2)
        
        exceptions_file_path = "exceptions.txt"
        with open(exceptions_file_path, "w") as f:
            f.write(str(exceptions))






