import os
import re
import requests
import pandas
import numpy as np
from bs4 import BeautifulSoup

default_datasets = {
    "MNLI", 
    "QNLI", 
    "RTE",
    "WNLI",
    "CoLA",
    "SST-2",
    "MRPC",
    "STS-B",
    "QQP",

    "LAMBADA", 
    "CBT-CN",
    "CBT-NE",
    "WikiText2",
    "WikiText103",
    "PTB",
    "enwiki8",
    "text8",
    "1BW",

    "SNLI",
    "SciTail",
    "RACE",
    "ROCStories",
    "COPA",
    "SQuAD 1.1",
    "SQuAD 2.0",

    "Assin",
    "Assin2",
    "stsb_multi_mt", 
    "SemEval-2015",
        
    "MasakhaNER",

    "XLSum"
}

default_dataset_metric = {
    "MNLI": "matched/mismatched ACC",
    "QNLI": "ACC",
    "RTE": "ACC",
    "WNLI": "ACC",
    "CoLA": "Matthews corr",
    "SST-2": "ACC",
    "MRPC": "ACC/F1",
    "STS-B": "Pearson/Spearman corr",
    "QQP": "ACC/F1",

    "SNLI": "ACC",
    "SciTail": "ACC",
    "RACE": "ACC",
    "ROCStories": "ACC",
    "COPA": "ACC",
    "SQuAD 1.1": "F1/EM",
    "SQuAD 2.0": "F1/EM",

    "Assin": "STS",
    "Assin2": "STS",
    "stsb_multi_mt": "STS",
    "SemEval-2015": "STS",

    "MasakhaNER": "F1"
}

default_metrics = {
    "ACC",
    "F1",
    "PPL",
    "Rouge",
    "Gen Len",
    "Bertscore",
    "cosine",
    "pearson",
    "spearman",
    "euclidean",
    "manhattan"
}


class Record:
    def __init__(self, dataset, model, metric, value):
        self._dataset = dataset
        self._model = model
        self._metric = metric
        self._value = value

    def __str__(self):
        display = f"dataset: {self._dataset}, model: {self._model}, metric: {self._metric}, value: {self._value}"
        return display


class DatasetTable:
    def __init__(self, dataset, metric=None):
        self._dataset = dataset
        self._metric = metric
        self._map = {}
        self._records = []

    @property
    def dataset(self):
        return self._dataset

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, metric):
        self._metric = metric

    def add_record(self, model, value):
        self._map[model] = value

    def show(self):
        print(f"Dataset: {self._dataset}, metric: {self._metric}")
        for model, value in self._map.items():
            print(f"model: {model}, value: {value}")

    def __str__(self):
        display = f"Dataset: {self._dataset}, metric: {self._metric}\n"
        for model, value in self._map.items():
            display += f"model: {model}, value: {value}\n"
        return display


def is_all_decimal(values: np.ndarray):
    decimal = "(?<!([A-Za-z]| ))(([0-9]+\.[0-9]+)|-)(?![A-Za-z])"
    decimal_types = [np.float_, np.int_, type(np.nan)]
    for value in values:
        if isinstance(value, str):
            match = re.search(value, decimal)
            if match is None:
                return False
        elif type(value) in decimal_types:
            continue
        else:
            raise TypeError(f"{value} has invalid value type: {type(value)}")
    return True


def has_no_decimal(values: np.ndarray):
    decimal = "(?<!([A-Za-z]| ))(([0-9]+\.[0-9]+)|-)(?![A-Za-z])"
    decimal_types = [np.float_, np.int_, type(np.nan)]
    for value in values:
        if type(value) in decimal_types:
            return False
        elif isinstance(value, str):
            match = re.search(decimal, value)
            if match:
                return False
        else:
            raise TypeError(f"{value} has invalid value type: {type(value)}")
    return True

        
def match_singleton(item: str, keywords: set):
    for key in keywords:
        match = re.search(key.lower(), item.lower())
        if match:
            return key
    return None


def match_set(items: set, default_datasets: set):
    if items & default_datasets:
        return True
    else:
        for item in items:
            if not isinstance(item, str):
                continue
            match = match_singleton(item, default_datasets)
            if match:
                return True
        return False


def match_datasets_by_row(data: pandas.DataFrame, default_datasets: set):
    for index in data.index:
        try:
            items = set(data.loc[index].values)
        except TypeError:
            print(index)
            data.drop(index, inplace=True)
        if match_set(items, default_datasets):
            return index
    return None


def match_datasets_modelcard(model: str, default_datasets: set):
    datasets = []
    url = "https://huggingface.co/" + model
    html = requests.get(url=url).text
    soup = BeautifulSoup(html, "lxml")
    for dataset in default_datasets:
        pattern = "(?<![A-Za-z])" + dataset + "(?![A-Za-z])"
        match_results = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        if match_results:
            datasets.append(dataset)
    return datasets


def find_keywords_in_row(row, keywords: set):
    min_index = len(row)
    max_index = -1
    for i in range(len(row)):
        match = match_singleton(row[i], keywords)
        if match is not None:
            if i < min_index:
                min_index = i
            if i > max_index:
                max_index = i
    if min_index < len(row) and max_index > -1:
        return min_index, max_index
    else:
        return None


def find_keywords_in_col(col, keywords: set):
    min_index = len(col)
    max_index = -1
    for i in range(len(col)):
        match = match_singleton(col[i], keywords)
        if match is not None:
            if i < min_index:
                min_index = i
            if i > max_index:
                max_index = i
    if min_index < len(col) and max_index > -1:
        return min_index, max_index
    else:
        return None


def first_column_as_index(data: pandas.DataFrame, default_metrics: set):
    keywords = {"dataset", "model", "task", "language", "testset"}
    first_column_name = data.columns[0]
    match_key = match_singleton(first_column_name, keywords)
    if match_key is not None:
        return match_key
    else:
        match = re.search("unnamed", first_column_name, re.IGNORECASE)
        if match is not None:
            # consider the first column whether is metrics
            indices = find_keywords_in_col(data[data.columns[0]], default_metrics)
            if indices is not None:
                data.rename(columns={data.columns[0]: "metric"}, inplace=True)
                return "metric"
            else:
                return "unnamed"
        else:
            return None
    

def columns_contain_model(columns, model):
    model_name = model.split("/")[-1]
    model_name_segs = model_name.split("-")
    for column in columns:
        # match = re.search(model_name.lower(), column.lower())
        match_key = match_singleton(column, model_name_segs)
        if match_key is not None:
            return column
    return None


def is_row_valid(row: np.ndarray):
    valid = "[A-Za-z0-9]+"
    decimal_types = [np.float_, np.int_, type(np.nan)]
    for item in row:
        if isinstance(item, str):
            match = re.search(valid, item)
            if match is not None:
                return True
        elif type(item) in decimal_types:
            return True
        else:
            return False
    return False


def delete_invalid_rows(data: pandas.DataFrame):
    for index in data.index:
        if not is_row_valid(data.loc[index].values):
            data.drop(index, inplace=True)


def write_exceptions(file_path, model):
    with open(file_path, "w") as f:
        f.write(model)
        f.write("\n")


def parse(data: pandas.DataFrame, default_datasets: set, default_metrics: set, default_dataset_metric: dict, model: str):
    delete_invalid_rows(data)

    exceptions_dir = "exceptions"
    if data.index.dtype == np.int_ and first_column_as_index(data, default_metrics) is not None:
        data.set_index(data.columns[0], inplace=True)  # set the first column as index
    else:
        print(f"{model}: first column not suitable to be index")
        write_exceptions(os.path.join(exceptions_dir, "first_column_index.txt"), model)
        return []

    print(f"-----model: {model}-----")
    print(data)

    
    dataset_tables = []
    dataset_columns = find_keywords_in_row(data.columns.values, default_datasets)
    if dataset_columns is not None:
        min_column, max_column = dataset_columns
        datasets = data.columns.values[min_column:max_column+1]
        for i, dataset in enumerate(datasets):
            dt = DatasetTable(dataset)
            for index in data.index:
                if has_no_decimal(data.loc[index].values):
                    # the whole row contains no decimal, consider whether it's about metric
                    if isinstance(index, str) and "metric" in index.lower():
                        # this row contains metric information
                        dt.metric = data.at[index, dataset]

                if data.index.name.lower() == "Model".lower():
                    dt.add_record(index, data.at[index, dataset])
                else:
                    dt.add_record(model, data.at[index, dataset])
            dataset_tables.append(dt)
    else:
        # try to find datasets in rows
        match_index = match_datasets_by_row(data, default_datasets)
        if match_index is not None:
            datasets = data.loc[match_index].values
            data.drop(match_index, inplace=True)
            for i, dataset in enumerate(datasets):
                dt = DatasetTable(dataset)
                for index in data.index:
                    dt.add_record(model, data.at[index, data.columns[i]])
                dataset_tables.append(dt)
        else:
            # try to find datasets information in the model card
            datasets = match_datasets_modelcard(model, default_datasets)
            if datasets:
                if len(datasets) == 1:
                    dataset = datasets[0]
                    column = columns_contain_model(data.columns, model)
                    if column is not None:
                        data.rename(columns={column: model}, inplace=True)
                        for index in data.index:
                            dt = DatasetTable(dataset + "-" + str(index)) # index可以看成是dataset里的一个branch
                            for col in data.columns:
                                dt.add_record(col, data.at[index, col])
                            dataset_tables.append(dt)
                    else:
                        print(f"{model}: columns are not models.")
                        write_exceptions(os.path.join(exceptions_dir, "columns_not_models.txt"), model)
                        return []
                else:
                    print(f"There are at least 2 datasets in the {model} model card.")
                    print("datasets: ", end="")
                    for dataset in datasets:
                        print(dataset, end=", ")
                    print()
                    write_exceptions(os.path.join(exceptions_dir, "2datasets_in_modelcard.txt"), model)
                    return []
            else:
                write_exceptions(os.path.join(exceptions_dir, "datasets_not_found.txt"), model)
                print("datasets not found")
                return []

    if default_dataset_metric:
        for i in range(len(dataset_tables)):
            if dataset_tables[i].metric is None:
                key = match_singleton(dataset_tables[i].dataset, default_dataset_metric.keys())
                if key:
                    dataset_tables[i].metric = default_dataset_metric[key] + "(default)"
    
    print("---------------------")
    return dataset_tables
    

def parse_csv(csv_path, default_datasets, default_metrics, default_dataset_metric):
    exceptions_dir = "exceptions"
    model = csv_path.split("/")[-1][:-4]
    try:
        data = pandas.read_csv(csv_path)
        print(data)
    except:
        print(f"{model}: read csv error")
        write_exceptions(os.path.join(exceptions_dir, "read_csv_error.txt"), model)
        return []
    else:
        dataset_tables = parse(data, default_datasets, default_metrics, default_dataset_metric, model)
        return dataset_tables


def is_pass(model, pass_list):
    if model in pass_list:
        return True
    else:
        return False

            
def parse_task(task_dir, default_datasets, default_metrics, default_dataset_metric):
    with open("pass.txt", "r") as f:
        pass_list = eval(f.readline())

    exceptions_dir = "exceptions"
    for item in os.listdir(task_dir):
        if os.path.isdir(os.path.join(task_dir, item)):
            developer_dir = os.path.join(task_dir, item)
            print(f"developer: {item}")
            for csv in os.listdir(developer_dir):
                model = os.path.join(item, csv[:-4])
                if is_pass(model, pass_list):
                    print(f"pass {model}")
                    continue

                csv_path = os.path.join(developer_dir, csv)
                dataset_tables = parse_csv(csv_path, default_datasets, default_metrics, default_dataset_metric)
                if dataset_tables is None:
                    write_exceptions(os.path.join(exceptions_dir, "return_None.txt"), csv[:-4])
                else:
                    print(dataset_tables)
          
        else:
            if is_pass(item, pass_list):
                print(f"pass {item}")
                continue

            csv_path = os.path.join(task_dir, item)
            dataset_tables = parse_csv(csv_path, default_datasets, default_metrics, default_dataset_metric)
            if dataset_tables is None:
                write_exceptions(os.path.join(exceptions_dir, "return_None.txt"), item[:-4])
            else:
                print(dataset_tables)
        


def print_tables(dataset_tables):
    for dt in dataset_tables:
        print(dt)



if __name__ == "__main__":

    # task_dir = "/home/v-junliang/DNNGen/concrete_trace_test/collect_evaluations/evaluation_results/nlp/sentence-similarity"
    # parse_task(task_dir, default_datasets, default_metrics, default_dataset_metric)

    csv_path = "/home/v-junliang/DNNGen/concrete_trace_test/collect_evaluations/evaluation_results/nlp/sentence-similarity/lighteternal/stsb-xlm-r-greek-transfer"
    parse_csv(csv_path, default_datasets, default_metrics, default_dataset_metric)
