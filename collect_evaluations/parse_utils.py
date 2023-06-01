import re
import requests
import pandas
import numpy as np
from bs4 import BeautifulSoup, Tag


default_datasets = {
    # GLUE
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

    "SciTail",
    "RACE",
    "ROCStories",
    "COPA",
    "SQuAD 1.1",
    "SQuAD 2.0",

    # STS
    "Assin",
    "Assin2",
    "stsb_multi_mt", 
    "SemEval-2015",

    # NLI
    "SNLI",
    "XNLI",
    "ESXNLI",
    "MultiNLI",

    # MRC
    "MS-MARCO",
    "TREC",
    "FiQA",
        
    "MasakhaNER",

    "XLSum",

    # Chinese
    "csts",
    "afqmc",
    "lcqmc",
    "bqcorpus",
    "pawsx",
    "xiaobu",
    "ATEC",
    "BQ",

    # Korean
    "KLUE",

    "germanDPR",
    "COALAS",

    # cv
    "ImageNet",
    "CIFAR",
    "Caltech-101",
    "Pascal VOC",
    "COCO",
    "CityScapes",
    "LVIS",
    "CoNLL",

}

default_dataset_metric = {
    # GLUE
    "MNLI": "matched/mismatched ACC",
    "QNLI": "ACC",
    "RTE": "ACC",
    "WNLI": "ACC",
    "CoLA": "Matthews corr",
    "SST-2": "ACC",
    "MRPC": "ACC/F1",
    "STS-B": "Pearson/Spearman corr",
    "QQP": "ACC/F1",

    "SciTail": "ACC",
    "RACE": "ACC",
    "ROCStories": "ACC",
    "COPA": "ACC",
    "SQuAD 1.1": "F1/EM",
    "SQuAD 2.0": "F1/EM",

    # STS
    "Assin": "STS",
    "Assin2": "STS",
    "stsb_multi_mt": "STS",
    "SemEval-2015": "STS",

    # NLI
    "SNLI": "ACC/F1",
    "XNLI": "ACC/F1",
    "ESXNLI": "ACC/F1",
    "MultiNLI": "matched/mismatched ACC/F1",

    # MRC
    "MS-MARCO": "MRR@10",
    "TREC": "NDCG@10",
    "FiQA": "NDCG@10",
    
    # Chinese
    "csts": "ACC/F1",
    "afqmc": "F1",
    "lcqmc": "F1",
    "bqcorpus": "F1",
    "pawsx": "F1",
    "xiaobu": "BLEU",
    "ATEC": "BLEU/Rouge",
    "BQ": "BLEU/TER",

    # Korean
    "KLUE": "ACC/F1",

    
    "germanDPR": "NDCG",

    "MasakhaNER": "F1",

    # cv
    "ImageNet": "Top 1 Performance",
    "CIFAR": "ACC",
    "Caltech-101": "ACC",
    "Pascal VOC": "AP",
    "COCO": "ACC/AP",

}

default_metrics = {
    "ACC",
    "F1",
    "PPL",
    "MRR",
    "precision",
    "recall",
    "NDCG",
    "Rouge",
    "Gen Len",
    "Bertscore",

    "pearson",
    "spearman",
    "cosine",
    "euclidean",
    "manhattan",
    "dot",

    "Top 1 Performance",
    "AP",
    "mAP",
    "MIoU",
}


class Record:
    def __init__(self, dataset, model, metric, value, kwargs=None):
        self._dataset = dataset
        self._model = model
        self._metric = metric
        self._value = value
        self._kwargs = kwargs

    def __str__(self):
        display = f"dataset: {self._dataset}, model: {self._model}, metric: {self._metric}, value: {self._value}, kwargs: {self._kwargs}"
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


def index_has_nan(index: pandas.Index):
    return pandas.isna(index).any()

        
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


def match_datasets_in_rows(data: pandas.DataFrame, default_datasets: set):
    for index in data.index:
        if isinstance(index, str) and (index.lower() == "dataset" or index.lower() == "datasets"):
            # 该行的索引就叫"dataset", 判断这一行就是dataset
            return index
        else:
            row_values = data.loc[index].values
            for value in row_values:
                if not isinstance(value, str):
                    continue
                matched_dataset = match_singleton(value, default_datasets)
                if matched_dataset is not None:
                    return index
    return None


def is_training_dataset(tag: Tag):
    # TODO: 需不需要考虑model description中的关键词顺序, 例如`train`出现在dataset前面
    previous_h2 = tag.find_previous("h2")
    if previous_h2 is not None:
        if "train" in previous_h2.text.lower():
            return True
        else:
            previous_h1 = previous_h2.find_previous("h1")
            if previous_h1 is not None:
                if "train" in previous_h1.text.lower():
                    return True
            return False
    else:
        previous_h1 = tag.find_previous("h1")
        if previous_h1 is not None:
            if "train" in previous_h1.text.lower():
                return True
        return False


def is_evaluation_dataset(tag: Tag):
    eval_keywords = {"eval", "test", "performance", "score"}
    previous_h2 = tag.find_previous("h2")
    if previous_h2 is not None:
        if match_singleton(previous_h2.text, eval_keywords) is not None:
            return True
        else:
            previous_h1 = previous_h2.find_previous("h1")
            if previous_h1 is not None:
                if match_singleton(previous_h1.text, eval_keywords) is not None:
                    return True
            return False
    else:
        previous_h1 = tag.find_previous("h1")
        if previous_h1 is not None and match_singleton(previous_h1.text, eval_keywords) is not None:
            return True
        return False


def exists_evaluation_section(html):
    soup = BeautifulSoup(html, "lxml")

    h1_tags = soup.find_all("h1")
    for tag in h1_tags:
        if "eval" in tag.text.lower():
            return True

    h2_tags = soup.find_all("h2")
    for tag in h2_tags:
        if "eval" in tag.text.lower():
            return True
            
    return False


def match_datasets_modelcard(model: str, default_datasets: set):
    datasets = []
    url = "https://huggingface.co/" + model
    html = requests.get(url=url).text
    soup = BeautifulSoup(html, "lxml")
    for dataset in default_datasets:
        if " " in dataset:
            patterns = []
            patterns.append("(?<![A-Za-z])" + dataset.replace(" ", "-") + "(?![A-Za-z])")
            patterns.append("(?<![A-Za-z])" + dataset.replace(" ", "") + "(?![A-Za-z])")
            for pattern in patterns:
                matched_strings = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                if len(matched_strings) > 0:
                    break
        elif "-" in dataset:
            patterns = []
            patterns.append("(?<![A-Za-z])" + dataset.replace("-", " ") + "(?![A-Za-z])")
            patterns.append("(?<![A-Za-z])" + dataset.replace("-", "") + "(?![A-Za-z])")
            for pattern in patterns:
                matched_strings = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                if len(matched_strings) > 0:
                    break
        else:
            pattern = "(?<![A-Za-z])" + dataset + "(?![A-Za-z])"
            matched_strings = soup.find_all(text=re.compile(pattern, re.IGNORECASE))

        if len(matched_strings) > 1:
            # 该dataset在model card中不止一处出现
            # 只要有一次不是出现在`train` section就认为可以作为eval dataset
            flag = False
            for matched_string in matched_strings:
                if not is_training_dataset(matched_string):
                    flag = True
            if flag:
                datasets.append(dataset)

        elif len(matched_strings) == 1:
            # 该dataset只出现了一次
            tag = matched_strings[0].parent
            if not is_training_dataset(tag):
                datasets.append(dataset)

    return datasets


def check_eval_htag(htag, tag):
    eval_keywords = {"eval", "test", "performance", "score"}
    train_keywords = {"train"}
    previous = tag.find_previous(htag)
    if previous is not None:
        if match_singleton(previous.text, eval_keywords) is not None:
            return True
        elif match_singleton(previous.text, train_keywords) is not None:
            return False
        else:
            return None
    else:
        return None


def get_tables(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    
    eval_keywords = {"eval", "test", "performance", "score"}
    train_keywords = {"train"}
    decimal = "(?<!([A-Za-z]| ))(([0-9]+\.[0-9]+)|-)(?![A-Za-z])"

    valid_tables = []
    for table in tables:
        if re.search(decimal, str(table)) is None:
            continue
        previous_htag = table.find_previous(re.compile(r'h\d'))
        while previous_htag is not None:
            if match_singleton(previous_htag.text, eval_keywords) is not None:
                valid_tables.append(table)
                break
            elif match_singleton(previous_htag.text, train_keywords) is not None:
                break
            else:
                previous_htag = previous_htag.find_previous(re.compile(r'h\d'))

    '''
    for table in tables:
        if re.search(decimal, str(table)) is None:
            continue
        check_h3 = check_eval_htag("h3", table)
        if check_h3 is True:
            valid_tables.append(table)
        elif check_h3 is False:
            continue
        elif check_h3 is None:
            check_h2 = check_eval_htag("h2", table)
            if check_h2 is True:
                valid_tables.append(table)
            elif check_h2 is False:
                continue
            elif check_h2 is None:
                check_h1 = check_eval_htag("h1", table)
                if check_h1 is True:
                    valid_tables.append(table)
                elif check_h1 is False:
                    continue
                elif check_h1 is None:
                    h1 = table.find_previous("h1")
                    if h1 is not None:
                        print(f"h1: {h1.text.strip()}")
                    h2 = table.find_previous("h2")
                    if h2 is not None:
                        print(f"h2: {h2.text.strip()}")
                    h3 = table.find_previous("h3")
                    if h3 is not None:
                        print(f"h3: {h3.text.strip()}")
    '''

    '''
    for table in tables:
        previous_h2 = table.find_previous("h2")
        if previous_h2 is not None:
            if match_singleton(previous_h2.text, eval_keywords) is not None:
                valid_tables.append(table)
            elif match_singleton(previous_h2.text, train_keywords) is not None:
                continue
            else:
                previous_h1 = previous_h2.find_previous("h1")
                if previous_h1 is not None:
                    if match_singleton(previous_h1.text, eval_keywords) is not None:
                        valid_tables.append(table)
                        continue
                    elif match_singleton(previous_h1.text, train_keywords) is not None:
                        continue
                    else:
                        print(f"h1: {previous_h1.text.strip()}")
                        print(f"h2: {previous_h2.text.strip()}")
        else:
            previous_h1 = table.find_previous("h1")
            if previous_h1 is not None:
                if match_singleton(previous_h1.text, eval_keywords) is not None:
                    valid_tables.append(table)
                elif match_singleton(previous_h1.text, train_keywords) is not None:
                    continue
                else:
                    print(f"h1: {previous_h1.text}")
                    print(f"no h2")
    '''
                
    dataframes = []
    for table in valid_tables:
        dataframes.append(pandas.read_html(str(table))[0])
    return dataframes


def find_keywords_in_row(row, keywords: set):
    '''find keywords in the given row
    return the index range of matched keywords
    '''
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


def fillna_with_range(data: pandas.DataFrame, col):
    num_nan = data[col].isnull().sum()
    for i in range(num_nan):
        data[col].fillna(i, limit=1)
    

def preprocess_first_column(data: pandas.DataFrame, model, default_metrics):
    delete_invalid_rows(data)

    if data[data.columns[0]].isnull().all():
        # 第一列的内容全是nan, 则删除第一列, 并将第一列的列名作为columns.name
        data.columns.name = data.columns[0]
        data.drop(data.columns[0], axis=1, inplace=True)
    elif data[data.columns[0]].isnull().any():
        # 第一列存在nan, 但不全为nan, 也认为第一列的列名是columns.name
        # 但不删除第一列, 将第一列作为index
        data.columns.name = data.columns[0]
        fillna_with_range(data, data.columns[0])
        data.set_index(data.columns[0], inplace=True)
        data.index.name = None
    else:
        # 第一列不存在nan值, 根据关键词匹配来判断是否需要set为index
        keywords = {"dataset", "model", "task", "language", "testset", "split", "label"}
        match_key = match_singleton(data.columns[0], keywords)
        if match_key is not None:
            if not data[data.columns[0]].duplicated().any():
                # 第一列没有重复值就可以设为索引
                data.set_index(data.columns[0], inplace=True)
            else:
                print(f"{model} exception case 7: first column values duplicated")
        else:
            match = re.search("unnamed", data.columns[0], re.IGNORECASE)
            if match is not None:
                # 第一列列名是"unnamed"

                # consider the first column whether is metrics
                indices = find_keywords_in_col(data[data.columns[0]], default_metrics)
                if indices is not None:
                    data.rename(columns={data.columns[0]: "metric"}, inplace=True)

                # consider the first columnn whether is models
                models = set()
                models.add(model.split("/")[-1])
                indices = find_keywords_in_col(data[data.columns[0]], models)
                if indices is not None:
                    data.rename(columns={data.columns[0]: "model"}, inplace=True)

                # consider the Teacher-Student pattern
                model_branches = set()
                model_branches.add("teacher")
                model_branches.add("student")
                indices = find_keywords_in_col(data[data.columns[0]], model_branches)
                if indices is not None:
                    data.rename(columns={data.columns[0]: "branch"}, inplace=True)
                    min_index, max_index = indices
                    todrop_indices = [i for i in range(min_index)]
                    todrop_indices.extend([i for i in range(max_index+1, len(data.index))])
                    data.drop(index=data.index[todrop_indices], inplace=True)

                data.set_index(data.columns[0], inplace=True)
            else:
                print(f"New keyword: {data.columns[0]}")


def columns_contain_model(columns, model):
    model_name = model.split("/")[-1]
    model_name_segs = model_name.split("-")
    for column in columns:
        # match = re.search(model_name.lower(), column.lower())
        match_key = match_singleton(column, model_name_segs)
        if match_key is not None:
            return column
    return None


def write_exceptions(file_path, model):
    with open(file_path, "w") as f:
        f.write(model)
        f.write("\n")


def match_metric(dataset, default_metrics):
    # 解析一个dataset字符串中是否含有metric
    key = match_singleton(dataset, default_metrics)
    return key


def contains_metrics(datasets, default_metrics):
    '''If there exists a dataset string containing metric, return True
    '''
    # 如果一行中有一个dataset字符串里含有metric, 就认为都含有
    for dataset in datasets:
        key = match_metric(dataset, default_metrics)
        if key is not None:
            return True
    return False


def get_default_metric(origin, default_dataset_metric):
    key = match_singleton(origin, default_dataset_metric.keys())
    if key is not None:
        metric = default_dataset_metric[key] + "(default)"
    else:
        metric = None
    return metric


def extract_metric(origin: str, default_datasets, default_metrics):
    '''从一个包含dataset和metric的字符串中提取metric
    '''
    dataset = match_singleton(origin, default_datasets)
    if dataset is not None:
        metric = re.sub(dataset, "", origin, flags=re.IGNORECASE)
    else:
        metric = origin

    metric = metric.strip()
    metric = metric.lstrip("(")
    metric = metric.rstrip(")")
    return dataset, metric


def get_metric(origin: str, metric_flag, default_datasets, default_metrics, default_dataset_metric):
    dataset = origin
    if metric_flag:
        dataset, metric = extract_metric(origin, default_datasets, default_metrics)
        if metric is None:
            metric = get_default_metric(origin, default_dataset_metric)
    else:
        metric = get_default_metric(origin, default_dataset_metric)
    return dataset, metric


def parse(data: pandas.DataFrame, model: str, default_datasets: set, default_metrics: set, default_dataset_metric: dict):
    print("-"*45, f"model: {model}", "-"*45)
        

    preprocess_first_column(data, model, default_metrics)

    print("data:")
    print(data)
    print()

    
    records = []

    column_range = find_keywords_in_row(data.columns, default_datasets)
    if column_range is not None:
        # 表头蕴含dataset信息, 列索引是dataset
        min_column, max_column = column_range
        valid_columns = data.columns.values[min_column:max_column+1]
        metric_flag = contains_metrics(valid_columns, default_metrics)

        if data.index.name is not None:
            # if data.index.name.lower() == "model" or data.index.name.lower() == "model name":
            if "model" in data.index.name.lower():
                # 行索引是model
                for column in valid_columns:
                    for index in data.index:
                        dataset, metric = get_metric(column, metric_flag, default_datasets, default_metrics, default_dataset_metric)
                        record = Record(dataset, index, metric, data.at[index, column])
                        records.append(record)
            # elif data.index.name.lower() == "model branch":
            elif data.index.name.lower() == "branch":
                # 认为行索引是model branch
                for column in valid_columns:
                    for index in data.index:
                        model_branch = model + "-" + str(index)
                        dataset, metric = get_metric(column, metric_flag, default_datasets, default_metrics, default_dataset_metric)
                        record = Record(dataset, model_branch, metric, data.at[index, column])
                        records.append(record)
            else:
                print(f"{model} exception case 9: New index name")
        else:
            # 行索引不是model
            if index_has_nan(data.index):
                # index里含有nan值, 猜测index无实际意义, 整个表是关于model在各个数据集的测试, 那么最多只有两行
                if len(data.index) == 1:
                    # 只有一行, 这一行就是测试结果数值
                    for column in valid_columns:
                        dataset, metric = get_metric(column, metric_flag, default_datasets, default_metrics, default_dataset_metric)
                        record = Record(dataset, model, metric, data.at[data.index[0], column])
                        records.append(record)
                elif len(data.index) == 2:
                    # 有两行, 猜测一行是metric
                    if isinstance(data.index[0], str) and "metric" in data.index[0].lower():
                        for column in valid_columns:
                            dataset = column
                            metric = data.at[data.index[0], column]
                            value = data.at[data.index[1], column]
                            record = Record(dataset, model, metric, value)
                            records.append(record)
                    else:
                        print(f"{model}: exception case 1")
                else:
                    print(f"{model} exception case 2")
            else:
                # 行索引中无nan值
                print(f"{model} exception case 3")
    else:
        # 表头不含dataset, 先尝试从行或者列中寻找dataset, 行和列中都没有时尝试从model card中找
        matched_index = match_datasets_in_rows(data, default_datasets)
        if matched_index is not None:
            # 某一行中含有dataset
            datasets = data.loc[matched_index].values
            data.drop(matched_index, inplace=True)
            metric_flag = contains_metrics(datasets, default_metrics)
            for i, dataset in enumerate(datasets):
                for index in data.index:
                    # 行索引不可能是model
                    dataset, metric = get_metric(dataset, metric_flag, default_datasets, default_metrics, default_dataset_metric)
                    record = Record(dataset, model, metric, data.at[index, data.columns[i]])
                    records.append(record)
        else:
            # try to find datasets in model card
            datasets = match_datasets_modelcard(model, default_datasets)
            if datasets:
                if len(datasets) == 1:
                    # model card中只出现了一个dataset
                    dataset = datasets[0]

                    print(dataset)

                    _, default_metric = get_metric(dataset, False, default_datasets, default_metrics, default_dataset_metric) # TODO: 这里是不是也应该试着从model card中查找metric?
                    # 开始分析列和行的意义
                    column = columns_contain_model(data.columns, model)
                    if column is not None:
                        # 列名中含有该model, 判断列名是models
                        data.rename(columns={column: model}, inplace=True)
                        if data.index.name == "metric":
                            for col in data.columns:
                                for index in data.index:
                                    record = Record(dataset, col, index, data.at[index, col])
                                    records.append(record)
                        else:
                            for col in data.columns:
                                for index in data.index:
                                    branch = dataset + "-" + str(index)
                                    record = Record(branch, col, default_metric, data.at[index, col])
                                    records.append(record)
                    elif data.columns.name is not None:
                        if data.columns.name.lower() == "language":
                            # 列名是language
                            for col in data.columns:
                                for index in data.index:
                                    branch = dataset + "-" + col
                                    record = Record(branch, model, metric, data.at[index, col])
                                    records.append(record)
                    else:
                        metric_range = find_keywords_in_row(data.columns, default_metrics)
                        if metric_range is not None:
                            # 列名是metric
                            min_column, max_column = metric_range
                            columns = data.columns[min_column:max_column+1]
                            if data.index.name is not None:
                                if data.index.name.lower() == "model":
                                    for col in columns:
                                        for index in data.index:
                                            record = Record(dataset, index, col, data.at[index, col])
                                            records.append(record)
                                elif data.index.name.lower() == "metric":
                                    for col in columns:
                                        for index in data.index:
                                            metric = index + "-" + col
                                            record = Record(dataset, model, metric, data.at[index, col])
                                            records.append(record)
                                elif data.index.name.lower() == "split":
                                    for col in columns:
                                        for index in data.index:
                                            branch = dataset + "-" + str(index)
                                            record = Record(branch, model, col, data.at[index, col])
                                            records.append(record)
                                else:
                                    print(f"{model} exception case 8: data.index.name = {data.index.name}")
                                    for col in columns:
                                        for index in data.index:
                                            branch = dataset + "-" + str(index)
                                            record = Record(branch, model, col, data.at[index, col])
                                            records.append(record)
                            else:
                                # data.index.name is None
                                for col in columns:
                                    for index in data.index:
                                        record = Record(dataset, model, col, data.at[index, col])
                                        records.append(record)
                        else:
                            print(f"{model} exception case 4")
                else:
                    print(f"{model} exception case 5")
                    print(f"There are at least 2 datasets in the {model} model card.")
                    print("datasets: ", end="")
                    for dataset in datasets:
                        print(dataset, end=", ")
                    print()
            else:
                print(f"{model} exception case 6: no matched datasets in model card")

    
    

    print(f"{len(records)} records:")
    if records:
        for record in records:
            print(record)
    else:
        print(records)
    print("-"*100)

    return records
    

def parse_model(model, default_datasets, default_metrics, default_dataset_metric):
    url = "https://huggingface.co/" + model
    tables = get_tables(url)
    records = []
    if tables:
        for data in tables:
            try:
                parsed = parse(data, model, default_datasets, default_metrics, default_dataset_metric)
            except:
                with open("exceptions/parse_error.txt", "a") as f:
                    f.write(f"{model}\n")
                continue
            if len(parsed) == 0:
                with open("exceptions/parsed_no_records.txt", "a") as f:
                    f.write(f"{model}\n")
                return []
            records.append(parsed)    
        return records
    
    print(f"{model} tables empty")
    with open("exceptions/table_empty.txt", "a") as f:
        f.write(f"{model}\n")
        
    # 没有table元素，则检查有无ul元素

    return None

    