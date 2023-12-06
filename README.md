# auto-model-development

## Introduction
This project mainly contains four components:
- collecting(done)
    - model: model cards and model names on [huggingface](https://huggingface.co/)
    - dataset: dataset descriptions from [paperswithcode](https://paperswithcode.com/datasets)
    - task: task descriptions from [paperswithcode](https://paperswithcode.com/)
- mapping(not done)
    - dataset to task(done)
    - model to dataset(not done)
    - build (model, dataset, task) tuples(not done)
- graph(not done)
    - produce models' [fx graphs](https://pytorch.org/docs/stable/fx.html) with concrete trace tools
    - produce xml graphs with traced fx graphs
- producing samples(not done)
    - a sample is a tuple of (model, dataset, task)
    - `model` is a xml graph of a model.
    - `dataset` is a dataset description. The dataset is used to train/evaluate the model.
    - `task` is a task description. The task is one of the tasks that the dataset is used on.

## Usage
More details can be found in README files under each folder. Most of codes need to be specified proper paths before running.
- collect_evaluations: early-stage python scripts used for parsing and collecting evaluation results of models on huggingface. They are probably not useful now.
- collect_scripts: python scripts used for collecting model cards, dataset descriptions, task descriptions and dataset-task mapping.
- mapping: codes used for collecting model to dataset mapping and build (model, dataset, task) tuples.
- graph: concrete_trace_utils for producing fx graphs and fxgraph_to_seq for producing xml graphs.
- generate_samples: uses the (model, dataset, task) tuples to concatenate xml graphs, dataset descriptions and task descriptions to produce samples.

## Storage
The prepared data is stored on MSRAGPUM19. Paths are listed below:
- collecting
    - model cards: /mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/model_cards, one file for each model, the file name is the model name, the file content is the markdown content of the model card.
    - model names: /home/yileiyang/workspace/DNNGen/auto_model_dev/graph/huggingface_model_names, each file is a model name set.
    - dataset descriptions: /mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/dataset_desc, one file for each dataset, the file name is the dataset index name(used in url) of paperswithcode, the file contains the dataset formal name and the description.
    - task descriptions: /mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/task_desc, one file for each task, the file name is the task index name of paperswithcode, the file contains the task formal name and the description.
- mapping
    - dataset to task: /mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/dataset_to_task, one file for each dataset, the file name is the dataset index name of paperswithcode, the file contains tasks that the dataset is used on.
    - huggingface
        - model to dataset: /mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/mapping/huggingface/model_to_datasets, one file for each model, the file contains datasets it the model used on(according to fuzzywuzzy, not sure for correctness).
        - triples: each file is under the folder structure {task}/{dataset}/{model}, the file content is a (model name, dataset index name, task index name) tuple.
- graph
    - it's better to use the latest tracer to produce new xml graphs that are wanted.
    - already produced xml graphs(each of the below folders contains only a part of the whole models, i.e., there are many models not traced):
        - 1.0: /mnt/msrasrg/yileiyang/DNNGen/graph2seq1.0, xml graphs are not folded and have arguments information for module node.
        - 2.0: /mnt/msrasrg/yileiyang/DNNGen/graph2seq2.0, not folded, and with shape propagation and dce(dce here may not be correct).
        - 3.0: /mnt/msrasrg/yileiyang/DNNGen/graph2seq3.0, folded to the first module level.
        - 4.0: internblob v-junliang xml, unfolded, without shape propagation, without dce.

## Unsolved
- mapping
    - huggingface
        - model to dataset: mapping/tried_models is the set of tried models, but it is not complete. And there is some error running the model_to_dataset.py script.
        - build (model, dataset, task) tuples: waiting for the above one.
    - mmdetection
        - model to dataset: not started.
        - build (model, dataset, task) tuples: not started.
- execute concrete trace for all models
- generate samples for both huggingface and mmdetection.
