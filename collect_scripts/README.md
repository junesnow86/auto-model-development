Each file's function can be inferred from their file names. Notice that there are many local paths in the scripts. You may need to change them to your own paths.

1. run collect_dataset_desc.py to get dataset description from 'https://paperswithcode.com', gen dataset_desc folder, not every dataset has a description
2. run collect_dataset_to_task.py to get supported task list of each dataset from 'https://paperswithcode.com/datasets', to gen ds2task folder
3. run collect_task_set.py to summarize all the datasets and tasks from folders above, and gen 'task_set' which contains all the tasks
4. run collect_task_desc.py to get all the task description from 'https://paperswithcode.com/task/', which will use 'task_set' generated above
5. run collect_model_cards.py to get all the model cards, which is the model's front page in huggingface. Using mapping/model_name_set_all, don't know where it from