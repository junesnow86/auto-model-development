import os
import json
import requests
from lxml import etree

# gen task_desc folder which contains {"name": standard_name, "content": desc_content} from paperswithcode

def get_task_desc(task_url):
    html = requests.get(task_url).text
    parsed_html = etree.HTML(html)
    try:
        standard_name = parsed_html.xpath('//*[@id="task-home"]')[0].text.strip()
        desc_content = parsed_html.xpath('/html/body/div[5]/main/div/div[3]/div[1]/div/p[1]')[0]
    except IndexError:
        return None
    except AttributeError:
        return None
    desc_content = list(desc_content.itertext())
    desc_content = [t.replace('\n', ' ') for t in desc_content]
    desc_content = ''.join(desc_content)
    if len(desc_content) == 0:
        return None
    return {"name": standard_name, "content": desc_content}

if __name__ == "__main__":
    print(get_task_desc("https://paperswithcode.com/task/object-detection"))
    url_root = "https://paperswithcode.com/task/"
    save_root = "/mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/task_desc"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    with open(os.path.join(current_folder, "task_set_new"), 'r') as f:
        task_set = eval(f.read())
    collected = 0
    none_count = 0
    for task in task_set:
        print(f"parsing {task}")
        if os.path.exists(os.path.join(save_root, task)):
            collected += 1
            if collected % 50 == 0:
                print(f"collected {collected} / {len(task_set)} tasks")
            continue
        task_url = url_root + task
        task_desc = get_task_desc(task_url)
        if task_desc is None:
            none_count += 1
            # print(f"task {task} has no description")
            continue
        with open(os.path.join(save_root, task), 'w') as f:
            json.dump(task_desc, f)
        collected += 1
        if collected % 50 == 0:
            print(f"collected {collected} / {len(task_set)} tasks")

    print(f"collected {collected} / {len(task_set)} tasks")
    print(f"none count: {none_count}")
