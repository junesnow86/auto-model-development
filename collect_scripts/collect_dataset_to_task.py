import os
import requests
from lxml import etree

# gen ds2task folder, the file name is a dataset, the content is the task list of this dataset

def get_dataset_tasks(url: str) -> tuple[str]:
    html_string = requests.get(url).text
    parsed_html = etree.HTML(html_string)
    elements = parsed_html.xpath('//a/@href')
    tasks = []
    for element in elements:
        if element.startswith('/task'):
            tasks.append(element.replace('/task/', ''))
    return tuple(tasks)

def parse_page(page_url: str) -> list[str]:
    page = requests.get(page_url).text
    parsed_html = etree.HTML(page)
    element = parsed_html.xpath('/html/body/div[2]/div[1]/div[2]/div[2]/div')[0]
    hrefs = element.xpath('//div/a/@href')
    dataset_suffixes = []
    for href in hrefs:
        if href.startswith('/dataset'):
            dataset_suffixes.append(href)
    return dataset_suffixes

def get_dataset_num(mod_url) -> int:
    mod_page = requests.get(mod_url).text
    parsed_mod_page = etree.HTML(mod_page)
    h1_list = parsed_mod_page.xpath('//h1')
    n_dataset = ''
    for h1 in h1_list:
        text = h1.xpath('string()')
        if 'dataset results' in text:
            n_dataset = text.split(' ')[0]
            break
    if len(n_dataset) > 0:
        n_dataset = eval(n_dataset)
    else:
        n_dataset = 0
    return n_dataset

def dataset_num_all(root_url) -> int:
    page = requests.get(root_url).text
    parsed_page = etree.HTML(page)
    h1_list = parsed_page.xpath('//h1')
    n_dataset = ''
    for h1 in h1_list:
        text = h1.xpath('string()')
        if 'dataset results' in text:
            n_dataset = text.split(' ')[0]
            break
    if len(n_dataset) > 0:
        n_dataset = eval(n_dataset)
    else:
        n_dataset = 0
    return n_dataset

def parse_mod(root, mod):
    save_root = '/mnt/msrasrg/yileiyang/DNNGen/collect_task_descriptions/dataset-tasks'
    mod_dir = os.path.join(save_root, mod)
    if not os.path.exists(mod_dir):
        os.mkdir(mod_dir)

    mod_url = root + '/datasets?mod=' + mod
    n_dataset = get_dataset_num(mod_url)
    print(f'modality: {mod} has {n_dataset} datasets')
    if n_dataset % 48 > 0:
        n_page = n_dataset // 48 + 1
    else:
        n_page = n_dataset / 48 
    count = 0
    for page_no in range(n_page):
        print(f'collecting page {page_no+1}')
        page_url = mod_url + f'&page={page_no+1}'
        dataset_suffixes = parse_page(page_url)
        for dataset_suffix in dataset_suffixes:
            dataset = dataset_suffix.replace('/dataset/', '')

            if os.path.exists(os.path.join(mod_dir, dataset)):
                count += 1
                continue

            url = root + dataset_suffix
            tasks = get_dataset_tasks(url)
            # save collections
            with open(os.path.join(mod_dir, dataset), 'w') as f:
                f.write(str(tasks))
            count += 1
        print(f'{count} datasets\' tasks collected')

def parse_all():
    save_root = '/mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/ds2task'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    root_url = 'https://paperswithcode.com/datasets'
    root = 'https://paperswithcode.com'
    n_dataset = dataset_num_all(root_url)
    print(f'has {n_dataset} datasets to collect')

    if n_dataset % 48 > 0:
        n_page = n_dataset // 48 + 1
    else:
        n_page = n_dataset / 48 

    count = 0
    for page_no in range(n_page):
        print(f'collecting page {page_no+1}')
        page_url = root_url + f'?page={page_no+1}'
        dataset_suffixes = parse_page(page_url)
        for dataset_suffix in dataset_suffixes:
            dataset = dataset_suffix.replace('/dataset/', '')

            if os.path.exists(os.path.join(save_root, dataset)):
                count += 1
                print(f'{dataset} already collected')
                continue

            url = root + dataset_suffix
            tasks = get_dataset_tasks(url)
            # save collections
            with open(os.path.join(save_root, dataset), 'w') as f:
                f.write(str(tasks))
            count += 1
        print(f'{count} datasets\' tasks collected')

if __name__ == '__main__':
    # url = 'https://paperswithcode.com/dataset/cifar-10'
    # tasks = get_dataset_tasks(url)
    # print(tasks)

    # root = 'https://paperswithcode.com/datasets'
    # mod = 'images'
    # page_url = root + '?mod=' + mod
    # parse_page(page_url)
    parse_all()
