import os
import json
import requests
from lxml import etree

# gen dataset_desc folder, which contains dataset details like: {"name": standard_name, "content": content}
def dataset_num_of_mod(mod):
    """
    Returns:
        int: number of datasets of the given modality
    """
    mod_url = "https://paperswithcode.com/datasets?mod=" + mod
    mod_page = requests.get(mod_url).text
    parsed_mod_page = etree.HTML(mod_page)
    h1_list = parsed_mod_page.xpath('//h1')
    n_ds = ""
    for h1 in h1_list:
        text = h1.xpath('string()')
        if "dataset results" in text:
            n_ds = text.split(" ")[0]
            break
    if len(n_ds) > 0:
        n_ds = eval(n_ds)
    else:
        n_ds = 0
    return n_ds

def dataset_num_all(root_url):
    page = requests.get(root_url).text
    parsed_page = etree.HTML(page)
    h1_list = parsed_page.xpath('//h1')
    num_ds = ""
    for h1 in h1_list:
        text = h1.xpath('string()')
        if "dataset results" in text:
            num_ds = text.split(" ")[0]
            break
    if len(num_ds) > 0:
        num_ds = eval(num_ds)
    else:
        num_ds = 0
    return num_ds

def dataset_suffixes_of_page(page_url):
    """
    Returns:
        list[str]: dataset suffixes
    """
    page = requests.get(page_url).text
    parsed_page = etree.HTML(page)
    element = parsed_page.xpath('/html/body/div[2]/div[1]/div[2]/div[2]/div')[0]
    hrefs = element.xpath('//div/a/@href')
    ds_suffixes = []
    for href in hrefs:
        if href.startswith('/dataset'):
            ds_suffixes.append(href)
    return ds_suffixes

def dataset_desc(dataset_url):
    """
    Returns:
        str: dataset description
    """
    html = requests.get(dataset_url).text
    parsed_html = etree.HTML(html)
    try:
        desc_content = parsed_html.xpath("/html/body/div[11]/div/div[4]/div[1]/div[1]/div[1]/p")[0]
        standard_name = parsed_html.xpath("/html/body/div[11]/div/div[2]/div/div/h1")[0].text.strip()
    except IndexError:
        return None
    desc_content = list(desc_content.itertext())
    desc_content = [t.replace('\n', ' ') for t in desc_content]
    desc_content = ''.join(desc_content)
    return standard_name, desc_content

def dataset_desc_of_mod(mod):
    """parse the given modality page to get and save dataset descriptions
    """
    save_root = "/mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/dataset_desc"
    mod_dir = os.path.join(save_root, mod)
    if not os.path.exists(mod_dir):
        os.mkdir(mod_dir)

    n_ds = dataset_num_of_mod(mod)
    mod_url = "https://paperswithcode.com/datasets?mod=" + mod
    print(f"modality {mod} has {n_ds} datasets")
    if n_ds % 48 > 0:
        n_page = n_ds // 48 + 1
    else:
        n_page = n_ds / 48
    count = 0
    for pno in range(n_page):
        print(f"collecting page {pno+1}")
        page_url = mod_url + f"&page={pno+1}"
        dataset_suffixes = dataset_suffixes_of_page(page_url)
        for suffix in dataset_suffixes:
            dataset = suffix.replace("/dataset/", "")
            
            if os.path.exists(os.path.join(mod_dir, dataset)):
                count += 1
                continue

            dataset_url = "https://paperswithcode.com" + suffix
            ret = dataset_desc(dataset_url)
            if ret is None:
                continue
            else:
                standard_name, content = ret
            if len(content) == 0:
                continue
            ds_desc = {"name": standard_name, "content": content}
            with open(os.path.join(mod_dir, dataset), 'w') as f:
                json.dump(ds_desc, f)
            count += 1
        print(f"collected {count} datasets")

def dataset_desc_all():
    root_url = "https://paperswithcode.com/datasets"
    save_root = "/mnt/msrasrg/yileiyang/DNNGen/auto_model_dev/dataset_desc"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    n_ds = dataset_num_all(root_url)
    print(f"have {n_ds} datasets to collect")


    if n_ds % 48 > 0:
        n_page = n_ds // 48 + 1
    else:
        n_page = n_ds / 48

    count = 0
    for pno in range(n_page):
        print(f"collecting page {pno+1}")
        page_url = root_url + f"?page={pno+1}"
        dataset_suffixes = dataset_suffixes_of_page(page_url)
        for suffix in dataset_suffixes:
            dataset = suffix.replace("/dataset/", "")
            
            if os.path.exists(os.path.join(save_root, dataset)):
                count += 1
                print(f"{dataset} collected, skip")
                continue

            dataset_url = "https://paperswithcode.com" + suffix
            ret = dataset_desc(dataset_url)
            if ret is None:
                continue
            else:
                standard_name, content = ret
            if len(content) == 0:
                continue
            ds_desc = {"name": standard_name, "content": content}
            with open(os.path.join(save_root, dataset), 'w') as f:
                json.dump(ds_desc, f)
            count += 1
        print(f"collected {count} / {n_ds} datasets")


if __name__ == "__main__":
    print(dataset_desc("https://paperswithcode.com/dataset/coco"))
    dataset_desc_all()
    print("all done")