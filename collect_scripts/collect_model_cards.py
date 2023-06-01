import os
import requests
import wget
from lxml import etree
from lxml.html import tostring


def get_model_card(url):
    html = requests.get(url).text
    parsed_html = etree.HTML(html)
    model_card = parsed_html.xpath("/html/body/div/main/div/section[1]/div[3]/div[2]")[0]
    model_card = tostring(model_card, encoding="utf-8").decode("utf-8")
    return model_card

if __name__ == "__main__":
    model_name_set_path = "/home/v-junliang/DNNGen/auto_model_dev/mapping/huggingface/model_names"
    model_name_set = set()
    for file in os.listdir(model_name_set_path):
        with open(os.path.join(model_name_set_path, file)) as f:
            model_name_set = model_name_set.union(set(eval(f.readline())))
    print("#all model:", len(model_name_set))

    save_dir = "/data/data0/v-junliang/DNNGen/auto_model_dev/model_cards"
    collected = 0
    for model_name in model_name_set:
        collected += 1
        url = f"https://huggingface.co/{model_name}/resolve/main/README.md"
        save_path = os.path.join(save_dir, model_name.replace("/", "--"))
        if not os.path.exists(save_path):
            try:
                wget.download(url, save_path)
            except:
                continue

        if collected % 100 == 0:
            print(f"collected {collected} / {len(model_name_set)} model cards")

    print("task done")
