import os
import requests
from lxml import etree
from fuzzywuzzy import process


def get_main_content(model_card):
    print("get_main_content")
    try:
        parsed_html = etree.HTML(model_card)
        main_content = parsed_html.xpath('/html/body/div/main/div/section[1]/div[3]/div[2]')[0]
        main_content = list(main_content.itertext())
        main_content = [t.strip() for t in main_content]
        main_content = ' '.join(main_content)
    except:
        main_content = None
    finally:
        print("end of get_main_content")
        return main_content

def model_card_match(model_card, ds_set):
    print("model_card_match")
    main_content = get_main_content(model_card)
    if main_content is None:
        return set()
    try:
        print("start fuzzywuzzy")
        matched_ds = process.extract(main_content, ds_set)
        print("end fuzzywuzzy")
    except:
        matched_ds = set()
    finally:
        print("end of model_card_match")
        if matched_ds:
            matched_ds = set([item[0] for item in matched_ds if item[1] > 50])
        return matched_ds

def model_to_datasets(model_name, ds_set):
    print("model_to_datasets")
    try:
        model_url = f"https://huggingface.co/{model_name}"
        model_card = requests.get(model_url).text
    except:
        return set()
    if model_card:
        results = model_card_match(model_card, ds_set)
    else:
        results = set()
    print("end of model_to_datasets")
    return results

def formal_to_index(formal_set, formal2index):
    return set([formal2index[formal] for formal in formal_set])

if __name__ == "__main__":
    with open("model_name_set_all") as f:
        model_name_set = eval(f.read())
    print("#all model:", len(model_name_set))

    # ds_set = set(os.listdir("/data/data0/v-junliang/DNNGen/auto_model_dev/dataset_desc"))
    ds_set = set()
    formal2index = {}
    for file in os.listdir("/data/data0/v-junliang/DNNGen/auto_model_dev/dataset_desc"):
        with open(os.path.join("/data/data0/v-junliang/DNNGen/auto_model_dev/dataset_desc", file)) as f:
            formal_name = eval(f.read())["name"]
            ds_set.add(formal_name)
            formal2index[formal_name] = file
    print("#dataset:", len(ds_set))

    save_root = "/data/data0/v-junliang/DNNGen/auto_model_dev/mapping/huggingface/model_to_datasets"
    collected = 0
    total = 0
    if os.path.exists("tried_models"):
        with open("tried_models") as f:
            tried_models = set(eval(f.read()))
        model_name_set = model_name_set - tried_models
    else:
        tried_models = set()
    collected_models = set(os.listdir(save_root))
    collected_models = set([model_name.replace("--", "/") for model_name in collected_models])
    model_name_set = model_name_set - collected_models
    print("#model left:", len(model_name_set))

    for model_name in model_name_set:
        total += 1
        tried_models.add(model_name)
        if total % 100 == 0:
            print(f"collected {collected} / {total} models, all {len(model_name_set)} models")

        if os.path.exists(os.path.join(save_root, model_name.replace('/', "--"))):
            collected += 1
            tried_models.add(model_name)
            continue

        print("collecting model:", model_name)
        try:
            matched_ds = model_to_datasets(model_name, ds_set)
        except:
            continue
        else:
            matched_ds = formal_to_index(matched_ds, formal2index)
            if len(matched_ds) == 0:
                continue
            with open(os.path.join(save_root, model_name.replace('/', "--")), "w") as f:
                print(matched_ds, file=f)
            collected += 1


        if total % 10 == 0:
            print("writing tried_models")
            with open("tried_models", "w") as f:
                print(tried_models, file=f)

    print(f"finally collected {collected} / {len(model_name_set)} models")
