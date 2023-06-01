import pandas

from parse_utils import *




# url = "https://huggingface.co/gpt2"
# url = "https://huggingface.co/openai-gpt"
# url = "https://huggingface.co/bert-base-uncased"
# url = "https://huggingface.co/bert-large-cased"
# url = "https://huggingface.co/Davlan/afro-xlmr-base"
# url = "https://huggingface.co/microsoft/deberta-v3-xsmall"
# url = "https://huggingface.co/rufimelo/Legal-BERTimbau-large-TSDAE-v4-GPL-sts"
# url = "https://huggingface.co/xlm-mlm-17-1280"
# url = "https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor"
# url = "https://huggingface.co/sentence-transformers/clip-ViT-B-32"
# url = "https://huggingface.co/vesteinn/XLMR-ENIS-finetuned-stsb"
# model = "dangvantuan/sentence-camembert-large"
# model = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
# model = "Visual-Attention-Network/VAN-Large-original"
# model = "SamMorgan/yolo_v4_tflite"
# model = "apple/mobilevit-small"
model = "lirondos/anglicisms-spanish-flair-cs"
url = "https://huggingface.co/" + model


# url = "https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
# url = "https://huggingface.co/hackathon-pln-es/bertin-roberta-base-finetuning-esnli"
# url = "https://huggingface.co/ahmeddbahaa/AraBART-finetuned-ar"

# data = pandas.read_html(url, match="(?<![A-Za-z])([0-9]+\.[0-9]+)(?![A-Za-z])")
tables = get_tables(url)
print(f"#tables: {len(tables)}")
if len(tables) > 1:
    print(f"{model}: more than 1 tables")
    for data in tables:
        records = parse(data, model, default_datasets, default_metrics, default_dataset_metric)

elif len(tables) == 1:
    data = tables[0]

    # records = parse(data, "gpt2", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "openai-gpt", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "bert-base-uncased", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "bert-large-cased", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "Davlan/afro-xlmr-base", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "microsoft/deberta-v3-xsmall", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "rufimelo/Legal-BERTimbau-large-TSDAE-v4-GPL-sts", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "xlm-mlm-17-1280", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "sentence-transformers/msmarco-bert-co-condensor", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "sentence-transformers/clip-ViT-B-32", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "vesteinn/XLMR-ENIS-finetuned-stsb", default_datasets, default_metrics, default_dataset_metric)
    records = parse(data, model, default_datasets, default_metrics, default_dataset_metric)


    # records = parse(data, "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "hackathon-pln-es/bertin-roberta-base-finetuning-esnli", default_datasets, default_metrics, default_dataset_metric)
    # records = parse(data, "ahmeddbahaa/AraBART-finetuned-ar", default_datasets, default_metrics, default_dataset_metric)
else:
    print("no tables")