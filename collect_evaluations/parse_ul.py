import requests
from bs4 import BeautifulSoup


def parse_ul(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "xml")
    ul_list = soup.find_all("ul")
    for ul in ul_list:
        h1_tag = ul.find_previous("h1")
        if h1_tag is not None:
            return ul
    return None


if __name__ == "__main__":
    model = "Ayush414/distilbert-base-uncased-finetuned-ner"
    url = "https://huggingface.co/" + model
    ul = parse_ul(url)
    print(ul)