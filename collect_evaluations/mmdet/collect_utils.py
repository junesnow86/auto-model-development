import re
import requests
import bs4
from bs4 import BeautifulSoup
import pandas

def prpcess_table(table: bs4.element.Tag):
    pass

def get_tables(config_type: str):
    url = 'https://github.com/open-mmlab/mmdetection/tree/master/configs/' + config_type
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    tables = soup.find_all('table')

    valid_tables = []
    for table in tables:
        print(type(table))
        previous_htag = table.find_previous(re.compile(r'h\d'))
        if previous_htag is not None:
            if 'Results and Models' in previous_htag.text:
                df = pandas.read_html(str(table))[0]
                for i, row in df.iterrows():
                    for j, cell in row.items():
                        print(cell)
                        link = BeautifulSoup(cell, 'lxml').find('a')
                        if link:
                            df.at[i, j] = link['href']
                valid_tables.append(df)
    return valid_tables


if __name__ == '__main__':
    config_type = 'rpn'
    print(get_tables(config_type))

