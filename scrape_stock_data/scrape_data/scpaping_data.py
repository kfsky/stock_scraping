"""
scraping stock data in stock data.
and create df.
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import numpy as np

"""
robots.txt allows the scraping.
"""


def get_stock_data(stock_number):
    dfs = []
    years = [2018, 2019]

    for year in years:
        try:
            print(year)
            # get stock_data
            url = "https://kabuoji3.com/stock/{}/{}/".format(stock_number, year)
            # Header Agent
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36"
            }
            soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
            tag_tr = soup.find_all("tr")
            column_data = [h.text for h in tag_tr[0].find_all('th')]

            # create df and add the data
            data = []
            for i in range(1, len(tag_tr)):
                data.append([d.text for d in tag_tr[i].find_all("td")])
            df = pd.DataFrame(data, columns=column_data)

            # type float
            col = ['始値', '高値', '安値', '終値', '出来高', '終値調整']
            for c in col:
                df[c] = df[c].astype(float)
            df["日付"] = [datetime.strptime(i,'%Y-%m-%d') for i in df['日付']]
            dfs.append(df)
        except IndexError:
            print("no data")
    return dfs


def concatenate(dfs):
    data = pd.concat(dfs, axis=0)
    data = data.reset_index(drop=True)
    col = ['始値', '高値', '安値', '終値', '出来高', '終値調整']
    for c in col:
        data[c] = data[c].astype(float)
    return data


"""
create the stock_list

list.csv needs stock_number, stock_name

ex.
code,name
2424,ブラス
...
 
"""

get_list = pd.read_csv("list.csv", sep=",")

for i in range(len(get_list)):
    k = get_list.loc[i, 'code']
    v = get_list.loc[i, 'name']
    print(k, v)
    dfs = get_stock_data(k)
    data = concatenate(dfs)
    data.to_csv('./data/{}-{}.csv'.format(k, v))

print("end")
