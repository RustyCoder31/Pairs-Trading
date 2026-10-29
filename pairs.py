import pandas as pd
import yfinance as yf
import numpy as np
from itertools import combinations

def find_pairs(url):

    url = url
    storage_options = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    tickers = pd.read_html(url, storage_options=storage_options)[0]["Symbol"].to_list()

    df = yf.download(tickers, start= '2020-01-01')['Close']

    combinations_df = pd.DataFrame(combinations(df.columns,2))
    combinations_df.columns = ["Stock 1", "Stock 2"]
    combinations_df['Correlation'] = combinations_df.apply(lambda row: np.corrcoef(df[row["Stock 1"]],df[row["Stock 2"]] )[0,1] , axis = 1)
    combinations_df = combinations_df[combinations_df.Correlation>0.95]

    return combinations_df, df