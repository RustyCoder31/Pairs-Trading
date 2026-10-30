import pandas as pd
import yfinance as yf
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*inplace.*")

def find_pairs(url):

    url = url
    storage_options = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    df_table = pd.read_html(url, storage_options=storage_options)[0]

    df_table["Symbol"] = df_table["Symbol"].str.replace(".", "-", regex=False)



    stocks_df = df_table[["Symbol", "GICS Sector", "GICS Sub-Industry"]].copy()
    stocks_df.columns = ["Ticker", "Sector", "Sub-Industry"]
    stocks_df = stocks_df.dropna(subset=["Ticker"])
    stocks_df["Ticker"] = stocks_df["Ticker"].astype(str).str.strip()

    ISEND = "2024-01-01"

    prices = yf.download(stocks_df["Ticker"].to_list(), start="2020-01-01", end=ISEND)["Close"]
    prices = prices.dropna(axis=1, how="all")

    sub_industry_dict = dict(zip(stocks_df["Ticker"], stocks_df["Sub-Industry"]))

    all_possible_pairs = list(combinations(stocks_df["Ticker"], 2))

    filtered_pairs = []
    for stock_a, stock_b in all_possible_pairs:

        if sub_industry_dict[stock_a] == sub_industry_dict[stock_b]:
            filtered_pairs.append((stock_a, stock_b))


    print("Size of filtered Pairs", len(filtered_pairs))
    print("Size of unfiltered Pairs", len(all_possible_pairs))

    

    rets = prices.pct_change()

    correlated_pairs = []

    for stock_a, stock_b in filtered_pairs:

        if stock_a in rets.columns and stock_b in rets.columns:
            correlation = rets[stock_a].corr(rets[stock_b])
        
            if correlation > 0.5:
                correlated_pairs.append({
                    "Stock 1": stock_a,
                    "Stock 2": stock_b,
                    "Correlation": correlation
                })

    pairs_df = pd.DataFrame(correlated_pairs)

    print("Size of final high-correlation pairs:", len(pairs_df))

    final_tickers = list(set(pairs_df["Stock 1"].to_list() + pairs_df["Stock 2"].to_list()))
    final_prices = yf.download(
    tickers = final_tickers, 
    start="2020-01-01", 
    end=ISEND
    )["Close"]


    final_prices = final_prices.dropna(how="all", axis=1)

    final_prices.to_csv("correlated_prices.csv")

    return pairs_df, final_prices



if __name__ == "__main__":
    find_pairs("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")