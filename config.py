import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*inplace.*")

def download_data(asset_1, asset_2, start_date, end_date):

    pair_x = yf.download(asset_1, start = start_date, end=end_date)["Close"]
    pair_y = yf.download(asset_2, start = start_date, end=end_date)["Close"]

    df = pd.concat([pair_x, pair_y], axis=1).dropna()
    pair_x, pair_y = df.iloc[:, 0], df.iloc[:, 1]

    return pair_x, pair_y

def split_data(asset_1, asset_2, split_percent):

    split_index = int(len(asset_1)*split_percent)

    train_x = asset_1.iloc[:split_index].squeeze()
    test_x = asset_1.iloc[split_index:].squeeze()

    train_y= asset_2.iloc[:split_index].squeeze()
    test_y = asset_2.iloc[split_index:].squeeze()

    return train_x, test_x, train_y, test_y








if __name__ == "__main__":
    download_data()