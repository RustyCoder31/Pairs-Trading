from statsmodels.tsa.stattools import coint
from config import download_data
import pandas as pd


def cointegration_check(asset_x, asset_y):
    
    score, p_value, crit_values = coint(asset_x,asset_y)

    # print(f"t-statistic (score): {score:.4f}")
    # print(f"p-value:             {p_value:.4f}")
    # print(f"Critical Values:     1%={crit_values[0]:.2f}, 5%={crit_values[1]:.2f}, 10%={crit_values[2]:.2f}")
    

    # if p_value < 0.05:
    #     print("\n[SUCCESS] The pairs are COINTEGRATED! Reject the null hypothesis.")
    #     print("You can proceed with building a pairs trading strategy.")
    # else:
    #     print("\n[WARNING] The pairs are NOT cointegrated. High risk of a diverging spread.")
    #     print("You might need a different time window or a different asset pair.")

    return p_value    


def cointegrated_pairs_check(correlated_pairs, pairs_data):

    results = []

    for i in range(len(correlated_pairs)):

        stock_1 = correlated_pairs["Stock 1"].iloc[i]
        stock_2 = correlated_pairs["Stock 2"].iloc[i]


        x = pairs_data[stock_1]
        y = pairs_data[stock_2]
        

        common_dates = x.dropna().index.intersection(y.dropna().index)
        

        if len(common_dates) < 30:
            continue
            

        x_aligned = x.loc[common_dates]
        y_aligned = y.loc[common_dates]
        

        p_val = cointegration_check(x_aligned, y_aligned)

        if p_val < 0.05:
            results.append({
                "Stock 1": stock_1,
                "Stock 2": stock_2,
                "P value": p_val
            })
            
    cointegration_df = pd.DataFrame(results)
    return cointegration_df