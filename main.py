
from config import split_data, download_data
from cointegration import cointegration_check, cointegrated_pairs_check
from kalman_filter import calculate_kalman_spread
from regimes import hmm_regimes
from strategy import strategy_test
from plots import (plot_prices, plot_spread_bands, plot_regime_timeline, plot_hmm_states,
                   plot_trades_in_sample, replay_out_of_sample, plot_trades_out_of_sample, plot_fixed_vs_kalman)
import pandas as pd
import matplotlib.pyplot as plt
from pairs import find_pairs


def regime_test(hmm):
    var_1 = hmm.covars_[0].flatten() 
    var_2 = hmm.covars_[1].flatten() 

    if var_1 < var_2:
        print ("The calm regime is 0 and volatile regime is 1\n")
        calm = 0

    else:
        print ("The calm regime is 1 and volatile regime is 0\n")   
        calm = 1

    return calm

correlated_pairs, pairs_data = find_pairs("https://en.wikipedia.org/wiki/List_of_Russell_1000_companies")

cointegrated_pairs = cointegrated_pairs_check(correlated_pairs, pairs_data)

cointegrated_pairs.to_csv("cointegrated_pairs.csv", index=False)


survivors = cointegrated_pairs[cointegrated_pairs["BH pass"]]
survivors.to_csv("cointegrated_pairs_BH.csv", index=False)
print("Pairs tested:", len(cointegrated_pairs), "Survivors:", len(survivors))

plt.hist(cointegrated_pairs["P value"], bins=20)
plt.xlabel("p-value"); plt.ylabel("number of pairs")
plt.savefig("Images/pvalue_hist.png")


# asset_1 = 'LEN'
# asset_2 = 'NVR'

# x, y = download_data(asset_1, asset_2, "2020-01-01", "2026-10-10")
# x_train, x_test, y_train, y_test = split_data(x, y, 0.7)

# split = 0
# p_val = cointegration_check(x_train, y_train)


# kalman_data = calculate_kalman_spread(x_train, y_train)

# regime_data, hmm = hmm_regimes(kalman_data, split, 0)

# print(regime_data.head(10))

# print("\n--- HMMLEARN BUILT-IN RESULTS ---")
# print("Transition Matrix:\n", hmm.transmat_)
# print("Means:\n", hmm.means_.flatten())
# print("Variances:\n", hmm.covars_.flatten())

# calm_train = regime_test(hmm)

# # %% Strategy (in-sample, slow)
# pnl_df = strategy_test(x_train, y_train, hmm, calm_train)


# # %% Plot: prices of the pair and the Kalman hedge ratio
# plot_prices(x_train, x_test, y_train, y_test, asset_1, asset_2)

# # %% Plot: Kalman spread with 2 sigma bands, and the rolling z-score (train)
# plot_spread_bands(kalman_data, regime_data)

# # %% Plot: HMM regime timeline (train)
# plot_regime_timeline(regime_data, calm_train)

# # %% Plot: |spread| by HMM state (train)
# plot_hmm_states(regime_data, hmm, calm_train)

# # %% Plot: in-sample trades (needs pnl_df from the strategy cell above)
# plot_trades_in_sample(regime_data, pnl_df, calm_train, asset_2)

# # %% Out-of-sample replay (slow, about a minute)
# oos_days, oos_trades = replay_out_of_sample(x_train, x_test, y_train, y_test, hmm, calm_train)

# # %% Plot: out-of-sample trades
# plot_trades_out_of_sample(oos_days, oos_trades, calm_train, asset_2)

# # %% Plot: Kalman spread vs a fixed train hedge on the test window
# plot_fixed_vs_kalman(x_train, x_test, y_train, y_test)



