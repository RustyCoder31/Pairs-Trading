from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_kalman_spread(asset_x, asset_y):

    obs_mat = np.vstack([asset_x, np.ones(len(asset_x))]).T[:, np.newaxis, :]

    delta = 1e-6 
    trans_cov = delta / (1 - delta) * np.eye(2)

    obs_cov = 1.0 

    kf = KalmanFilter(
            n_dim_obs=1, # Number of dimensions of the target real-world observation (Visa's price)
            n_dim_state=2, # No of hidden states filter needs to guess (beta and intercept)
            initial_state_mean=np.zeros(2), # Initial guesses for beta and intercept
            initial_state_covariance=np.eye(2), # how uncertain we are about our initial guesses (P) --> setting it one tells we are highly uncertain 
            transition_matrices= np.eye(2), # how the state transitions from yesterday to today --> We assume yesterday's beta and intercept for today
            transition_covariance= trans_cov, # The variance you set here defines how much day-to-day drift is allowed to be structural (true parameter shift). Any error smaller or different from this that can be explained by your observation_covariance (noise R)
            observation_matrices=obs_mat, # This is our observation matrix. How to multiply states day to day
            observation_covariance=obs_cov # This is our noise R, Donot change hedge ratio for this when compairing
    )

    state_means, _ = kf.filter(asset_y.values)

    hedge_ratios = pd.Series(state_means[:, 0], index = asset_y.index).shift(1)
    intercepts = pd.Series(state_means[:, 1], index = asset_y.index).shift(1)

# spread = Today's visa close - yesterday's hedge ratio* today's master price + yesterday's intercept
    spread = asset_y - (hedge_ratios * asset_x + intercepts)
    predicted = hedge_ratios * asset_x + intercepts


    rolling_mean = spread.rolling(window=20).mean()
    rolling_std = spread.rolling(window=20).std()
    z_scores = (spread - rolling_mean) / rolling_std


    kalman_df = pd.DataFrame({
    'Asset_Y_Close': asset_y,
    'Asset_X_Close': asset_x,
    'Hedge_Ratio': hedge_ratios,
    'Predicted' : predicted,
    'Spread': spread,
    'Z_Score': z_scores
    }, index=asset_y.index)

    # plt.figure(figsize=(12, 6))
    # plt.title("Static Spread with ±2σ Bands", fontsize=12)
    # plt.plot(z_scores.index, z_scores, color='blue', linewidth=1, label='Spread')
    # plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label='Mean (0)')
    # plt.axhline(2 * z_scores.std(), color='red', linestyle='--', linewidth=1.5, label='+2 Sigma')
    # plt.axhline(-2 * z_scores.std(), color='lime', linestyle='--', linewidth=1.5, label='-2 Sigma')
    # plt.ylabel("Spread Deviation")
    # plt.legend(loc='upper left')
    # plt.grid(alpha=0.2)

    return kalman_df




if __name__ == "__main__":
    calculate_kalman_spread()   