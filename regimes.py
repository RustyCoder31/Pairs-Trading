from hmmlearn.hmm import GaussianHMM
import numpy as np
import matplotlib.pyplot as plt


def hmm_regimes(df_kalman, split , hmm_test):

    df_kalman_clean = df_kalman.dropna().copy() 

    if split == 0:
        std_spread = np.array(df_kalman_clean['Spread']).std()
        df_kalman_clean['Spread'] = df_kalman_clean['Spread'].clip(lower=-3*std_spread, upper=3*std_spread)
    
    z_scores = df_kalman_clean[['Spread']].values
    z_scores = np.abs(z_scores)

    if split == 0:
        hmm = GaussianHMM(n_components=2, covariance_type= 'full', n_iter= 100, random_state= 42)
        hmm.fit(z_scores)
        regimes = hmm.predict(z_scores)
        df_kalman_clean['Regimes'] = regimes

    else:
        regimes = hmm_test.predict(z_scores)
        df_kalman_clean['Regimes'] = regimes
        hmm = hmm_test

    # for i in range(hmm.n_components):
    #     state_mean = hmm.means_[i]
    #     state_variance = np.diag(hmm.covars_[i])

    # plt.figure(figsize=(12, 6))
    # plt.plot(df_kalman_clean.index, regimes, label='Hand-coded Viterbi Path', color='blue')
    # plt.title('Viterbi Decoded Regime Timeline')
    # plt.xlabel('Date Index')
    # plt.ylabel('Regime (0=Calm, 1=Volatile)')
    # plt.legend()
    # plt.show()
    

    return df_kalman_clean, hmm


if __name__ == "__main__":
    hmm_regimes()   
 