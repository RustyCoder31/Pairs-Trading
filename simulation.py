import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests

# Positive control for the pairs screen.
# Plant pairs that are cointegrated by construction, hide them among pure noise pairs,
# run the same test (both directions, take the larger p-value) and BH, and count how many are found.

N_DAYS = 1006        # same number of bars as the real run (2020 to 2023)
N_PLANTED = 50       # pairs with a real leash
N_NOISE = 2168       # pairs with no leash, so planted + noise = 2218 tests like the real run
PHIS = [0.80, 0.90, 0.95]   # how fast the gap is pulled back, higher means a looser leash
SEED = 0

rng = np.random.default_rng(SEED)


def random_walk():
    return np.cumsum(rng.normal(0, 1, N_DAYS)) + 100


def make_planted_pair(phi, beta=1.0):
    x = random_walk()
    gap = np.zeros(N_DAYS)
    for t in range(1, N_DAYS):
        gap[t] = phi * gap[t - 1] + rng.normal(0, 1)
    y = beta * x + gap
    return x, y


def make_noise_pair():
    return random_walk(), random_walk()


def pair_p_value(x, y):
    p_xy = coint(x, y)[1]
    p_yx = coint(y, x)[1]
    return max(p_xy, p_yx)


def half_life(phi):
    return -np.log(2) / np.log(phi)


noise_p = np.array([pair_p_value(*make_noise_pair()) for _ in range(N_NOISE)])
print("Noise pairs:", N_NOISE, "| raw p < 0.05:", int((noise_p < 0.05).sum()),
      "| expected by chance:", round(0.05 * N_NOISE, 1))
print()

for phi in PHIS:
    planted_p = np.array([pair_p_value(*make_planted_pair(phi)) for _ in range(N_PLANTED)])

    all_p = np.concatenate([planted_p, noise_p])
    keep = multipletests(all_p, alpha=0.05, method="fdr_bh")[0]

    found = int(keep[:N_PLANTED].sum())
    false_alarms = int(keep[N_PLANTED:].sum())

    print(f"phi = {phi:.2f} (gap half-life about {half_life(phi):.1f} days)")
    print(f"  planted pairs, raw p < 0.05:  {int((planted_p < 0.05).sum())} of {N_PLANTED}")
    print(f"  planted pairs, survive BH:    {found} of {N_PLANTED}")
    print(f"  noise pairs, survive BH:      {false_alarms} of {N_NOISE}")
    print()
