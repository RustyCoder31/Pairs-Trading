import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import coint

from kalman_filter import calculate_kalman_spread
from regimes import hmm_regimes

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})


def _trade_markers(ax_price, ax_z, z, long_in, short_in, exits):
    for d in long_in:
        ax_z.scatter(d, z.loc[d], marker="^", color="green", s=150, zorder=5)
        ax_price.axvline(d, color="green", alpha=0.4, lw=2)
    for d in short_in:
        ax_z.scatter(d, z.loc[d], marker="v", color="red", s=150, zorder=5)
        ax_price.axvline(d, color="red", alpha=0.4, lw=2)
    for d in exits:
        ax_z.scatter(d, z.loc[d], marker="X", color="black", s=120, zorder=5)
        ax_price.axvline(d, color="black", ls=":", alpha=0.7, lw=2)
    ax_z.legend(handles=[
        Line2D([0], [0], color="green", marker="^", ls="None", markersize=10, label="Entry long"),
        Line2D([0], [0], color="red", marker="v", ls="None", markersize=10, label="Entry short"),
        Line2D([0], [0], color="black", marker="X", ls="None", markersize=10, label="Exit")], loc="lower left")


def _price_and_z_panels(index, y, predicted, z, regimes, vol, y_name, title):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    a1.plot(index, y, label=f"{y_name} actual", color="blue", alpha=0.7)
    a1.plot(index, predicted, label=f"{y_name} Kalman prediction", color="orange", alpha=0.7)
    a1.set_ylabel(f"{y_name} price")
    a1.legend(loc="upper left", fontsize=9)
    a1.set_title(title)
    a2.plot(index, z, color="purple", lw=1)
    a2.fill_between(index, 0, 1, where=(regimes == vol), transform=a2.get_xaxis_transform(), color="orange", alpha=0.15)
    a2.axhline(0, color="black", lw=1)
    a2.axhline(2, color="red", ls="--", alpha=0.5)
    a2.axhline(-2, color="green", ls="--", alpha=0.5)
    a2.set_title("Kalman spread z-score (orange shading = volatile state, no entries)")
    return fig, a1, a2


def plot_prices(x_train, x_test, y_train, y_test, x_name, y_name):
    """Both prices rebased to 100, with the Kalman hedge ratio underneath."""
    x_all = pd.concat([x_train, x_test])
    y_all = pd.concat([y_train, y_test])
    px = pd.concat([x_all, y_all], axis=1, keys=[x_name, y_name])
    kalman_all = calculate_kalman_spread(x_all, y_all)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    (px / px.iloc[0] * 100).plot(ax=a1)
    a1.axvline(x_test.index[0], color="black", ls="--", lw=1)
    a1.text(x_test.index[0], a1.get_ylim()[1], " test starts", va="top", fontsize=9)
    a1.set_title(f"{x_name} and {y_name}, both rebased to 100")
    a2.plot(kalman_all.index, kalman_all["Hedge_Ratio"], color="tab:green")
    a2.axvline(x_test.index[0], color="black", ls="--", lw=1)
    a2.set_title("Kalman hedge ratio (beta)")
    fig.tight_layout()
    plt.show()


def plot_spread_bands(kalman_data, regime_data):
    """Kalman spread with +/-2 sigma bands, and the rolling z-score (train)."""
    spread = kalman_data["Spread"].dropna()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    a1.plot(spread.index, spread, color="blue", lw=1, label="Kalman spread (unclipped)")
    a1.axhline(0, color="black", lw=1.2, label="mean (0)")
    a1.axhline(2 * spread.std(), color="red", ls="--", lw=1.2, label="+2 sigma")
    a1.axhline(-2 * spread.std(), color="lime", ls="--", lw=1.2, label="-2 sigma")
    a1.set_title("Kalman spread with 2 sigma bands (train)")
    a1.legend(loc="upper left", fontsize=9)
    a2.plot(regime_data.index, regime_data["Z_Score"], color="purple", lw=1)
    a2.axhline(0, color="black", lw=1)
    a2.axhline(2, color="red", ls="--", alpha=0.5)
    a2.axhline(-2, color="green", ls="--", alpha=0.5)
    a2.set_title("Rolling z-score of the spread (train)")
    fig.tight_layout()
    plt.show()


def plot_regime_timeline(regime_data, calm):
    """HMM regime timeline on train, calm vs volatile."""
    reg = regime_data["Regimes"]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.step(reg.index, (reg != calm).astype(int), where="post", color="blue", lw=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["calm", "volatile"])
    ax.set_title("HMM regime timeline (train)")
    fig.tight_layout()
    plt.show()


def plot_hmm_states(regime_data, hmm, calm):
    """What the HMM sees: |spread| by state, and the distribution per state (train)."""
    vol = 1 - calm
    a = regime_data["Spread"].abs()
    reg = regime_data["Regimes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for s, name, col in [(calm, "calm", "tab:blue"), (vol, "volatile", "tab:orange")]:
        m = reg == s
        var = float(hmm.covars_[s].ravel()[0])
        a1.scatter(a.index[m], a[m], s=9, color=col, label=f"{name} (state {s}, variance {var:.4f}, {int(m.sum())} days)")
        a2.hist(a[m], bins=30, alpha=0.65, color=col)
    a1.set_title("Train |spread| coloured by HMM state (clipped at 3 std before fitting)")
    a1.legend(fontsize=9)
    a2.set_title("Distribution per state")
    fig.tight_layout()
    plt.show()


def plot_trades_in_sample(regime_data, pnl_df, calm, y_name):
    """In-sample trades from strategy_test: price vs Kalman prediction, z-score with entries and exits."""
    vol = 1 - calm
    trades = pnl_df if pnl_df is not None and not pnl_df.empty else pd.DataFrame(columns=["Entry", "Exit", "Type"])
    longs = trades.loc[trades["Type"] == "Long Spread", "Entry"]
    shorts = trades.loc[trades["Type"] == "Short Spread", "Entry"]

    fig, a1, a2 = _price_and_z_panels(regime_data.index, regime_data["Asset_Y_Close"], regime_data["Predicted"],
                                      regime_data["Z_Score"], regime_data["Regimes"], vol, y_name,
                                      f"In-sample trades ({len(trades)}): entries and exits")
    _trade_markers(a1, a2, regime_data["Z_Score"], longs, shorts, trades["Exit"])
    fig.tight_layout()
    plt.show()


def replay_out_of_sample(x_train, x_test, y_train, y_test, hmm, calm, z_entry=2.0):
    """Walk-forward over the test window with the train HMM, same rules as strategy.py (slow, about a minute).
    Returns (days dataframe, trades dict)."""
    x_all = pd.concat([x_train, x_test])
    y_all = pd.concat([y_train, y_test])

    rows = []
    for i in range(len(x_train), len(x_all)):
        kd = calculate_kalman_spread(x_all.iloc[:i + 1], y_all.iloc[:i + 1])
        rd, _ = hmm_regimes(kd, 1, hmm)
        rows.append((rd.index[-1], rd["Z_Score"].iloc[-1], rd["Regimes"].iloc[-1],
                     rd["Asset_Y_Close"].iloc[-1], rd["Predicted"].iloc[-1]))
    days = pd.DataFrame(rows, columns=["date", "z", "regime", "y", "pred"]).set_index("date")

    position = 0
    trades = {"long_in": [], "short_in": [], "exits": []}
    for d, r in days.iterrows():
        if position == 1 and r.z >= 0:
            trades["exits"].append(d); position = 0
        elif position == -1 and r.z <= 0:
            trades["exits"].append(d); position = 0
        elif position == 0:
            if r.z > z_entry and r.regime == calm:
                trades["short_in"].append(d); position = -1
            elif r.z < -z_entry and r.regime == calm:
                trades["long_in"].append(d); position = 1
    print(f"out-of-sample: {len(days)} days | {len(trades['long_in'])} long and {len(trades['short_in'])} short entries | "
          f"{len(trades['exits'])} exits | open at end: {position != 0}")
    return days, trades


def plot_trades_out_of_sample(days, trades, calm, y_name):
    """Out-of-sample trades, same layout as the in-sample chart. Takes the output of replay_out_of_sample."""
    vol = 1 - calm
    n = len(trades["long_in"]) + len(trades["short_in"])
    fig, a1, a2 = _price_and_z_panels(days.index, days["y"], days["pred"], days["z"], days["regime"], vol, y_name,
                                      f"Out-of-sample trades ({n}): entries and exits")
    _trade_markers(a1, a2, days["z"], trades["long_in"], trades["short_in"], trades["exits"])
    fig.tight_layout()
    plt.show()


def plot_fixed_vs_kalman(x_train, x_test, y_train, y_test):
    """Does the pair really mean-revert? Kalman spread vs a fixed train hedge, on the test window."""
    x_all = pd.concat([x_train, x_test])
    y_all = pd.concat([y_train, y_test])
    kalman_all = calculate_kalman_spread(x_all, y_all)
    p_train = coint(x_train, y_train)[1]
    p_test = coint(x_test, y_test)[1]

    b, c = np.polyfit(x_train, y_train, 1)
    fixed = y_test - (b * x_test + c)
    ar1 = np.polyfit(fixed.values[:-1], fixed.values[1:], 1)[0]
    kt = kalman_all.loc[x_test.index, "Spread"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    a1.plot(kt.index, kt, color="tab:blue")
    a1.axhline(0, color="black", lw=1)
    a1.set_title("Kalman spread on test (it re-centres itself, so it looks stationary almost regardless of the pair)", fontsize=10)
    a2.plot(fixed.index, fixed, color="tab:red")
    a2.axhline(0, color="black", lw=1)
    a2.set_title(f"Fixed train hedge (beta {b:.3f}) on test: lag-1 autocorrelation {ar1:.3f}\n"
                 f"close to 1 = wanders off, well below 1 = comes back | cointegration p: train {p_train:.3f}, test {p_test:.3f}",
                 fontsize=10)
    fig.tight_layout()
    plt.show()


