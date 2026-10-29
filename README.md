# Pairs Trading with a Kalman Filter and an HMM

This is my first quant project. I am learning quant finance, so I did the maths on paper first (Kalman filter, ADF, Engle-Granger, HMM, Baum-Welch and Viterbi, all worked by hand with small numbers) and then built the pipeline in Python with `pykalman` and `hmmlearn`.

It is a learning project. It is not a trading system, it does not prove anything makes money, and nothing here is financial advice. I wrote down the problems I found, because they taught me more than the results did.

## The idea

Think of two stocks tied together by a rubber band. Most days they move together. Sometimes the gap between them stretches, and if the band is real, it pulls back. A pairs trade bets on the pull back: sell the one that ran ahead, buy the other, and close when the gap is gone.

Two things can go wrong. The band might not be real (the stocks only looked linked), or the band might move (the hedge ratio drifts). The project is built around those two questions.

![process](pairs_project_process_with_problem.png)

## How it works

1. **Find candidates.** `pairs.py` takes the S&P 500 list from Wikipedia, downloads daily closes from Yahoo Finance since 2020-01-01, and keeps pairs whose price correlation is above 0.95.
2. **Check the band is real.** `cointegration.py` runs the Engle-Granger test (`statsmodels.coint`) and keeps pairs with p < 0.05 (at least 30 common dates). In `main.py` this runs on the whole window, so it is a screen. When I study one pair, I split by date and check cointegration on the train and test parts separately.
3. **Track the hedge ratio.** `kalman_filter.py` uses a Kalman filter with two hidden states, the hedge ratio (beta) and the intercept. Transition covariance is 1e-6 and observation covariance is 1.0. Today's spread uses yesterday's estimate (`.shift(1)`), so there is no peeking.
4. **Score the gap.** The spread is turned into a rolling 20 day z-score.
5. **Label calm and volatile days.** `regimes.py` fits a 2 state Gaussian HMM on the absolute spread. The spread is clipped at 3 standard deviations before fitting (see problem 1 below). I call the lower variance state "calm".
6. **Trade rules.** `strategy.py` enters when the z-score is beyond 2 or below -2 and the day is calm. It exits when the z-score crosses 0. One unit of Y against beta units of X, P&L in dollars, filled at the same day's close, no costs.

`main.py` currently runs the screen (steps 1 and 2) and writes `cointegrated_pairs.csv`. The strategy cells are commented out in `main.py`.

## What I found

**Fixed hedge ratio vs Kalman (KO and PEP, 2023-01-04 to 2026-10-28).** I fit a fixed hedge ratio on the first 252 trading days (2022) and compared it with the Kalman one. The biggest gap between Coca-Cola and its hedge was $42.49 per share with the fixed ratio and $5.94 with the Kalman filter. This shows gap size, not backtest profit. The fixed one is a deliberately simple baseline. Anyone betting the gap closes would have sat through a gap that stayed wide.

**The screen.** 503 tickers gave 1,351 pairs above 0.95 correlation. 660 pass p < 0.05 and 311 pass p < 0.01. If no pair were truly cointegrated, about 68 would still pass at 5% by luck alone (1,351 times 0.05), so a lot of the 660 will not be real. The check ran on the whole window, so it is in-sample, and the CSV has tickers and p-values only, with no hedge ratio and no direction. It is a screen to start from, not a list of trades.

**Real band vs no band.**

| Pair | p-value, train | p-value, test | Lag-1 autocorrelation of the fixed-hedge residual |
|---|---|---|---|
| SPY / IVV | 0.008 | 0.007 | 0.34 |
| V / MA | 0.079 | 0.70 | 0.986 |
| V / AAPL | 0.41 | 0.93 | 0.992 |

A low autocorrelation means the gap comes back fast, close to 1 means it drifts. This was the cleanest separator I found. The Kalman spread on its own hides the difference: for V and MA the half-life looks like 1.2 days with the Kalman spread and 49 days with a fixed hedge. So check cointegration on the fixed hedge before trusting the Kalman spread.

SPY and IVV track the same index, so they work as a demo of the algorithm and not a way to make money: the spread is a few cents, and the three trades made 1 to 3 basis points each.

I checked the numbers in this section with separate replay scripts that are not in this repo yet.

## Problems I found (and what I did)

1. **The HMM collapsed onto one day.** On SPY and IVV, one day (2025-04-09) sat 21 standard deviations out. The HMM spent a whole state on it and the calm filter did nothing. Clipping the spread at 3 standard deviations before fitting fixed it (variances 0.0061 and 0.067, then 3 trades, all wins). The clip bound uses a standard deviation that still includes the outlier, so it is a patch, not a clean fix.
2. **The calm state number is not stable.** Which state the HMM calls 0 flips with the random seed (4 of 8 seeds). With the clip, state 0 is the high variance state for V/MA and V/AAPL, so a rule that says `regime == 0` would trade the volatile days. I now pick calm by reading the variances by hand. That works, but it is manual.
3. **Fills are at the same close that made the signal.** That flatters results. Next-day fills are the fair test and I have not built that yet.
4. **Too few trades to say anything.** With 2 to 11 trades per pair, P&L is noise. V/AAPL goes from -121 bps to +51 bps depending only on which state I trade.
5. **No costs.** No commissions, no borrow cost for the short leg, no slippage.
6. **Old code in `main.py`.** The commented strategy cells call `split_data` with an older set of arguments, so they need fixing before they run.

## What I want to do next

- Fill at the next day's close and see how much of the result is left.
- Add costs.
- Run the same pipeline on more cointegrated pairs from the screen, checking each on a fixed hedge first.
- Write the problem log up properly, with the replay scripts in the repo.

## Run it

Python with `yfinance`, `pandas`, `numpy`, `statsmodels`, `pykalman`, `hmmlearn` and `matplotlib`.

```
python main.py
```

This downloads about 500 tickers, tests every pair above 0.95 correlation, and writes `cointegrated_pairs.csv` (columns: Stock 1, Stock 2, P value). It takes a few minutes. Yahoo data gets revised, so a new run will not match my numbers exactly.

## Files

| File | What it does |
|---|---|
| `pairs.py` | S&P 500 list, download, correlation filter |
| `cointegration.py` | Engle-Granger check on each pair |
| `kalman_filter.py` | Kalman hedge ratio, spread, rolling z-score |
| `regimes.py` | 2 state HMM on the spread |
| `strategy.py` | Entry and exit rules and P&L |
| `plots.py` | Charts |
| `config.py` | Data download and train/test split |
| `main.py` | Runs the screen |
| `cointegrated_pairs.csv` | Output of the screen |

## Disclaimer

This is research and study material, not financial advice. Past data says nothing certain about the future. Do not trade real money on it.
