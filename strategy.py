from kalman_filter import calculate_kalman_spread
from regimes import hmm_regimes
import pandas as pd

def strategy_test(x_asset, y_asset, hmm, calm):
    split = 1
    current_position = 0
    trade_log = []

    # Generic memory for entry execution
    entry_y = 0.0
    entry_x = 0.0
    entry_beta = 0.0
    entry_date = None

    print("\n🚀 STARTING STRATEGY EXECUTION...")

    for i in range(20, len(x_asset)):
    
        x = x_asset.iloc[:i+1]
        y = y_asset.iloc[:i+1]

        kalman_data = calculate_kalman_spread(x, y)
        regime_data, hmm_test = hmm_regimes(kalman_data, split, hmm)

        current_date = regime_data.index[-1]
        today_z = regime_data["Z_Score"].iloc[-1]
        today_regime = regime_data["Regimes"].iloc[-1]
        
        # Pull from the generic column names
        today_y = regime_data["Asset_Y_Close"].iloc[-1]
        today_x = regime_data["Asset_X_Close"].iloc[-1]
        today_beta = regime_data["Hedge_Ratio"].iloc[-1]

        # --- EXIT LOGIC & PNL ---
        if current_position == 1 and today_z >= 0.0:
            y_pnl = today_y - entry_y
            x_pnl = (entry_x - today_x) * entry_beta
            total_pnl = y_pnl + x_pnl
            
            trade_log.append({'Entry': entry_date, 'Exit': current_date, 'Type': 'Long Spread', 'PnL': total_pnl})
            print(f"[{current_date.date()}] EXIT Long Spread | PnL: ${total_pnl:.2f}")
            current_position = 0  
            
        elif current_position == -1 and today_z <= 0.0:
            y_pnl = entry_y - today_y
            x_pnl = (today_x - entry_x) * entry_beta
            total_pnl = y_pnl + x_pnl
            
            trade_log.append({'Entry': entry_date, 'Exit': current_date, 'Type': 'Short Spread', 'PnL': total_pnl})
            print(f"[{current_date.date()}] EXIT Short Spread | PnL: ${total_pnl:.2f}")
            current_position = 0  

        # --- ENTRY LOGIC ---
        elif current_position == 0:
            if (today_z > 2.0) and (today_regime == calm):
                entry_y, entry_x, entry_beta, entry_date = today_y, today_x, today_beta, current_date
                current_position = -1
                print(f"[{current_date.date()}] ENTRY Short Spread")

            elif (today_z < -2.0) and (today_regime == calm):
                entry_y, entry_x, entry_beta, entry_date = today_y, today_x, today_beta, current_date
                current_position = 1
                print(f"[{current_date.date()}] ENTRY Long Spread")

    pnl_df = pd.DataFrame(trade_log)
    if not pnl_df.empty:
        print("\n--- IN-SAMPLE BACKTEST RESULTS ---")
        print(pnl_df.to_string(index=False))
        print(f"\nTotal PnL: ${pnl_df['PnL'].sum():.2f}")
    else:
        print("\nNo trades were executed.")
        
    return pnl_df