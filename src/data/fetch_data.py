import yfinance as yf
import os
from datetime import datetime

# function for fetching data
def fetch_asset_data(tickers, start_date = '2015-01-01', end_date = None, save_dir = 'data'):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    os.makedirs(save_dir, exist_ok= True)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start = start_date, end = end_date)
            if df.empty:
                print(f"No data for {ticker}")
                continue
        
            filename = f"{ticker.replace('^', '')}.csv"
            df.to_csv(os.path.join(save_dir, filename))
            print(f"saved {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")