# run_fetch.py

from fetch_data import fetch_asset_data

tickers = ['AAPL', '^GSPC', 'BTC-USD']  # Add or change tickers as needed
fetch_asset_data(tickers, start_date='2015-01-01', save_dir='data')
