import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def plot_prices_and_returns(datasets, figsize=(14, 10), recent_days = None):
    """
    Plots the prices and log returns for multiple assets with improved formatting.

    Args:
        datasets (list): A list of pandas DataFrames containing stock price data.
        figsize (tuple): The size of the figure (width, height).
        recent_days (int): The number of recent days to plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(2, len(datasets), figsize=figsize)
    
    # aplying style for better readability
    plt.style.use('seaborn-v0_8-whitegrid')
    
     # Determine colors for each asset (using a colorblind-friendly palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # if the dataset is one only, convert the axes to 2D array
    if len(datasets) == 1:
        ax = np.array(ax).reshape(2, 1)
    
    # Processing each dataset
    for i, (asset_name, df) in enumerate(datasets.items()):
        data = df.copy()
        
        # Applying time filtering if requested
        if recent_days is not None:
            data = data.iloc[-recent_days:]
        
        # Plotting prices ON TOP ROW
        ax_price = ax[0, i]
        data['Price'].plot(ax=ax_price, color=colors[i % len(colors)], linewidth=1.5)
        
        # Format price plot
        ax_price.set_title(f'{asset_name} Price', fontweight = 'bold')
        ax_price.set_ylabel('Price ($)', fontweight='bold')
        ax_price.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.2f}'))
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Plotting returns ON BOTTOM ROW
        ax_returns = ax[1, i]
        data['log_return'].plot(ax = ax_returns, color = colors[i % len(colors)], linewidth = 1.5)
        
        # Adding the mean line to log returns
        mean_return = data['log_return'].mean()
        ax_returns.axhline(y = mean_return, color = 'red', linestyle = '--', linewidth = 1.5, label = f"Mean : {mean_return:.6f}")
        
        # Adding zero line for reference
        ax_returns.axhline(y = 0, color = 'black', linestyle = '-', linewidth = 1.5, alpha = 0.3)
        
        # formtting log returns plot
        ax_returns.set_title(f"{asset_name} Log Returns", fontweight = 'bold')
        ax_returns.set_ylabel('Log Return', fontweight = 'bold')
        ax_returns.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_returns.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax_returns.legend(loc = 'upper right', frameon = True, framealpha = 0.9)
        
        # Rotate date labels for better readability
        for label in ax_returns.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    # adjusting layout
    plt.tight_layout()
    return fig
    
    
    
    
    

