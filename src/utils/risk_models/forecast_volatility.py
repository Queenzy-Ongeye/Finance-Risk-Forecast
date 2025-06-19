import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3. VOLATILITY FORECASTING FUNCTION
def forecast_volatility(results, name, horizon=30):
    """Generate volatility forecasts"""
    try:
        forecasts = results.forecast(horizon=horizon)
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical volatility (last 100 points)
        hist_vol = results.conditional_volatility
        plt.plot(hist_vol.iloc[-100:], label='Historical Volatility', color='blue')
        
        # Plot forecasts
        forecast_vol = np.sqrt(forecasts.variance.values[-1, :])
        forecast_index = range(len(hist_vol), len(hist_vol) + horizon)
        
        plt.plot(forecast_index, forecast_vol, 
                label=f'{horizon}-day Forecast', color='red', linestyle='--')
        
        plt.title(f'{name} - Volatility Forecast')
        plt.xlabel('Time Period')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return forecasts
    except Exception as e:
        print(f"Forecasting failed for {name}: {e}")
        return None
