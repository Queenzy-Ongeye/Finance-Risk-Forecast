import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Volatility Forecasting
def forecasting_volatility(results, name, horizon = 30):
    """ Generating volatility forecasts"""  
    forecasts = results.forecast(horizon = horizon)
    
    plt.figure(figsize= (12, 8))
    
    # Plotting historical volatility
    hist_vol = results.conditional_volatility
    plt.plot(hist_vol.index[-100: ], hist_vol.iloc[-100: ], 
             label = 'Historical Volatility', color = 'blue')
    
    #Plotting forecasts
    forecast_index = pd.date_range(start = hist_vol.index[-1] + pd.Timedelta(days = 1), periods= horizon, freq='D')
    forcast_vol = np.sqrt(forecasts.variance.values[-1, :]) 
    plt.plot(forecast_index, forcast_vol, label = f"{horizon} - day Forecast", color = 'red', linestyle = "--")
    
    plt.title(f"{name} - Volatility Forecast")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.grind(True)
    plt.show()
    
    return forecasts
           
    
    