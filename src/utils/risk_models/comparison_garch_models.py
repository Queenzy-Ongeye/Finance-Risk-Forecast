
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import pandas as pd
import numpy as np

# 2. Model Comparison & Selection
def compare_garch_models(returns, name):
    """ Comparing different garch specifications""" 
    models = {
        'GARCH(1, 1)' : arch_model(returns, vol = 'GARCH', p = 1, q = 2),
        'GARCH(1, 2)' : arch_model(returns, vol= 'GARCH', p=1, q=2),
        'GARCH(2, 1)': arch_model(returns, vol= 'GARCH', p=2, q=1),
        'EGARCH(1, 1)' : arch_model(returns, vol= 'EGARCH', p=1, q=1),
        'GJR-GARCH(1, 1)': arch_model(returns, vol= 'GARCH', p=1, o=1, q=1)
    }
    
    results_comparison = {}
    for model_name, model in models.items():
        try:
            fit = model.fit(disp='off')
            results_comparison[model_name] = {
                'AIC': fit.aic,
                'BIC' : fit.bic,
                'Log-Likelihood': fit.loglikelihood
            }
        except:
            print(f"could not fit {model_name} for {name}")
        
    comparison_df = pd.Dataframe(results_comparison).T
    print(f"\n {name} - Model Comparison")
    print(comparison_df.sort_values('AIC'))
    return comparison_df
