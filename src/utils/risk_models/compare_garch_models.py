
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import pandas as pd
import numpy as np


# 2. MODEL COMPARISON FUNCTION
def compare_garch_models(returns, name):
    """Compare different GARCH specifications"""
    models_to_test = {
        'GARCH(1,1)': {'vol': 'GARCH', 'p': 1, 'q': 1},
        'GARCH(1,2)': {'vol': 'GARCH', 'p': 1, 'q': 2},
        'GARCH(2,1)': {'vol': 'GARCH', 'p': 2, 'q': 1},
        'EGARCH(1,1)': {'vol': 'EGARCH', 'p': 1, 'q': 1},
        'GJR-GARCH(1,1)': {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1}
    }
    
    results_comparison = {}
    
    for model_name, params in models_to_test.items():
        try:
            model = arch_model(returns, **params)
            fit = model.fit(disp='off')
            results_comparison[model_name] = {
                'AIC': fit.aic,
                'BIC': fit.bic,
                'Log-Likelihood': fit.loglikelihood
            }
        except Exception as e:
            print(f"Could not fit {model_name} for {name}: {str(e)}")
    
    if results_comparison:
        comparison_df = pd.DataFrame(results_comparison).T
        print(f"\n{name} - Model Comparison:")
        print(comparison_df.sort_values('AIC'))
        return comparison_df
    else:
        print(f"No models could be fitted for {name}")
        return None
