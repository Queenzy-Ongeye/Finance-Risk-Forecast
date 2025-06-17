import numpy as np
from scipy.stats import norm

def calculate_risk_metric(results, returns, name, confidence_levels = [0.01, 0.05]):
    """Calculating the VaR expected shortfall"""
    std_resid = results.std_resid
    cond_vol = results.conditional_volatility
    
    risk_metrics = {}
    
    for conf_level in confidence_levels:
        # parametric VaR (assuming normal distribution)
        var_normal = norm.ppf(conf_level) * cond_vol.iloc[-1]
        
        #Historical simulation VaR
        var_hist = np.percentile(returns, conf_level * 100)
        
        # Expected shartfall (CVaR)
        es_hist = returns[returns <= var_hist].mean()
        risk_metrics[f"VaR_{int(conf_level * 100)}%"] = {
            'Parametric': var_normal,
            'Historical': var_hist,
            'Expected_Shortfall': es_hist
        }
    
    print(f"\n{name} - Risk Metrics:")
    for metric, values in risk_metrics.items():
        print(f"{metric}:")
        for method, value in values.items():
            print(f"{method}: {value: .4f}%")
    
    return risk_metrics