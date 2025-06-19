import numpy as np
from scipy import stats
from scipy.stats import norm

# 4. RISK METRICS FUNCTION
def calculate_risk_metrics(results, returns, name, confidence_levels=[0.01, 0.05]):
    """Calculate VaR and Expected Shortfall"""
    try:
        std_resid = results.std_resid
        cond_vol = results.conditional_volatility
        
        risk_metrics = {}
        
        for conf_level in confidence_levels:
            # Parametric VaR (assuming normal distribution)
            var_normal = stats.norm.ppf(conf_level) * cond_vol.iloc[-1]
            
            # Historical simulation VaR
            var_hist = np.percentile(returns, conf_level * 100)
            
            # Expected Shortfall (CVaR)
            es_hist = returns[returns <= var_hist].mean()
            
            risk_metrics[f'VaR_{int(conf_level*100)}%'] = {
                'Parametric': var_normal,
                'Historical': var_hist,
                'Expected_Shortfall': es_hist
            }
        
        print(f"\n{name} - Risk Metrics:")
        for metric, values in risk_metrics.items():
            print(f"{metric}:")
            for method, value in values.items():
                print(f"  {method}: {value:.4f}%")
        
        return risk_metrics
    except Exception as e:
        print(f"Risk metrics calculation failed for {name}: {e}")
        return None