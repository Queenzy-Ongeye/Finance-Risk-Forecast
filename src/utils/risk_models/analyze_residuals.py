import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox


# 1. RESIDUAL ANALYSIS FUNCTION
def analyze_residuals(results, name):
    """Comprehensive residual analysis"""
    residuals = results.resid
    std_residuals = results.std_resid
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{name} - GARCH Model Diagnostics", fontsize=14, fontweight='bold')
    
    # Standardized residuals over time
    axes[0,0].plot(std_residuals)
    axes[0,0].set_title("Standardized Residuals Over Time")
    axes[0,0].grid(True)
    
    # Q-Q Plot for normality
    stats.probplot(std_residuals, dist='norm', plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Test)')
    
    # ACF of squared residuals
    lags = range(1, 21)
    acf_sq_resid = acf(std_residuals ** 2, nlags=20)[1:]
    axes[1, 0].bar(lags, acf_sq_resid)
    axes[1, 0].set_title("ACF of Squared Std. Residuals")
    axes[1, 0].set_xlabel("Lag")
    
    # Histogram of standardized residuals
    axes[1, 1].hist(std_residuals, bins=50, density=True, alpha=0.7)
    axes[1, 1].set_title("Distribution of Std. Residuals")
    
    plt.tight_layout()
    plt.show()
    
    # Ljung-Box test for remaining autocorrelation
    try:
        lb_test = acorr_ljungbox(std_residuals ** 2, lags=10, return_df=True)
        print(f"\n{name} - Ljung-Box Test (p-values < 0.05 indicate remaining autocorrelation):")
        print(lb_test['lb_pvalue'].head())
    except Exception as e:
        print(f"Ljung-Box test failed for {name}: {e}")
