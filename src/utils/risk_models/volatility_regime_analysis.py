import numpy as np
import matplotlib.pyplot as plt



# 5. VOLATILITY REGIME ANALYSIS FUNCTION
def volatility_regime_analysis(cond_vol, name, threshold_percentile=75):
    """Identify high/low volatility regimes"""
    try:
        threshold = np.percentile(cond_vol, threshold_percentile)
        
        # Create regime indicator
        high_vol_regime = cond_vol > threshold
        
        # Calculate regime statistics
        regime_stats = {
            'High Vol Periods': high_vol_regime.sum(),
            'Low Vol Periods': (~high_vol_regime).sum(),
            'Avg High Vol': cond_vol[high_vol_regime].mean(),
            'Avg Low Vol': cond_vol[~high_vol_regime].mean(),
            'High Vol %': (high_vol_regime.sum() / len(cond_vol)) * 100
        }
        
        print(f"\n{name} - Volatility Regime Analysis:")
        for stat, value in regime_stats.items():
            print(f"{stat}: {value:.2f}")
        
        # Plot regimes
        plt.figure(figsize=(12, 6))
        plt.plot(cond_vol, alpha=0.7, label='Conditional Volatility')
        plt.axhline(y=threshold, color='red', linestyle='--', 
                    label=f'{threshold_percentile}th Percentile Threshold')
        plt.fill_between(range(len(cond_vol)), 0, cond_vol, 
                         where=high_vol_regime, alpha=0.3, color='red', 
                         label='High Volatility Regime')
        plt.title(f'{name} - Volatility Regimes')
        plt.xlabel('Time Period')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return regime_stats
    except Exception as e:
        print(f"Regime analysis failed for {name}: {e}")
        return None
