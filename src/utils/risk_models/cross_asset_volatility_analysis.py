import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 6. CROSS-ASSET VOLATILITY ANALYSIS FUNCTION
def cross_asset_volatility_analysis(all_results):
    """Analyze volatility relationships across assets"""
    try:
        # Extract conditional volatilities
        vol_data = {}
        for name, results in all_results.items():
            vol_data[name] = results.conditional_volatility
        
        vol_df = pd.DataFrame(vol_data)
        
        # Correlation matrix
        correlation_matrix = vol_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Cross-Asset Volatility Correlations')
        plt.show()
        
        print("\nVolatility Correlation Matrix:")
        print(correlation_matrix)
        
        return correlation_matrix
    except Exception as e:
        print(f"Cross-asset analysis failed: {e}")
        return None