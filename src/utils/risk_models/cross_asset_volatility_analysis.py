import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cross_asset_volatility_analysis(all_results):
    """ Analyzeing volatility relationships across assets"""
    # Extract conditional volatilities
    vol_data = {}
    for name, results in all_results.items():
        vol_data[name] = results.conditional_volatility
    
    vol_df = pd.DataFrame(vol_data)
    
    # correlation matrix
    corr_matrix = vol_df.corr()
    
    plt.figure(figsize = (12, 6))
    sns.headmap(corr_matrix, annot = True, cmap = 'coolwarm', center = 0)
    plt.title('Cross-Asset Volatility Correlations')
    plt.show()
    
    # volatility spillover analysis
    print("\n Volatility Correlation Matrix:")
    print(corr_matrix)
    
    return corr_matrix