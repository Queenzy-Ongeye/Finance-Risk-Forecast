# Stock Price GARCH Analysis

A comprehensive financial analysis tool that implements GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models to analyze and forecast stock price volatility patterns.

## 📊 Overview

This project provides advanced volatility modeling for financial time series data using GARCH models. It helps investors, analysts, and researchers understand volatility clustering, forecast future volatility, and assess financial risk in stock markets.

## ✨ Features

- **Stock Data Retrieval**: Automated fetching of historical stock prices using Yahoo Finance API
- **GARCH Modeling**: Implementation of various GARCH model variants (GARCH, EGARCH, GJR-GARCH)
- **Volatility Forecasting**: Predict future volatility patterns based on historical data
- **Interactive Visualizations**: Dynamic charts showing price movements, returns, and volatility
- **Statistical Analysis**: Comprehensive statistical tests and model diagnostics
- **Risk Metrics**: Calculate Value at Risk (VaR) and other risk measures
- **Streamlit Dashboard**: User-friendly web interface for analysis

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** - Web application framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Data visualization
- **plotly** - Interactive charts
- **yfinance** - Stock data retrieval
- **arch** - GARCH modeling library
- **scipy/statsmodels** - Statistical analysis

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Queenzy-Ongeye/Stock-Price-GARCH-Analysis.git
   cd Stock-Price-GARCH-Analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Basic Python Usage
```python
import pandas as pd
from arch import arch_model
import yfinance as yf

# Fetch stock data
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Calculate returns
returns = 100 * data['Adj Close'].pct_change().dropna()

# Fit GARCH model
model = arch_model(returns, vol='Garch', p=1, q=1)
fitted_model = model.fit()

# Forecast volatility
forecast = fitted_model.forecast(horizon=30)
```

## 📈 Key Features Explained

### GARCH Models
- **GARCH(1,1)**: Basic model capturing volatility clustering
- **EGARCH**: Exponential GARCH for asymmetric volatility effects
- **GJR-GARCH**: Handles leverage effects in volatility

### Analysis Components
1. **Data Preprocessing**: Handles missing values, outliers, and data cleaning
2. **Exploratory Analysis**: Statistical summaries and preliminary visualizations
3. **Model Fitting**: Automated parameter estimation and model selection
4. **Diagnostics**: Residual analysis and goodness-of-fit tests
5. **Forecasting**: Multi-step ahead volatility predictions
6. **Risk Assessment**: VaR calculations and stress testing

## 📊 Output Examples

The application provides:
- **Time Series Plots**: Stock prices and returns over time
- **Volatility Charts**: Historical and forecasted volatility
- **Statistical Summaries**: Model parameters and fit statistics
- **Risk Metrics**: VaR estimates and confidence intervals
- **Model Comparisons**: Performance metrics across different GARCH variants

## 🔧 Configuration

Customize your analysis by modifying parameters in the configuration section:

- **Time Period**: Adjust start and end dates for analysis
- **GARCH Parameters**: Modify p, q values for different model orders
- **Forecast Horizon**: Set the number of periods to forecast
- **Risk Levels**: Configure VaR confidence levels

## 📝 Model Interpretation

### GARCH Parameters
- **α (alpha)**: Impact of recent shocks on volatility
- **β (beta)**: Persistence of volatility
- **ω (omega)**: Long-term average variance

### Key Metrics
- **AIC/BIC**: Model selection criteria
- **Log-Likelihood**: Model fit quality
- **Ljung-Box Test**: Residual autocorrelation test

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ARCH Library**: For providing robust GARCH implementation
- **Yahoo Finance**: For free access to financial data
- **Streamlit Team**: For the excellent web app framework
- **Financial Econometrics Community**: For theoretical foundations

## 📞 Contact

- **GitHub**: [Queenzy-Ongeye](https://github.com/Queenzy-Ongeye)
- **Email**: queenzyongeye@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🔗 Useful Resources

- [GARCH Models Documentation](https://arch.readthedocs.io/)
- [Financial Time Series Analysis](https://www.investopedia.com/articles/trading/07/statarb.asp)
- [Volatility Modeling Best Practices](https://quantpedia.com/volatility-modeling/)

---

⭐ **Star this repository if you found it helpful!**