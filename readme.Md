# 📉 Finance Risk Forecasting
A time series modeling project that forecasts financial risk using historical asset prices. This project analyzes log returns, estimates volatility with GARCH models, and computes Value at Risk (VaR) using multiple techniques. The results are visualized through an interactive Streamlit dashboard.

---

## 🌟 Project Objective

To build a robust system that:

* Retrieves asset price data (stocks, crypto, indices) via Yahoo Finance
* Cleans and preprocesses time series data
* Computes log returns and volatility
* Forecasts risk metrics such as Value at Risk (VaR)
* Visualizes results interactively using Streamlit

---

## 🏠 Project Structure

```
finance_risk_forecast/
├── data/                  # Contains downloaded raw and processed CSVs
├── notebooks/             # Jupyter notebooks for exploration and modeling
│   ├── 01_price_fetching.ipynb
│   ├── 02_volatility_analysis.ipynb
│   └── 03_var_modeling.ipynb
├── src/                   # Core Python codebase
│   ├── data/              # Data fetching logic
│   ├── utils/             # Preprocessing, helpers
│   ├── risk_models/       # GARCH and VaR model code
│   └── visualization/     # Plotting functions
├── streamlit/             # Streamlit dashboard and components
│   ├── app.py             # Main Streamlit app
│   └── pages/             # Subpages for visualizations
├── tests/                 # Unit tests for components
├── requirements.txt       # Python dependencies
├── README.md              # Project overview
└── setup.py               # Optional: pip installable package
```

---

## 🧐 Key Features

* 📦 Fetch asset prices using `yfinance`
* 🧹 Preprocess and compute log returns
* 🔥 Forecast volatility with GARCH
* 📉 Calculate VaR (Historical, Parametric, Monte Carlo)
* 📊 Visualize prices, returns, and risk metrics
* 🌐 Deploy results via a Streamlit app

---

## 🚀 Getting Started

### 🔧 1. Clone the repo

```bash
git clone https://github.com/yourusername/finance-risk-forecast.git
cd finance-risk-forecast
```

### 🐍 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 📦 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 💻 4. Run Streamlit app

```bash
streamlit run streamlit/app.py
```

---

## 📓 Notebooks Overview

| Notebook                       | Purpose                         |
| ------------------------------ | ------------------------------- |
| `01_price_fetching.ipynb`      | Download & visualize price data |
| `02_volatility_analysis.ipynb` | Compute returns and GARCH vol   |
| `03_var_modeling.ipynb`        | Calculate Value at Risk         |

---

## 📁 Sample Output

* ✅ AAPL, BTC-USD, GSPC price trends
* ✅ Daily log returns and rolling volatility
* ✅ Historical & Parametric VaR for 95%, 99% confidence levels
* ✅ GARCH volatility forecast curve

---

## 📊 Streamlit Dashboard

Features:

* Asset selector dropdown
* Price & return charts
* VaR plots with dynamic confidence levels
* GARCH volatility forecasts
* Downloadable CSV of forecasts

---

## 🧲 Testing

```bash
pytest tests/
```

---

## 🔮 Future Work

* Portfolio VaR (multi-asset weights)
* Conditional Expected Shortfall (CVaR)
* Copula modeling for joint risk
* Live data streaming and re-forecasting

---

## 🤝 Contributing

Feel free to fork this repo and submit pull requests. Ideas, improvements, or bug fixes are welcome!

---

## 📜 License

MIT License

## 👨‍💻 Author

Quienzy Ong'eye — [@quienzy-ong'eye](https://www.linkedin.com/in/quienzy-ong-eye-a66491210/)
