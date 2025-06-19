import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title= "GARCH Volatility Dashboard",
    page_icon= "📈",
    layout = "wide",
    initial_sidebar_state= "expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main >div {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
       font-size: 1rem;
       opacity: 0.9; 
    }
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b, #ff5252);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .alert-medium {
        background: linear-gradient(135deg, #ffc107, #ffb300);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255,193,7,0.3);
    }
    .alert-low {
         background: linear-gradient(135deg, #4caf50, #2e7d32);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(76,175,80,0.3);
    }
    .dashboard-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
</style>        
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown(
    """
        <div class="dashboard-header">
            <h1>GARCH Volatility Dashboard</h1>
            <p> Real-Time volatility monitoring and risk analysis</p>
        </div>
    """, unsafe_allow_html=True
    )

#side bar for data input options
st.sidebar.header("Data Input Options")
input_method = st.sidebar.radio(
    "Choose Input method: ",
    ["Upload CSV Files", "Manual Data Entry", "API Integration"]
)

# Initialize session state for data storage
if 'garch_data' not in st.session_state:
    st.session_state.garch_data = {}
    
def load_sample_data():
    # load sample data
    return {
        'AAPL': {
            'current_vol': 2.45,
            'forecast_vol': [2.78, 2.82, 2.75, 2.69, 2.63, 2.58, 2.54],
            'var_95': -3.42,
            'var_99': -4.89,
            'persistence': 0.957,
            'historical_vol': np.random.normal(2.5, 0.5, 100).tolist(),
            'returns': np.random.normal(0, 2.5, 1000).tolist(),
            'dates': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist(),
            'garch_params': {'alpha': 0.05, 'beta': 0.92, 'omega': 0.01}
        },
        'BTC-USD': {
            'current_vol': 4.23,
            'forecast_vol': [4.89, 4.95, 4.82, 4.67, 4.52, 4.38, 4.25],
            'var_95': -6.15,
            'var_99': -8.34,
            'persistence': 0.966,
            'historical_vol': np.random.normal(4.5, 1.0, 100).tolist(),
            'returns': np.random.normal(0, 4.5, 1000).tolist(),
            'dates': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist(),
            'garch_params': {'alpha': 0.08, 'beta': 0.89, 'omega': 0.02}
        }
    }

def process_csv_data(df):
    """Process CSV data into the expected format for the dashboard"""
    processed_data = {}
    
    # Detect CSV format and process accordingly
    columns = [col.lower().strip() for col in df.columns]
    original_columns = df.columns.tolist()
    
    # Try to detect different CSV formats
    if 'asset' in columns or 'symbol' in columns or 'ticker' in columns or 'name' in columns:
        # Format 1: Multiple assets in one CSV with asset column
        asset_col = None
        for possible_col in ['asset', 'symbol', 'ticker', 'name']:
            if possible_col in columns:
                asset_col = original_columns[columns.index(possible_col)]
                break
        
        if asset_col:
            for asset in df[asset_col].unique():
                if pd.notna(asset):  # Skip NaN values
                    asset_data = df[df[asset_col] == asset].copy()
                    processed_data[str(asset)] = process_asset_data(asset_data, str(asset))
            
    elif len(df.columns) > 3:  # Assume time series data for single asset
        # Format 2: Time series data - try to detect asset name from filename or use first non-date/non-numeric column
        asset_name = detect_asset_name_from_data(df)
        processed_data[asset_name] = process_timeseries_data(df, asset_name)
    
    else:
        # Format 3: Summary statistics format - check if first column contains asset name
        asset_name = detect_asset_name_from_summary(df)
        processed_data[asset_name] = process_summary_data(df, asset_name)
    
    return processed_data

def detect_asset_name_from_data(df):
    """Try to detect asset name from time series data"""
    columns = [col.lower().strip() for col in df.columns]
    
    # Look for common asset identifier columns
    for col_name in df.columns:
        col_lower = col_name.lower().strip()
        if col_lower in ['asset', 'symbol', 'ticker', 'name', 'security']:
            # Use the first non-null value from this column
            first_value = df[col_name].dropna().iloc[0] if not df[col_name].dropna().empty else None
            if first_value:
                return str(first_value)
    
    # If no explicit asset column, check if filename or column names give clues
    for col in df.columns:
        if any(marker in col.upper() for marker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC', 'ETH', 'SPY', 'QQQ']):
            return col
    
    # Check if any column has a consistent non-numeric value that could be an asset name
    for col in df.columns[:3]:  # Check first 3 columns
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 1 and not str(unique_vals[0]).replace('.', '').replace('-', '').isdigit():
                return str(unique_vals[0])
    
    # Default fallback
    return "UPLOADED_ASSET"

def detect_asset_name_from_summary(df):
    """Try to detect asset name from summary statistics data"""
    # Check if there's an explicit asset row
    if len(df.columns) >= 2:
        first_col = df.iloc[:, 0].astype(str).str.lower()
        
        # Look for asset identifier rows
        asset_indicators = ['asset', 'symbol', 'ticker', 'name', 'security']
        for idx, value in enumerate(first_col):
            if any(indicator in value for indicator in asset_indicators):
                asset_name = df.iloc[idx, 1]
                if pd.notna(asset_name):
                    return str(asset_name)
    
    # Check if filename or any value looks like an asset name
    for col in df.columns:
        if any(marker in col.upper() for marker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC', 'ETH', 'SPY', 'QQQ']):
            return col
    
    return "UPLOADED_ASSET"

def process_asset_data(asset_df, asset_name):
    """Process individual asset data"""
    columns = [col.lower().strip() for col in asset_df.columns]
    
    # Default values
    result = {
        'current_vol': 2.0,
        'forecast_vol': [2.0, 2.1, 2.05, 1.95, 1.9, 1.85, 1.8],
        'var_95': -3.0,
        'var_99': -4.5,
        'persistence': 0.95,
        'historical_vol': np.random.normal(2.0, 0.3, 100).tolist(),
        'returns': np.random.normal(0, 2.0, 1000).tolist(),
        'dates': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist(),
        'garch_params': {'alpha': 0.05, 'beta': 0.90, 'omega': 0.01}
    }
    
    # Map columns to expected fields
    column_mapping = {
        'current_vol': ['current_vol', 'volatility', 'vol', 'current_volatility', 'realized_vol'],
        'var_95': ['var_95', 'var95', 'var_0.95', 'value_at_risk_95', 'var_5', 'var5'],
        'var_99': ['var_99', 'var99', 'var_0.99', 'value_at_risk_99', 'var_1', 'var1'],
        'persistence': ['persistence', 'garch_persistence', 'alpha_beta_sum'],
        'alpha': ['alpha', 'garch_alpha', 'arch_param'],
        'beta': ['beta', 'garch_beta', 'garch_param'],
        'omega': ['omega', 'garch_omega', 'constant']
    }
    
    for field, possible_names in column_mapping.items():
        for name in possible_names:
            if name in columns:
                col_idx = columns.index(name)
                value = asset_df.iloc[0, col_idx]
                if pd.notna(value):
                    try:
                        if field in ['alpha', 'beta', 'omega']:
                            result['garch_params'][field] = float(value)
                        else:
                            result[field] = float(value)
                        break
                    except (ValueError, TypeError):
                        continue
    
    # Generate forecast based on current volatility
    current_vol = result['current_vol']
    result['forecast_vol'] = [current_vol * (1 + np.random.normal(0, 0.05)) for _ in range(7)]
    result['historical_vol'] = np.random.normal(current_vol, current_vol*0.2, 100).tolist()
    result['returns'] = np.random.normal(0, current_vol, 1000).tolist()
    
    return result

def process_timeseries_data(df, asset_name):
    """Process time series data"""
    columns = [col.lower().strip() for col in df.columns]
    
    # Try to identify date column
    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp']:
        if col in columns:
            date_col = df.columns[columns.index(col)]
            break
    
    # Try to identify returns column
    returns_col = None
    for col in ['returns', 'return', 'pct_change', 'change', 'log_return']:
        if col in columns:
            returns_col = df.columns[columns.index(col)]
            break
    
    # Try to identify volatility column
    vol_col = None
    for col in ['volatility', 'vol', 'realized_vol', 'garch_vol', 'conditional_vol']:
        if col in columns:
            vol_col = df.columns[columns.index(col)]
            break
    
    # Extract data
    if returns_col and not df[returns_col].isna().all():
        returns = df[returns_col].dropna().tolist()
    else:
        returns = np.random.normal(0, 2.0, len(df)).tolist()
    
    if vol_col and not df[vol_col].isna().all():
        historical_vol = df[vol_col].dropna().tolist()
        current_vol = historical_vol[-1] if historical_vol else 2.0
    else:
        # Calculate rolling volatility from returns if available
        if returns_col and not df[returns_col].isna().all():
            returns_series = df[returns_col].dropna()
            if len(returns_series) > 20:
                rolling_vol = returns_series.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized %
                historical_vol = rolling_vol.dropna().tolist()
                current_vol = historical_vol[-1] if historical_vol else 2.0
            else:
                historical_vol = [returns_series.std() * np.sqrt(252) * 100] * len(returns_series)
                current_vol = historical_vol[-1] if historical_vol else 2.0
        else:
            historical_vol = np.random.normal(2.0, 0.3, len(df)).tolist()
            current_vol = 2.0
    
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d').tolist()
        except:
            dates = pd.date_range('2024-01-01', periods=len(df), freq='D').strftime('%Y-%m-%d').tolist()
    else:
        dates = pd.date_range('2024-01-01', periods=len(df), freq='D').strftime('%Y-%m-%d').tolist()
    
    # Calculate basic statistics
    if returns:
        returns_array = np.array(returns)
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)
    else:
        var_95 = -3.0
        var_99 = -4.5
    
    return {
        'current_vol': float(current_vol),
        'forecast_vol': [current_vol * (1 + np.random.normal(0, 0.05)) for _ in range(7)],
        'var_95': float(var_95),
        'var_99': float(var_99),
        'persistence': 0.95,
        'historical_vol': historical_vol,
        'returns': returns,
        'dates': dates,
        'garch_params': {'alpha': 0.05, 'beta': 0.90, 'omega': 0.01}
    }

def process_summary_data(df, asset_name):
    """Process summary statistics data"""
    # If it's a simple key-value format
    if len(df.columns) == 2:
        # Create dictionary from first two columns
        keys = df.iloc[:, 0].astype(str).str.lower().str.strip()
        values = df.iloc[:, 1]
        summary_dict = dict(zip(keys, values))
        
        # Helper function to safely get numeric values
        def safe_get_float(key_list, default):
            for key in key_list:
                if key in summary_dict:
                    try:
                        return float(summary_dict[key])
                    except (ValueError, TypeError):
                        continue
            return default
        
        current_vol = safe_get_float(['current_vol', 'volatility', 'vol', 'current_volatility'], 2.0)
        var_95 = safe_get_float(['var_95', 'var95', 'var_0.95', 'value_at_risk_95'], -3.0)
        var_99 = safe_get_float(['var_99', 'var99', 'var_0.99', 'value_at_risk_99'], -4.5)
        persistence = safe_get_float(['persistence', 'garch_persistence', 'alpha_beta_sum'], 0.95)
        alpha = safe_get_float(['alpha', 'garch_alpha'], 0.05)
        beta = safe_get_float(['beta', 'garch_beta'], 0.90)
        omega = safe_get_float(['omega', 'garch_omega'], 0.01)
        
        return {
            'current_vol': current_vol,
            'forecast_vol': [current_vol * (1 + np.random.normal(0, 0.05)) for _ in range(7)],
            'var_95': var_95,
            'var_99': var_99,
            'persistence': persistence,
            'historical_vol': np.random.normal(current_vol, current_vol*0.2, 100).tolist(),
            'returns': np.random.normal(0, current_vol, 1000).tolist(),
            'dates': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist(),
            'garch_params': {
                'alpha': alpha,
                'beta': beta,
                'omega': omega
            }
        }
    
    # Default processing
    return load_sample_data()['AAPL']

# Data input section
if input_method == 'Upload CSV Files':
    st.sidebar.subheader("Upload your Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload GARCH results (CSV/JSON)",
        type=['csv', 'json'],
        help="Upload a file containing your GARCH model results"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                st.session_state.garch_data = data
                st.sidebar.success("✅ JSON data loaded successfully!")
            else:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ CSV uploaded!")
                
                # Show preview
                with st.sidebar.expander("📊 Data Preview"):
                    st.dataframe(df.head())
                
                # Process CSV data
                with st.sidebar.expander("⚙️ Processing Options"):
                    processing_method = st.radio(
                        "Data Format:",
                        ["Auto-detect", "Time Series", "Summary Statistics", "Multi-Asset"]
                    )
                    
                    if st.button("Process CSV Data"):
                        processed_data = process_csv_data(df)
                        if processed_data:
                            st.session_state.garch_data.update(processed_data)
                            st.sidebar.success(f"✅ Processed {len(processed_data)} asset(s)!")
                        else:
                            st.sidebar.error("❌ Could not process CSV data. Please check format.")
                            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            st.sidebar.info("Please check your file format and try again.")
        
    # Show expected formats
    with st.sidebar.expander("📋 Expected CSV Formats"):
        st.markdown("""
        **Format 1 - Multi Asset:**
        ```
        Asset,Current_Vol,VaR_95,VaR_99,Persistence
        AAPL,2.45,-3.42,-4.89,0.957
        BTC,4.23,-6.15,-8.34,0.966
        ```
        
        **Format 2 - Time Series:**
        ```
        Date,Returns,Volatility
        2024-01-01,0.1,2.1
        2024-01-02,-0.2,2.3
        ```
        
        **Format 3 - Summary:**
        ```
        Metric,Value
        current_vol,2.45
        var_95,-3.42
        persistence,0.957
        ```
        """)

elif input_method == "Manual Data Entry":
    st.sidebar.subheader("✏️ Manual Entry")
    
    asset_name = st.sidebar.text_input("Asset Name", "CUSTOM_ASSET")
    current_vol = st.sidebar.number_input("Current Volatility (%)", 0.0, 100.0, 2.5)
    var_95 = st.sidebar.number_input("VaR 95% (%)", -50.0, 0.0, -3.0)
    var_99 = st.sidebar.number_input("VaR 99% (%)", -50.0, 0.0, -5.0)
    persistence = st.sidebar.number_input("GARCH Persistence", 0.0, 1.0, 0.95)
    
    if st.sidebar.button("Add Asset Data"):
        st.session_state.garch_data[asset_name] = {
            'current_vol': current_vol,
            'forecast_vol': [current_vol * (1 + np.random.normal(0, 0.1)) for _ in range(7)],
            'var_95': var_95,
            'var_99': var_99,
            'persistence': persistence,
            'historical_vol': np.random.normal(current_vol, current_vol*0.2, 100).tolist(),
            'returns': np.random.normal(0, current_vol, 1000).tolist(),
            'dates': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist(),
            'garch_params': {'alpha': 0.05, 'beta': 0.92, 'omega': 0.01}
        }
        st.sidebar.success(f"✅ {asset_name} added!")

else:  # API Integration
    st.sidebar.subheader("🔗 API Integration")
    api_endpoint = st.sidebar.text_input("API Endpoint URL", placeholder="https://your-api.com/garch-data")
    api_key = st.sidebar.text_input("API Key", type="password")
    
    if st.sidebar.button("Fetch Data from API"):
        # Placeholder for API integration
        st.sidebar.info("🔄 API integration code goes here")
        st.sidebar.code("""
        import requests
        
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get(api_endpoint, headers=headers)
        data = response.json()
        st.session_state.garch_data = data
        """)

# Load sample data if no data exists
if not st.session_state.garch_data:
    st.session_state.garch_data = load_sample_data()
    st.info("📊 Using sample data for demonstration. Upload your own data using the sidebar.")

# Asset selection
available_assets = list(st.session_state.garch_data.keys())
if available_assets:
    selected_asset = st.selectbox("🎯 Select Asset:", available_assets)
    
    # Get selected asset data
    asset_data = st.session_state.garch_data[selected_asset]
    
    # Display key metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{asset_data['current_vol']:.2f}%</div>
            <div class="metric-label">Current Volatility</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_forecast = np.mean(asset_data['forecast_vol'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_forecast:.2f}%</div>
            <div class="metric-label">Avg Forecast Vol</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{asset_data['var_95']:.2f}%</div>
            <div class="metric-label">VaR (95%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{asset_data['persistence']:.3f}</div>
            <div class="metric-label">GARCH Persistence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.markdown("---")
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown("### 📈 Volatility Timeline & Forecast")
        
        fig_vol = go.Figure()
        
        # Historical volatility
        dates = pd.to_datetime(asset_data['dates'])
        fig_vol.add_trace(go.Scatter(
            x=dates,
            y=asset_data['historical_vol'],
            mode='lines',
            name='Historical Volatility',
            line=dict(color='#4ecdc4', width=2),
            fill='tonexty',
            fillcolor='rgba(78, 205, 196, 0.1)'
        ))
        
        # Forecast
        forecast_dates = pd.date_range(
            start=dates[-1] + timedelta(days=1),
            periods=len(asset_data['forecast_vol']),
            freq='D'
        )
        
        fig_vol.add_trace(go.Scatter(
            x=forecast_dates,
            y=asset_data['forecast_vol'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=8, color='#ff6b6b')
        ))
        
        fig_vol.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=True,
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with chart_col2:
        st.markdown("### 🎯 Risk Metrics")
        
        # VaR comparison chart
        fig_var = go.Figure()
        
        fig_var.add_trace(go.Bar(
            x=['VaR 95%', 'VaR 99%'],
            y=[abs(asset_data['var_95']), abs(asset_data['var_99'])],
            marker_color=['#ffc107', '#ff6b6b'],
            text=[f"{asset_data['var_95']:.2f}%", f"{asset_data['var_99']:.2f}%"],
            textposition='auto'
        ))
        
        fig_var.update_layout(
            template='plotly_dark',
            height=200,
            showlegend=False,
            yaxis_title="VaR (%)",
            title="Value at Risk Comparison"
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Risk gauge
        st.markdown("### 🌡️ Risk Level")
        current_vol = asset_data['current_vol']
        
        if current_vol < 2:
            risk_level = "Low"
            risk_color = "#28a745"
        elif current_vol < 4:
            risk_level = "Medium"
            risk_color = "#ffc107"
        else:
            risk_level = "High"
            risk_color = "#ff6b6b"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_vol,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Level: {risk_level}"},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 4], 'color': "yellow"},
                    {'range': [4, 10], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ))
        
        fig_gauge.update_layout(
            template='plotly_dark',
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Returns analysis
    st.markdown("---")
    returns_col1, returns_col2 = st.columns(2)
    
    with returns_col1:
        st.markdown("### 📊 Returns Distribution")
        
        fig_hist = px.histogram(
            x=asset_data['returns'],
            nbins=50,
            title="Returns Distribution",
            template='plotly_dark'
        )
        fig_hist.update_traces(marker_color='#667eea', opacity=0.7)
        fig_hist.add_vline(x=asset_data['var_95'], line_dash="dash", line_color="#ffc107",
                          annotation_text=f"VaR 95%: {asset_data['var_95']:.2f}%")
        fig_hist.add_vline(x=asset_data['var_99'], line_dash="dash", line_color="#ff6b6b",
                          annotation_text=f"VaR 99%: {asset_data['var_99']:.2f}%")
        fig_hist.update_layout(xaxis_title="Returns (%)", yaxis_title="Frequency")
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with returns_col2:
        st.markdown("### ⚙️ GARCH Parameters")
        
        if 'garch_params' in asset_data:
            params = asset_data['garch_params']
            
            # Create parameter visualization
            fig_params = go.Figure()
            
            param_names = list(params.keys())
            param_values = list(params.values())
            
            fig_params.add_trace(go.Bar(
                x=param_names,
                y=param_values,
                marker_color=['#4ecdc4', '#ff6b6b', '#ffc107'],
                text=[f"{v:.4f}" for v in param_values],
                textposition='auto'
            ))
            
            fig_params.update_layout(
                template='plotly_dark',
                height=300,
                title="GARCH Model Parameters",
                yaxis_title="Parameter Value",
                showlegend=False
            )
            
            st.plotly_chart(fig_params, use_container_width=True)
        
        # Model summary table
        st.markdown("**Model Summary:**")
        summary_data = {
            "Metric": ["Current Volatility", "Persistence", "VaR 95%", "VaR 99%"],
            "Value": [f"{asset_data['current_vol']:.2f}%", 
                     f"{asset_data['persistence']:.3f}",
                     f"{asset_data['var_95']:.2f}%",
                     f"{asset_data['var_99']:.2f}%"]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True)
    
    # Risk alerts
    st.markdown("---")
    st.markdown("### ⚠️ Risk Alerts")
    
    # Generate alerts based on data
    if asset_data['current_vol'] > 5:
        st.markdown(f"""
        <div class="alert-high">
            🚨 <strong>High Volatility Alert:</strong> {selected_asset} volatility is {asset_data['current_vol']:.2f}%, exceeding 5% threshold
        </div>
        """, unsafe_allow_html=True)
    
    if asset_data['persistence'] > 0.98:
        st.markdown(f"""
        <div class="alert-medium">
            ⚡ <strong>High Persistence Warning:</strong> GARCH persistence is {asset_data['persistence']:.3f}, indicating slow volatility decay
        </div>
        """, unsafe_allow_html=True)
    
    if abs(asset_data['var_95']) > 5:
        st.markdown(f"""
        <div class="alert-high">
            💥 <strong>Extreme VaR Alert:</strong> 95% VaR is {asset_data['var_95']:.2f}%, indicating high tail risk
        </div>
        """, unsafe_allow_html=True)
    
    if not (asset_data['current_vol'] > 5 or asset_data['persistence'] > 0.98 or abs(asset_data['var_95']) > 5):
        st.markdown("""
        <div class="alert-low">
            ✅ <strong>All Clear:</strong> All risk metrics are within normal ranges
        </div>
        """, unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export current asset data as JSON
        json_data = json.dumps({selected_asset: asset_data}, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"{selected_asset}_garch_data.json",
            mime="application/json"
        )
    
    with col2:
        # Export forecast data as CSV
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(start=pd.to_datetime(asset_data['dates'][-1]) + timedelta(days=1),
                                periods=len(asset_data['forecast_vol']), freq='D'),
            'Forecast_Volatility': asset_data['forecast_vol']
        })
        st.download_button(
            label="📊 Download Forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name=f"{selected_asset}_forecast.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export summary report
        report = f"""
        GARCH Volatility Report - {selected_asset}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Current Volatility: {asset_data['current_vol']:.2f}%
        GARCH Persistence: {asset_data['persistence']:.3f}
        VaR (95%): {asset_data['var_95']:.2f}%
        VaR (99%): {asset_data['var_99']:.2f}%
        
        Forecast (next {len(asset_data['forecast_vol'])} days):
        {', '.join([f'{v:.2f}%' for v in asset_data['forecast_vol']])}
        """
        
        st.download_button(
            label="📋 Download Report",
            data=report,
            file_name=f"{selected_asset}_report.txt",
            mime="text/plain"
        )

else:
    st.warning("⚠️ No data available. Please upload your GARCH results using the sidebar.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>GARCH Volatility Dashboard</strong> | Built with Streamlit</p>
    <p>⚡ Connect your existing GARCH models and VaR calculations</p>
</div>
""", unsafe_allow_html=True)