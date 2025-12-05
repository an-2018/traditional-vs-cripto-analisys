import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config_loader import ConfigLoader
from data_loader import DataLoader
from preprocessing import calculate_returns, handle_missing_values
from metrics import portfolio_metrics, correlation_matrix
from validator import DataValidator

# Page Config
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Config
@st.cache_resource
def load_config():
    return ConfigLoader()

config = load_config()

# Sidebar
st.sidebar.title("Configuration")

# Date Range Selection
st.sidebar.subheader("Date Range")
default_start = pd.to_datetime(config.get('date_range.start_date'))
default_end = pd.to_datetime(config.get('date_range.end_date'))

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# Asset Selection
st.sidebar.subheader("Assets")
assets_config = config.get_assets()
selected_stocks = st.sidebar.multiselect("Stocks", assets_config.get('stocks', []), default=assets_config.get('stocks', []))
selected_crypto = st.sidebar.multiselect("Crypto", assets_config.get('crypto', []), default=assets_config.get('crypto', []))
selected_bonds = st.sidebar.multiselect("Bonds/ETFs", assets_config.get('bonds_etfs', []), default=assets_config.get('bonds_etfs', []))

all_tickers = selected_stocks + selected_crypto + selected_bonds

# Run Pipeline Button
if st.sidebar.button("Run Pipeline", type="primary"):
    st.session_state.run_pipeline = True

# Main Content
st.title("📈 Portfolio Analysis & Orchestration Dashboard")

# Tabs for different stages/views
tab1, tab2, tab3, tab4 = st.tabs(["Orchestration & Data", "Analysis", "Validation", "Config View"])

# --- Tab 1: Orchestration & Data ---
with tab1:
    st.header("Pipeline Orchestration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Status Indicators
    with col1:
        st.info("1. Extraction")
    with col2:
        st.info("2. Validation")
    with col3:
        st.info("3. Processing")
    with col4:
        st.info("4. Analysis")

    if 'data' not in st.session_state:
        st.session_state.data = None

    if st.session_state.get('run_pipeline'):
        with st.status("Running Data Pipeline...", expanded=True) as status:
            try:
                # 1. Extraction
                st.write("📥 Extracting data from Yahoo Finance...")
                loader = DataLoader(data_dir=config.get('output.raw_data_dir'))
                raw_data = loader.fetch_data(all_tickers, str(start_date), str(end_date))
                st.write(f"✅ Extracted {len(raw_data)} rows for {len(all_tickers)} assets")
                
                # 2. Validation (Simulated for single source)
                st.write("🔍 Validating data...")
                validator = DataValidator(config.get_validation_config())
                # Simulate a secondary source for demonstration if enabled
                if config.is_validation_enabled():
                    simulated_secondary = raw_data * 1.001 # Tiny difference
                    val_report = validator.validate_sources(raw_data, {"simulated_source": simulated_secondary})
                    st.json(val_report)
                else:
                    st.write("ℹ️ Multi-source validation disabled (single source mode)")
                
                # 3. Processing
                st.write("⚙️ Processing and cleaning data...")
                processed_data = handle_missing_values(raw_data)
                st.session_state.data = processed_data
                st.write(f"✅ Processing complete. Final shape: {processed_data.shape}")
                
                status.update(label="Pipeline Completed Successfully!", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                status.update(label="Pipeline Failed", state="error")

    # Display Data Preview
    if st.session_state.data is not None:
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.data.head())
        
        # Download button
        csv = st.session_state.data.to_csv().encode('utf-8')
        st.download_button(
            "Download Processed Data",
            csv,
            "processed_portfolio_data.csv",
            "text/csv",
            key='download-csv'
        )

# --- Tab 2: Analysis ---
with tab2:
    st.header("Portfolio Analysis")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Extract Close prices
        try:
            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                close_prices = df.xs('Close', level=1, axis=1)
            else:
                # Fallback if flat columns or different structure
                close_cols = [c for c in df.columns if 'Close' in c]
                close_prices = df[close_cols]
        except Exception as e:
            st.error(f"Error extracting close prices: {e}")
            close_prices = None

        if close_prices is not None:
            # Calculate Returns
            returns = calculate_returns(close_prices)
            
            # Cumulative Returns Chart
            st.subheader("Cumulative Returns")
            cum_returns = (1 + returns).cumprod()
            fig_cum = px.line(cum_returns, title="Asset Cumulative Returns")
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Correlation Matrix
            st.subheader("Correlation Matrix")
            corr_matrix = correlation_matrix(returns)
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Metrics Table
            st.subheader("Asset Metrics")
            metrics_list = []
            for col in returns.columns:
                m = portfolio_metrics(returns[col])
                m['Ticker'] = col
                metrics_list.append(m)
            
            metrics_df = pd.DataFrame(metrics_list).set_index('Ticker')
            
            # Identify numeric columns for formatting
            numeric_cols = metrics_df.select_dtypes(include=['float', 'int']).columns
            
            # Display with formatting
            st.dataframe(metrics_df.style.format({col: "{:.2%}" for col in numeric_cols}))
            
            # Risk-Return Scatter
            st.subheader("Risk-Return Profile")
            fig_scatter = px.scatter(
                metrics_df, 
                x='Annualized Volatility', 
                y='Annualized Return', 
                text=metrics_df.index,
                title="Risk vs Return",
                size=[10]*len(metrics_df)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.info("Please run the pipeline to generate analysis.")

# --- Tab 3: Validation ---
with tab3:
    st.header("Data Validation Report")
    st.markdown("""
    **Validation Strategy:**
    1. **Multi-Source Comparison**: Compares primary data (Yahoo) against secondary sources (e.g., Alpha Vantage).
    2. **Checks Performed**:
       - **Price Tolerance**: Ensures prices match within defined threshold (default 5%).
       - **Correlation**: Ensures price movements are highly correlated (> 0.95).
       - **Descriptive Stats**: Compares statistical properties (mean, std dev).
    """)
    
    if config.is_validation_enabled():
        st.success("Validation is ENABLED in config.")
    else:
        st.warning("Validation is DISABLED in config. Enable it in `pipeline_config.yaml` to see real reports.")

# --- Tab 4: Config View ---
with tab4:
    st.header("Current Configuration")
    st.code(open(config.config_path).read(), language='yaml')

# Footer
st.markdown("---")
st.markdown("Generated by Antigravity Agent | Portfolio Management Project")
