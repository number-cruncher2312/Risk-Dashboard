import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import streamlit as st

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser





warnings.filterwarnings('ignore')




# Initialize session state for storing analyses
if 'deep_analysis' not in st.session_state:
    st.session_state.deep_analysis = None
if 'fast_analysis' not in st.session_state:
    st.session_state.fast_analysis = None
if 'cached_ticker' not in st.session_state:
    st.session_state.cached_ticker = None
if 'cached_metrics' not in st.session_state:
    st.session_state.cached_metrics = None
if 'cached_returns' not in st.session_state:
    st.session_state.cached_returns = None
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = ""

@st.cache_data # Cache this so it doesn't reload every click
def get_sp500_tickers():
    # Wikipedia has a clean table of the S&P 500
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].head(20).tolist() # Just top 20 for speed, or all
    except:
        return ["AAPL", "MSFT", "GOOGL"] # Fallback


def calculate_volatility(returns):
    """Calculate daily and annualized volatility"""
    daily_volatility = float(np.std(returns) * 100)
    yearly_volatility = daily_volatility * np.sqrt(252)
    return daily_volatility, yearly_volatility


def calculate_drawdown(close_prices):
    """Calculate drawdown series and max drawdown"""
    cumulative_max = close_prices.cummax()
    drawdown = (close_prices - cumulative_max) / cumulative_max * 100
    max_drawdown = float(drawdown.min())
    return drawdown, max_drawdown


def calculate_var_cvar_breach(returns, confidence_level):
    """
    Calculate VaR, CVaR, and breach rate for given confidence level
    
    Args:
        returns: pandas Series of log returns
        confidence_level: int (90, 95, or 99)
    
    Returns:
        dict with var, cvar, breach_rate, var_cutoff
    """
    returns_clean = returns.dropna()
    percentile = 100 - confidence_level
    
    # Historical VaR (percentile method)
    var_cutoff = float(np.percentile(returns_clean, percentile))
    var = 100 * abs(var_cutoff)
    window = 60
    window_returns = returns_clean[-window:]
    var_last60_raw = np.percentile(window_returns, percentile)
    
    # Calculate CVaR BEFORE converting to percentage
    bad_days = window_returns[window_returns <= var_last60_raw]
    cvar_last60_raw = np.mean(bad_days) if len(bad_days) > 0 else var_last60_raw
    
    # Now convert to percentages
    var_last60 = 100 * abs(float(var_last60_raw))
    cvar_last60 = 100 * abs(float(cvar_last60_raw))
    
    rolling_vars = []
    breaches = []

    for i in range(window, len(returns_clean)):
        history = returns_clean.iloc[i - window: i]
        var_i = np.percentile(history, percentile)
        today = returns_clean.iloc[i]
        breached = today <= var_i
        rolling_vars.append(var_i)
        breaches.append(breached)
    
    breach_count = sum(breaches)
    breach_rate = float((breach_count / len(breaches)) * 100) if len(breaches) > 0 else 0.0
    
    var = var_last60
    cvar = cvar_last60
    
    return {
        'var': float(var),
        'cvar': float(cvar),
        'breach_rate': float(breach_rate),
        'var_cutoff': float(var_cutoff)
    }



st.title("Financial Risk Metrics Dashboard")

# Use selected_ticker from sidebar if available
default_ticker = st.session_state.selected_ticker if st.session_state.selected_ticker else ""
ticker = st.text_input("Enter Stock Ticker", value=default_ticker, placeholder="e.g., AAPL, TSLA, MSFT")

    # Confidence Level Selection
confidence_level = st.radio(
        "Confidence Level",
        options=[90, 95, 99],
        horizontal=True
    )
confx  = 100  - confidence_level



if ticker:
    # Only recalculate if ticker changed
    if st.session_state.cached_ticker != ticker:
        # Quick validation first - check if ticker exists
        test_data = yf.download(ticker, period="1d", progress=False)
        if test_data.empty:
            st.error(f"❌ Invalid ticker '{ticker}'. Please enter a valid stock symbol.")
            st.session_state.cached_ticker = None
            st.session_state.cached_metrics = None
            st.session_state.cached_returns = None
            st.stop()
        
        with st.spinner("Fetching data and calculating metrics..."):
            try:
                yearly_stock_data = yf.download(ticker, period="1y", progress=False)

                returns = np.log(yearly_stock_data['Close']/yearly_stock_data['Close'].shift(1))

                daily_volatility, yearly_volatility = calculate_volatility(returns)
                
                # Calculate drawdown
                close_prices = yearly_stock_data['Close']
                drawdown, max_drawdown = calculate_drawdown(close_prices)
                
                # Cache the raw data for recalculation
                st.session_state.cached_ticker = ticker
                st.session_state.cached_returns = returns
                st.session_state.cached_drawdown = drawdown
                st.session_state.cached_metrics = {
                    'yearly_volatility': yearly_volatility,
                    'drawdown': drawdown,
                    'max_drawdown': max_drawdown
                }

                
                # Clear old analyses when ticker changes
                st.session_state.deep_analysis = None
                st.session_state.fast_analysis = None
                
            except Exception as e:
                st.error(f"❌ Error fetching data for '{ticker}': {str(e)}")
                st.session_state.cached_ticker = None
                st.session_state.cached_metrics = None
                st.session_state.cached_returns = None
                st.stop()
    
    # Check if we have valid cached metrics
    if st.session_state.cached_metrics is None:
        st.warning("⚠️ No data available. Please enter a valid ticker.")
        st.stop()
    
    # Use cached metrics
    metrics = st.session_state.cached_metrics
    yearly_volatility = metrics['yearly_volatility']
    drawdown = metrics['drawdown']
    max_drawdown = metrics['max_drawdown']
    returns = st.session_state.cached_returns
    
    # Ensure returns is available
    if returns is None:
        st.error("⚠️ Error: Returns data not available. Please reload the ticker.")
        st.stop()
    
    # Recalculate var, cvar, and breach rate based on confidence level
    risk_metrics = calculate_var_cvar_breach(returns, confidence_level)
    var = risk_metrics['var']
    cvar = risk_metrics['cvar']
    breach_rate = risk_metrics['breach_rate']

    # Determine color based on breach rate vs expected
    expected_breach_rate = confx
    deviation = abs(breach_rate - expected_breach_rate)
    
    if deviation <= 1:  # Within 1% of expected
        color = "green"
    elif deviation <= 3:  # Within 3% of expected
        color = "orange"
    else:  # More than 3% off
        color = "red"

    st.write(f"<h2 style='color: {color}'>Breach Rate: {breach_rate:.2f}% (Expected: {expected_breach_rate:.1f}%)</h2>", unsafe_allow_html=True)

    prompt_fast = ChatPromptTemplate.from_template(
            """
            Generate a concise risk assessment for {ticker} based on these metrics:
            - Annualized Volatility: {annual_vol:.2f}%
            - 95% Daily Value at Risk (VaR): {var:.2f}%
            - 95% Expected Shortfall (CVaR): {cvar:.2f}%

            Output ONLY the following sections. Do not include any preamble or introduction. Start directly with the Executive Summary:

            **Executive Summary:** (1 sentence classification of the risk level: Low/Moderate/High/Extreme).
            
            **Tail Risk Analysis:** Explain the gap between VaR and CVaR. If the gap is large, highlight the presence of "fat tails" or extreme downside events. Compare to S&P 500 benchmarks (Vol ~15%, VaR ~1%).
            
            **Investment Suitability:** Briefly describe the type of portfolio this asset fits (e.g., "High-growth speculative allocation" vs "Core defensive holding").

            Be professional, objective, and data-driven. No fluff.
            """
    )

    prompt_deep = ChatPromptTemplate.from_template(
            """
            Generate a comprehensive risk assessment for {ticker} based on these metrics:
            - Annualized Volatility: {annual_vol:.2f}%
            - 95% Daily Value at Risk (VaR): {var:.2f}%
            - 95% Expected Shortfall (CVaR): {cvar:.2f}%

            Output ONLY the following sections with detailed analysis. Do not include any preamble, introduction, or role description. Start directly with section 1:

            **1. Executive Summary & Risk Classification:**
            - Provide a detailed classification of the risk level (Low/Moderate/High/Extreme)
            - Explain the reasoning behind this classification with specific metric comparisons
            - Compare to market benchmarks (S&P 500: ~15% vol, ~1% VaR; Tech sector: ~20-25% vol; Crypto: ~60-80% vol)

            **2. Volatility Analysis:**
            - Interpret the {annual_vol:.2f}% annualized volatility in detail
            - Explain what this means for daily price swings and investor experience
            - Compare to historical volatility patterns and sector norms
            - Discuss implications for portfolio construction and position sizing

            **3. Value at Risk (VaR) Deep Dive:**
            - Explain the {var:.2f}% VaR metric in practical terms
            - Calculate and explain the dollar impact on a $100,000 investment
            - Discuss the 95% confidence level and what the remaining 5% represents
            - Compare this VaR to industry standards

            **4. Tail Risk & CVaR Analysis:**
            - Explain the {cvar:.2f}% Expected Shortfall (CVaR) metric
            - Calculate the gap between VaR ({var:.2f}%) and CVaR ({cvar:.2f}%)
            - If gap is >30%: Highlight severe fat-tail risk and potential for extreme losses
            - If gap is 15-30%: Moderate tail risk with occasional extreme events
            - If gap is <15%: Relatively normal distribution with limited tail risk
            - Discuss what this means for worst-case scenarios and stress testing

            **5. Market Regime Considerations:**
            - Discuss how this asset might behave in different market conditions (bull/bear/sideways)
            - Consider correlation with broader market indices
            - Identify potential catalysts that could increase or decrease risk

            **6. Portfolio Integration Strategy:**
            - Recommend specific portfolio allocation percentages based on risk tolerance:
              * Conservative portfolios (low risk tolerance): X%
              * Moderate portfolios (balanced approach): Y%
              * Aggressive portfolios (high risk tolerance): Z%
            - Suggest complementary assets or hedging strategies
            - Discuss position sizing and risk management techniques

            **7. Risk Mitigation Recommendations:**
            - Provide specific, actionable recommendations for managing downside risk
            - Suggest stop-loss levels or hedging strategies if appropriate
            - Recommend monitoring frequency and key indicators to watch

            **8. Investment Suitability Assessment:**
            - Define the ideal investor profile for this asset
            - Describe investment horizons (short/medium/long-term suitability)
            - List key risk factors investors should be aware of

            Be thorough, analytical, and specific. Provide numbers and actionable insights.
            """
    )

    # Display metrics first
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Volatility", f"{yearly_volatility:.2f}%")
    col2.metric("Daily VaR", f"{var:.2f}%")
    col3.metric("Expected Shortfall",f"{cvar:.2f}%")
    col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")



    # Drawdown Chart
    st.subheader("  Drawdown Chart")
    drawdown_df = pd.DataFrame(drawdown)
    drawdown_df.columns = ['Drawdown (%)']
    st.area_chart(drawdown_df)

    st.divider()
    
    # Put both buttons side by side
    col_deep, col_fast = st.columns(2)
    
    with col_deep:
        deep_clicked = st.button("Generate Deep AI Analysis")
    
    with col_fast:
        fast_clicked = st.button("Generate Fast AI Analysis")
    
    if deep_clicked:
        with st.spinner("Generating deep AI analysis..."):
            try:
                llm1 = ChatNVIDIA(model="deepseek-ai/deepseek-v3.1", max_tokens=4000)
                response = llm1.invoke(prompt_deep.format(ticker=ticker, annual_vol=yearly_volatility, var=var, cvar=cvar))
                st.session_state.deep_analysis = response.content
                    
            except Exception as e:
                st.error(f"Error generating deep analysis: {str(e)}")
                st.session_state.deep_analysis = None
    
    if st.session_state.deep_analysis:
        st.write("### Deep AI Analysis")
        st.write(st.session_state.deep_analysis)

    if fast_clicked:
        with st.spinner("Generating fast AI analysis..."):
            try:
                llm1 = ChatNVIDIA(model="deepseek-ai/deepseek-v3.1")
                response = llm1.invoke(prompt_fast.format(ticker=ticker, annual_vol=yearly_volatility, var=var, cvar=cvar))
                st.session_state.fast_analysis = response.content
            except Exception as e:
                st.error(f"❌ API Error: Unable to generate analysis. Please check your API key or try again later.\n\nDetails: {str(e)}")
                st.session_state.fast_analysis = None
    
    if st.session_state.fast_analysis:
        st.write("### Fast AI Analysis")
        st.write(st.session_state.fast_analysis)

st.sidebar.header("📡 Market Scanner")

list_type = st.sidebar.radio("Select Source", ["My Watchlist", "S&P 500 Leaders", "Crypto Majors"])

if list_type == "S&P 500 Leaders":
    tickers = get_sp500_tickers()
elif list_type == "Crypto Majors":
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
else:
    tickers = ["NVDA", "TSLA", "AMD", "PLTR", "JNJ", "KO"]

# The selection box
selected_ticker = st.sidebar.selectbox("Pick Asset", tickers)

if st.sidebar.button("Load Ticker"):
    st.session_state.selected_ticker = selected_ticker
    st.rerun()


