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



st.title("Financial Risk Metrics Dashboard")

# Use selected_ticker from sidebar if available
default_ticker = st.session_state.selected_ticker if st.session_state.selected_ticker else ""
ticker = st.text_input("Enter Stock Ticker", value=default_ticker, placeholder="e.g., AAPL, TSLA, MSFT")



if ticker:
    # Only recalculate if ticker changed
    if st.session_state.cached_ticker != ticker:
        # Quick validation first - check if ticker exists
        test_data = yf.download(ticker, period="1d", progress=False)
        if test_data.empty:
            st.error(f"❌ Invalid ticker '{ticker}'. Please enter a valid stock symbol.")
            st.session_state.cached_ticker = None
            st.session_state.cached_metrics = None
            st.stop()
        
        with st.spinner("Fetching data and calculating metrics..."):
            try:
                yearly_stock_data = yf.download(ticker, period="1y", progress=False)

                returns= np.log(yearly_stock_data['Close']/yearly_stock_data['Close'].shift(1))

                var_cutoff = np.percentile(returns.dropna(), 5)
                var = 100 * abs(var_cutoff)

                bad_days = returns[returns <= var_cutoff].dropna()
                cvar = 100 * abs(np.mean(bad_days))

                daily_volatility = float(np.std(returns) * 100)
                yearly_volatility = daily_volatility * np.sqrt(252)
                
                # Calculate drawdown
                close_prices = yearly_stock_data['Close']
                cumulative_max = close_prices.cummax()
                drawdown = (close_prices - cumulative_max) / cumulative_max * 100
                
                # Cache the results
                st.session_state.cached_ticker = ticker
                st.session_state.cached_metrics = {
                    'yearly_volatility': yearly_volatility,
                    'var': var,
                    'cvar': cvar,
                    'drawdown': drawdown,
                    'max_drawdown': float(drawdown.min())
                }
                # Clear old analyses when ticker changes
                st.session_state.deep_analysis = None
                st.session_state.fast_analysis = None
                
            except Exception as e:
                st.error(f"❌ Error fetching data for '{ticker}': {str(e)}")
                st.session_state.cached_ticker = None
                st.session_state.cached_metrics = None
                st.stop()
    
    # Check if we have valid cached metrics
    if st.session_state.cached_metrics is None:
        st.warning("⚠️ No data available. Please enter a valid ticker.")
        st.stop()
    
    # Use cached metrics
    metrics = st.session_state.cached_metrics
    yearly_volatility = metrics['yearly_volatility']
    var = metrics['var']
    cvar = metrics['cvar']
    drawdown = metrics['drawdown']
    max_drawdown = metrics['max_drawdown'] 


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


