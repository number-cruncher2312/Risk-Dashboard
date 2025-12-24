# Financial Risk Assessment Dashboard

## Overview
This application is a real-time risk analysis tool developed using Python and Streamlit. It integrates quantitative financial modeling with Large Language Model (LLM) inference to provide a dual-layer risk assessment for equity and cryptocurrency assets.

The system calculates standard risk metrics—Value at Risk (VaR) and Conditional Value at Risk (CVaR)—while simultaneously leveraging NVIDIA's NIM endpoints to generate qualitative risk reports based on current market volatility and asset-specific data.

## Key Features

*   **Real-Time Market Data Integration**: Automated fetching of OHLCV data via the Yahoo Finance API (yfinance), supporting dynamic ticker resolution for both equities and digital assets.
*   **Quantitative Risk Metrics**:
    *   **Annualized Volatility**: Statistical measure of the dispersion of returns.
    *   **Value at Risk (VaR 95%)**: Estimation of the maximum potential loss over a specified time frame with 95% confidence.
    *   **Conditional Value at Risk (CVaR 95%)**: Assessment of the expected loss exceeding the VaR threshold (tail risk).
*   **Generative AI Analysis**: Integration with NVIDIA NIM (LangChain) to produce contextual risk narratives, interpreting numerical indicators for non-technical stakeholders.
*   **Automated Ticker Standardization**: Logic to handle API-specific formatting discrepancies (e.g., standardized suffix handling for cryptocurrency pairs).

## Technical Architecture

*   **Frontend**: Streamlit
*   **Data Layer**: yfinance (Pandas/NumPy for calculation)
*   **Inference Engine**: LangChain (NVIDIA AI Endpoints)
*   **Language**: Python 3.13

## Installation and Setup

### Prerequisites
*   Python 3.10 or higher
*   Git
*   NVIDIA API Key (for LLM inference)

### Local Development

1.  **Clone the repository**
    ```
    git clone https://github.com/your-username/risk-dashboard.git
    cd risk-dashboard
    ```

2.  **Initialize virtual environment**
    ```
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/MacOS
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4.  **Configuration**
    Create a `.env` file in the project root directory to store sensitive credentials:
    ```
    NVIDIA_API_KEY=nvapi-your_key_here
    ```

5.  **Execution**
    ```
    streamlit run app.py
    ```

## Deployment Configuration

This application is optimized for Streamlit Cloud deployment.

**Note on Secrets Management:**
The `.env` file is excluded from version control for security. When deploying to a remote environment, API keys must be injected via the platform's secrets management system.

**Streamlit Cloud Secrets (TOML format):**
