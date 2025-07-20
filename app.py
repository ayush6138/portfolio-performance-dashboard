import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyxirr
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Performance Dashboard")

@st.cache_data
def fetch_prices(tickers, start_date=None, end_date=None, period=None):
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
    elif period:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=False)['Adj Close']
    else:
        raise ValueError("Provide either 'start_date'/'end_date' or 'period'.")
    return data.to_frame() if isinstance(data, pd.Series) else data

def append_prices_to_df(df):
    min_purchase_date = df['Purchase Date'].min()
    all_tickers = df.index.tolist()
    price_data = yf.download(all_tickers, start=min_purchase_date - pd.Timedelta(days=5),
                              end=datetime.today() + pd.Timedelta(days=1),
                              progress=False, auto_adjust=False)['Adj Close']
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()
    df['Purchase Price'] = None
    df['Current Price'] = None
    for ticker in df.index:
        purchase_date = df.loc[ticker, 'Purchase Date']
        ticker_prices = price_data[ticker].loc[purchase_date:]
        df.loc[ticker, 'Purchase Price'] = ticker_prices.iloc[0] if not ticker_prices.empty else np.nan
        df.loc[ticker, 'Current Price'] = price_data[ticker].iloc[-1] if not price_data[ticker].empty else np.nan
    return df

def calculate_portfolio_metrics(df):
    df['Current Holding Value'] = df['No. of Shares'] * df['Current Price']
    df['Invested Holding Value'] = df['No. of Shares'] * df['Purchase Price']
    current_value = df['Current Holding Value'].sum()
    invested_value = df['Invested Holding Value'].sum()
    gain_loss = current_value - invested_value
    return current_value, invested_value, gain_loss

def calculate_xirr(df, current_value):
    xirr_df = df.copy()
    xirr_df['Cash Flow'] = -(xirr_df['No. of Shares'] * xirr_df['Purchase Price'])
    today = datetime.today()
    current_row = pd.DataFrame([{'Purchase Date': today, 'Cash Flow': current_value}])
    transactions = pd.concat([xirr_df[['Purchase Date', 'Cash Flow']], current_row], ignore_index=True).dropna()
    if len(transactions) < 2:
        return np.nan
    try:
        return pyxirr.xirr(transactions['Purchase Date'], transactions['Cash Flow'])
    except:
        return np.nan

def compute_additional_metrics(portfolio_returns):
    annual_return = np.mean(portfolio_returns) * 252
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else np.nan
    downside_std = np.std(portfolio_returns[portfolio_returns < 0]) * np.sqrt(252)
    sortino = annual_return / downside_std if downside_std > 0 else np.nan
    var_95 = np.percentile(portfolio_returns, 5)
    return sharpe, sortino, var_95

def plot_portfolio_pie_chart(df):
    df_clean = df.dropna(subset=['Current Price', 'No. of Shares'])
    if df_clean.empty:
        st.warning("No data available for pie chart.")
        return
    values = df_clean['Current Holding Value']
    labels = df_clean.index
    fig = px.pie(values=values, names=labels, title="Portfolio Allocation by Market Value", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.header("Portfolio Input")
num_stocks = st.sidebar.number_input("Number of Stocks", min_value=1, max_value=20, value=3)

portfolio_data = {}
for i in range(num_stocks):
    st.sidebar.subheader(f"Stock {i+1}")
    ticker = st.sidebar.text_input(f"Ticker {i+1}", value="RELIANCE.NS", key=f"ticker_{i}")
    shares = st.sidebar.number_input(f"No. of Shares {i+1}", min_value=1, value=10, key=f"shares_{i}")
    purchase_date = st.sidebar.date_input(f"Purchase Date {i+1}", value=datetime.today(), key=f"date_{i}")
    portfolio_data[ticker] = {"No. of Shares": shares, "Purchase Date": pd.to_datetime(purchase_date)}

if st.sidebar.button("Analyze Portfolio"):
    with st.spinner("Fetching data and calculating..."):
        stock_df = pd.DataFrame.from_dict(portfolio_data, orient='index')
        stock_df.index.name = "Tickers"
        stock_df = append_prices_to_df(stock_df)
        current_value, invested_value, gain_loss = calculate_portfolio_metrics(stock_df)
        xirr = calculate_xirr(stock_df, current_value)

        overall_start_date = stock_df['Purchase Date'].min()
        today = datetime.today()
        tickers = stock_df.index.tolist()
        price_history = fetch_prices(tickers, start_date=overall_start_date, end_date=today)
        price_history.index = pd.to_datetime(price_history.index.date)
        shares_over_time = pd.DataFrame(0, index=price_history.index, columns=tickers)
        for ticker in tickers:
            purchase_date = pd.to_datetime(stock_df.loc[ticker, 'Purchase Date'].date())
            shares = stock_df.loc[ticker, 'No. of Shares']
            shares_over_time.loc[shares_over_time.index >= purchase_date, ticker] = shares
        daily_values = price_history.multiply(shares_over_time).replace(0, np.nan)
        portfolio_value = daily_values.sum(axis=1).dropna()
        returns = price_history.pct_change().fillna(0)
        weights = daily_values.divide(portfolio_value, axis=0).fillna(0)
        portfolio_returns = (returns * weights).sum(axis=1).fillna(0)
        sharpe, sortino, var_95 = compute_additional_metrics(portfolio_returns)

        st.subheader("Portfolio Metrics")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("Current Value", f"{current_value:,.2f}")
        col2.metric("Invested Value", f"{invested_value:,.2f}")
        col3.metric("Gain/Loss", f"{gain_loss:,.2f}")
        col4.metric("XIRR", f"{xirr*100:.2f}%" if not np.isnan(xirr) else "N/A")
        col5.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col6.metric("Sortino Ratio", f"{sortino:.2f}")
        col7.metric("VaR (95%)", f"{var_95*100:.2f}%")

        nifty_df = fetch_prices("^NSEI", start_date=overall_start_date, end_date=today)
        nifty_df.columns = ['NIFTY']
        nifty_df.index = pd.to_datetime(nifty_df.index.date)
        nifty_returns = nifty_df['NIFTY'].pct_change().fillna(0)
        comparison_df = pd.DataFrame({
            'Portfolio': 100 * (1 + portfolio_returns).cumprod(),
            'NIFTY 50': 100 * (1 + nifty_returns).cumprod()
        })

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Portfolio'], mode='lines', name='Portfolio'))
        fig_line.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['NIFTY 50'], mode='lines', name='NIFTY 50'))
        fig_line.update_layout(title="Portfolio vs NIFTY 50 (Indexed to 100)", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Rolling Sharpe Analysis")
        rolling_mean = portfolio_returns.rolling(21).mean()
        rolling_std = portfolio_returns.rolling(21).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        fig_roll = px.line(rolling_sharpe, title="Rolling 21-Day Sharpe Ratio", labels={"value": "Sharpe Ratio", "index": "Date"})
        st.plotly_chart(fig_roll, use_container_width=True)

        st.subheader("Volatility vs Return Scatter")
        rolling_vol = portfolio_returns.rolling(21).std() * np.sqrt(252)
        scatter_df = pd.DataFrame({'Volatility': rolling_vol, 'Return': rolling_mean}).dropna()
        fig_scatter = px.scatter(scatter_df, x='Volatility', y='Return', trendline='ols', title="Volatility vs Rolling Return")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Portfolio Allocation")
        plot_portfolio_pie_chart(stock_df)

        st.subheader("Current Holdings")
        st.dataframe(stock_df)
