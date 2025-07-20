# portfolio-performance-dashboard
Streamlit-based interactive dashboard for analyzing stock portfolio performance with metrics like XIRR, Sharpe Ratio, Sortino Ratio, Value at Risk, and comparisons against NIFTY 50.
# Portfolio Performance Dashboard

## Overview

The Portfolio Performance Dashboard is a Python-based interactive web application built using Streamlit, designed to help investors analyze and visualize the performance of their stock portfolios. It fetches real-time data from Yahoo Finance and computes essential financial metrics, tracks portfolio allocations, compares performance against the NIFTY 50 index, and provides multiple interactive visualizations.

This tool is suitable for both individual investors and financial analysts seeking to monitor and evaluate portfolio performance efficiently.

## Features

- Real-Time Portfolio Valuation: Retrieves live stock prices from Yahoo Finance and calculates current holdings' value.
- Performance Metrics:
  - XIRR (Extended Internal Rate of Return)
  - Sharpe Ratio
  - Sortino Ratio
  - Value at Risk (95% VaR)
- Benchmark Comparison:
  - Compare your portfolio's performance against the NIFTY 50 index.
- Risk Analysis Tools:
  - Rolling Sharpe Ratio visualization.
  - Volatility vs. Rolling Return scatter plots.
- Portfolio Allocation Visualization:
  - Interactive pie chart showing allocation by market value.
- Interactive Portfolio Input:
  - Define portfolio details via an intuitive sidebar interface.
- Data Presentation:
  - Clean, interactive tables displaying current holdings and calculated values.

## Technologies Used

- Streamlit for the interactive web interface.
- Yahoo Finance API (`yfinance`) for real-time financial data retrieval.
- Pandas and NumPy for data manipulation and analysis.
- PyXIRR for internal rate of return (XIRR) calculation.
- Plotly for interactive data visualizations.
- Matplotlib (optional) for static plotting.

## Setup Instructions

1. Clone the Repository:
   ```bash
   git clone https://github.com/your-username/portfolio-performance-dashboard.git
   cd portfolio-performance-dashboard
