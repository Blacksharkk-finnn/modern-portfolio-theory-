# 📈 Modern Portfolio Theory (MPT) — Portfolio Optimization in Python

### 🧠 Overview
This project applies **Modern Portfolio Theory (MPT)** to construct and optimize a portfolio of financial assets using **Python**.  
It demonstrates how investors can **maximize returns** for a given level of risk (or minimize risk for a target return) through diversification.

---

## 🧩 Objectives
- Compute **daily log returns** for multiple assets  
- Calculate the **covariance matrix** and **expected returns**  
- Simulate **random portfolios** and compute their risk-return profiles  
- Identify the **efficient frontier**  
- Find the **optimal portfolio** with the **maximum Sharpe ratio**

---

## 📊 Theory — Modern Portfolio Theory (MPT)

### ⚙️ 1. Expected Return
\[
E(R_p) = \sum_{i=1}^{n} w_i E(R_i)
\]

Where:  
- \( E(R_p) \) = Expected return of the portfolio  
- \( w_i \) = Weight of asset *i*  
- \( E(R_i) \) = Expected return of asset *i*

---

### ⚙️ 2. Portfolio Variance
\[
\sigma_p^2 = w^T \Sigma w
\]

Where:  
- \( \Sigma \) = Covariance matrix of asset returns  
- \( w \) = Vector of asset weights

---

### ⚙️ 3. Portfolio Volatility (Risk)
\[
\sigma_p = \sqrt{w^T \Sigma w}
\]

---

### ⚙️ 4. Sharpe Ratio
\[
S = \frac{E(R_p) - R_f}{\sigma_p}
\]

Where:  
- \( R_f \) = Risk-free rate  
- Higher Sharpe Ratio ⇒ Better risk-adjusted performance

---

## 🧮 Implementation Steps

### 1️⃣ Import Libraries
```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')['Adj Close']
log_returns = np.log(data / data.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252
expected_returns = log_returns.mean() * 252
num_portfolios = 50000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio
plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.show()
pip install numpy pandas yfinance matplotlib
📁 Portfolio Optimization Using MPT/
├── MPT.py
├── requirements.txt
├── README.md
└── data/





