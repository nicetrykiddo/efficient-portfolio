import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, expected_returns, risk_models, plotting
from scipy.optimize import minimize
from datetime import datetime
import os


class PortfolioOptimizer:

    def __init__(self, tickers, total_investment, risk_tolerance, constraints=None, leverage=1.0, risk_free_rate=0.02, custom_returns=None):
        self.tickers = tickers
        self.total_investment = total_investment
        self.risk_tolerance = risk_tolerance
        self.constraints = constraints if constraints else {}
        self.leverage = leverage
        self.risk_free_rate = risk_free_rate
        self.custom_returns = custom_returns
        self.historical_data = self.fetch_historical_data(tickers)
        self.returns = self.historical_returns()
        self.cov_matrix = self.covariance_matrix()
        self.mean_returns_ = self.mean_returns()

    def fetch_historical_data(self, tickers):
        try:
            data = yf.download(tickers, start="2018-01-01", end="2023-01-01")['Adj Close']
        except Exception as e:
            print(f"Error fetching data: {e}")
            data = pd.DataFrame()
        return data.dropna()

    def historical_returns(self):
        if self.historical_data.empty:
            raise ValueError("No valid historical data found for the provided tickers.")
        return self.historical_data.pct_change().mean() * 252

    def covariance_matrix(self):
        return self.historical_data.pct_change().cov() * 252

    def mean_returns(self):
        if self.custom_returns is not None:
            self.custom_returns = np.maximum(self.custom_returns, 0)
            return self.custom_returns
        else:
            return self.returns

    def portfolio_risk(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def portfolio_return(self, weights):
        return np.dot(weights, self.mean_returns_)

    def calculate_metrics(self, weights):
        portfolio_return = self.portfolio_return(weights)
        portfolio_risk = self.portfolio_risk(weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk  # Sharpe ratio formula
        return portfolio_return, portfolio_risk, sharpe_ratio

    def optimize_portfolio(self):
        num_assets = len(self.tickers)
        bounds = [(0, self.constraints.get('max_position_limit', 1.0) / 100)] * num_assets
        initial_weights = [1.0 / num_assets] * num_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        return_constraint = {'type': 'ineq', 'fun': lambda w: np.dot(w, self.mean_returns_) - 0.01}

        result = minimize(
            lambda w: -self.calculate_metrics(w)[2],
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints + [return_constraint]
        )
        if not result.success:
            raise ValueError("Optimization failed.")
        return result.x
    
    def min_volatility_portfolio(self):
        num_assets = len(self.tickers)
        bounds = [(0, self.constraints.get('max_position_limit', 1.0) / 100)] * num_assets
        initial_weights = [1.0 / num_assets] * num_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(
            lambda w: self.portfolio_risk(w),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if not result.success:
            raise ValueError("Optimization failed.")
        return result.x

    def plot_efficient_frontier(self):
        target_returns = np.linspace(0.05, 0.3, 50)
        results = []
        num_assets = len(self.mean_returns_)
        bounds = [(0, 1)] * num_assets

        for target_return in target_returns:
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                           {'type': 'eq', 'fun': lambda w: np.dot(w, self.mean_returns_) - target_return}]
            result = minimize(
                lambda w: self.portfolio_risk(w),
                [1.0 / num_assets] * num_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            if result.success:
                portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(self.cov_matrix, result.x)))
                results.append((portfolio_volatility, target_return))

        risks, returns = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(risks, returns, label="Efficient Frontier", color="blue")
        plt.xlabel("Portfolio Risk (Volatility)")
        plt.ylabel("Portfolio Return")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def summary(self):
        print(f"Portfolio for {', '.join(self.tickers)}")
        print("Mean Annual Returns:\n", self.mean_returns_)
        print("\nCovariance Matrix:\n", self.cov_matrix)
        equal_weights = np.array([1.0 / len(self.tickers)] * len(self.tickers))
        print("\nPortfolio Risk (Equal Weights):", self.portfolio_risk(equal_weights))
        print("Portfolio Return (Equal Weights):", self.portfolio_return(equal_weights))
        print("\nSharpe Ratio (Equal Weights):", (self.portfolio_return(equal_weights) - self.risk_free_rate) / self.portfolio_risk(equal_weights))


def main():
    os.system("cls" if os.name == "nt" else "clear")

    print("Choose a Portfolio Objective:")
    print("1. Retirement Planning")
    print("2. Mutual Fund Creation")
    print("3. Hedge Fund Creation")
    print("4. Inflation Protection")
    user_choice = input("Enter your choice (1-4): ").strip()

    try:
        if user_choice not in ["1", "2", "3", "4"]:
            raise ValueError("Invalid choice. Please select between 1 and 4.")

        total_investment = float(input("Enter your total investment amount: "))
        risk_tolerance = input("Enter your risk tolerance level (Low, Medium, High): ").strip().lower()
        ticker_choice = input("Do you want to input your own asset tickers? (Yes/No): ").strip().lower()

        default_tickers = {
            "1": ["AAPL", "MSFT", "JNJ", "PG", "V"],
            "2": ["SPY", "VOO", "QQQ", "DIA", "IWM"],
            "3": ["GOOGL", "AMZN", "TSLA", "NVDA", "META"],
            "4": ["TIP", "GLD", "SLV", "VNQ", "PDBC"],
        }

        if ticker_choice == "yes":
            user_tickers = input("Enter asset tickers (comma-separated): ").strip().split(",")
        else:
            user_tickers = default_tickers.get(user_choice, [])

        if not user_tickers:
            raise ValueError("No valid tickers provided.")
        
        custom_returns = input("Enter custom expected returns as a comma-separated list (or leave empty): ").strip()

        if custom_returns:
            custom_returns = np.array([float(x) for x in custom_returns.split(",")])
        else:
            custom_returns = None

        max_position_limit = float(input("Enter maximum position limit per asset (%): ") or 100)
        transaction_cost = float(input("Enter expected transaction costs (%): ") or 0.1)
        leverage = 1.0
        if user_choice == "3":
            leverage = float(input("Enter leverage level (default = 1.0): ") or 1.0)

        constraints = {
            "max_position_limit": max_position_limit,
            "transaction_cost": transaction_cost,
        }

        optimizer = PortfolioOptimizer(
            tickers=user_tickers,
            total_investment=total_investment,
            risk_tolerance=risk_tolerance,
            constraints=constraints,
            leverage=leverage,
            custom_returns=custom_returns
        )

        print("\nFetching historical data and calculating portfolio metrics...")
        optimizer.summary()

        max_sharpe_weights = optimizer.optimize_portfolio()
        min_volatility_weights = optimizer.min_volatility_portfolio()

        max_sharpe_return, max_sharpe_risk, max_sharpe_ratio = optimizer.calculate_metrics(max_sharpe_weights)
        min_volatility_return, min_volatility_risk, _ = optimizer.calculate_metrics(min_volatility_weights)

        print("\nOptimized Portfolio Weights:")
        for ticker, weight in zip(user_tickers, max_sharpe_weights):
            print(f"{ticker}: {weight:.2%}")

        print("\nMaximum Sharpe Ratio Portfolio:")
        print(f"Optimal Weights: {max_sharpe_weights}")
        print(f"Expected Return: {max_sharpe_return}")
        print(f"Expected Risk: {max_sharpe_risk}")
        print(f"Sharpe Ratio: {max_sharpe_ratio}")

        print("\nMinimum Volatility Portfolio:")
        print(f"Optimal Weights: {min_volatility_weights}")
        print(f"Expected Return: {min_volatility_return}")
        print(f"Expected Risk: {min_volatility_risk}")

        optimizer.plot_efficient_frontier()

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

