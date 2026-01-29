import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self):
        pass

    def add_sma(self, data, window=14):
        series = pd.Series(data.flatten())
        sma = series.rolling(window=window).mean()
        return sma.values.reshape(-1, 1)

    def add_rsi(self, data, window=14):
        series = pd.Series(data.flatten())
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values.reshape(-1, 1)

    def create_dataset(self, prices, sma_window=14, rsi_window=14):
        prices = np.array(prices).reshape(-1, 1)
        
        sma = self.add_sma(prices, sma_window)
        rsi = self.add_rsi(prices, rsi_window)
        
        dataset = np.hstack((prices, sma, rsi))
        
        dataset = dataset[~np.isnan(dataset).any(axis=1)]
        
        return dataset

    def add_returns(self, prices):
        series = pd.Series(prices.flatten())
        returns = series.pct_change()
        return returns.values.reshape(-1, 1)

    def create_returns_dataset(self, prices, sma_window=5, rsi_window=14):
        prices = np.array(prices).reshape(-1, 1)
        
        returns = self.add_returns(prices)
        
        sma_returns = pd.Series(returns.flatten()).rolling(window=sma_window).mean().values.reshape(-1, 1)
        
        rsi = self.add_rsi(prices, rsi_window)
        
        dataset = np.hstack((returns, sma_returns, rsi))
        
        dataset = dataset[~np.isnan(dataset).any(axis=1)]
        
        return dataset

