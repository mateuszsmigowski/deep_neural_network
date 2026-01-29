import yfinance as yf


class FinancialData:

    def __init__(self):

        self.data = None

    def get_financial_data(self, ticker, start_date=None, end_date=None, interval="1d", period=None):

        if period:
            df = yf.download(ticker, period=period, interval=interval)
        else:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        self.data = df["Close"].values.reshape(-1, 1)
        return self.data
