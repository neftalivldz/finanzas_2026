"""
Multi-Stock Downloader Module

Provides a class to download closing prices for multiple tickers,
returning a single DataFrame with dates as the index and each ticker
as a column.
"""

import pandas as pd
from typing import List, Optional

from stock_price_downloader import StockPriceDownloader


class MultiStockDownloader:
    """
    Downloads closing prices for a list of stock tickers.

    Uses StockPriceDownloader under the hood for each ticker and
    combines the results into a single DataFrame.

    Attributes:
        tickers (List[str]): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (Optional[str]): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g. '1D', '1W', '1M').
        closing_prices (Optional[pd.DataFrame]): Combined closing-price
            DataFrame; populated after calling ``download()``.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: Optional[str] = "1D",
    ):
        """
        Initialize the MultiStockDownloader.

        Args:
            tickers: List of stock ticker symbols, e.g. ['AAPL', 'AMZN', 'NVDA'].
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format. Defaults to today.
            interval: Time interval for data (e.g. '1D', '1W', '1M').

        Raises:
            ValueError: If the tickers list is empty.
        """
        if not tickers:
            raise ValueError("The tickers list cannot be empty.")

        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.closing_prices: Optional[pd.DataFrame] = None

    def download(self) -> pd.DataFrame:
        """
        Download closing prices for all tickers.

        Iterates over each ticker, downloads its historical data via
        StockPriceDownloader, and assembles a combined DataFrame whose
        index is the date and whose columns are the ticker symbols.

        Returns:
            pd.DataFrame: DataFrame with date as index and one column
                per ticker containing closing prices.

        Raises:
            Exception: If downloading fails for any ticker.
        """
        series_list: List[pd.Series] = []

        for ticker in self.tickers:
            downloader = StockPriceDownloader(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval,
            )
            downloader.download_data()
            closing = downloader.get_closing_prices().rename(ticker)
            series_list.append(closing)

        self.closing_prices = pd.concat(series_list, axis=1)
        self.closing_prices.index.name = "Date"
        return self.closing_prices

    def get_closing_prices(self) -> pd.DataFrame:
        """
        Return the combined closing-price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with date as index and one column
                per ticker.

        Raises:
            RuntimeError: If ``download()`` has not been called yet.
        """
        if self.closing_prices is None:
            raise RuntimeError(
                "Data not downloaded yet. Call download() first."
            )
        return self.closing_prices


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "NVDA"]

    downloader = MultiStockDownloader(
        tickers=tickers,
        start_date="2023-01-01",
        end_date="2024-01-01",
    )
    df = downloader.download()
    print(df.head())
