"""
Stock Price Downloader Module

This module provides a clean, reusable class for downloading and analyzing
stock price data from Yahoo Finance.
"""
import lseg.data as ld
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, Any


class StockPriceDownloader:
    """
    A class to download and analyze stock price time series data from Yahoo Finance.
    
    This class follows clean code principles with clear separation of concerns,
    single responsibility, and comprehensive documentation.
    
    Attributes:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        data (pd.DataFrame): Downloaded stock data
        closing_prices (pd.Series): Series of closing prices
    """
    
    def __init__(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: Optional[str] = None,
        interval: Optional[str] = "1D"
    ):
        """
        Initialize the StockPriceDownloader.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format. Defaults to today if None.
            interval: Time interval for data (e.g., '1D', '1W', '1M')
        
        Raises:
            ValueError: If ticker is empty or dates are in invalid format
        """
        self._validate_inputs(ticker, start_date, end_date)
        
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.interval_period = interval
        self.data: Optional[pd.DataFrame] = None
        self.closing_prices: Optional[pd.Series] = None
    
    def __str__(self):
        return f"{self.ticker}"

    def get_session(self):
        try:
            ld.open_session()
        except Exception as e:
            print(f"Error opening session: {e}")
        return ld  
    
    def _validate_inputs(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: Optional[str]
    ) -> None:
        """
        Validate input parameters.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string or None
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker symbol cannot be empty")
        
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Use 'YYYY-MM-DD'. Error: {e}"
            )
    
    def download_data(self) -> pd.DataFrame:
        """
        Download historical stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Downloaded stock data with OHLCV columns
            
        Raises:
            Exception: If download fails
        """
        try:
            print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
            #ld = self.get_session()

            ##self.data = ld.get_history(
            ##    universe=[self.ticker], 
            ##    fields=['TR.PriceClose', 'TR.PriceOpen', 'TR.PriceHigh', 'TR.PriceLow', 'TR.Volume'], 
            ##    interval=self.interval_period,
            ##    start = self.start_date, 
            ##    end = self.end_date) \
            ##    .rename(columns={
            ##        'Price Close': 'close', 
            ##        'Price Open': 'open', 
            ##        'Price High': 'high',
            ##        'Price Low': 'low'
            ##    })

            ##

            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date).rename(columns={
                    'Close': 'close', 
                    'Open': 'open', 
                    'High': 'high',
                    'Low': 'low'
                })
            
            if self.data.empty:
                raise ValueError(
                    f"No data found for ticker '{self.ticker}'. "
                    "Please verify the ticker symbol is correct."
                )
            
            self.closing_prices = self.data['close']
            print(f"Successfully downloaded {len(self.data)} records")
            return self.data
            
        except Exception as e:
            raise Exception(f"Failed to download data: {e}")
    
    def get_closing_prices(self) -> pd.Series:
        """
        Get the closing prices series.
        
        Returns:
            pd.Series: Series of closing prices
            
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        if self.closing_prices is None:
            raise RuntimeError(
                "Data not downloaded yet. Call download_data() first."
            )
        return self.closing_prices

    def get_returns(self):

        """
        Get the daily returns series.
        
        Returns:
            pd.Series: Series of daily returns
            
        Raises:
            RuntimeError: If prices hasn't been downloaded 
        """
        prices = self.get_prices()
        if prices is not None:
            try:
                prices['returns'] = np.log(prices[f"{self.RIC}"].div(prices[f"{self.RIC}"].shift(1)))
                returns = prices.drop([f"{self.RIC}"], axis=1).rename(columns={'returns': f"{self.RIC}"})
                return returns
            except Exception as e:
                print(f"Error calculating returns for {self.RIC}: {e}")
                return None
        else:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the closing prices.
        
        Returns:
            dict: Dictionary containing statistical measures
            
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        if self.closing_prices is None:
            raise RuntimeError(
                "Data not downloaded yet. Call download_data() first."
            )
        
        first_price = self.closing_prices.iloc[0]
        last_price = self.closing_prices.iloc[-1]
        price_change_pct = ((last_price - first_price) / first_price) * 100
        
        return {
            'ticker': self.ticker,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_records': len(self.closing_prices),
            'max_price': float(self.closing_prices.max()),
            'max_price_date': self.closing_prices.idxmax().strftime('%Y-%m-%d'),
            'min_price': float(self.closing_prices.min()),
            'min_price_date': self.closing_prices.idxmin().strftime('%Y-%m-%d'),
            'mean_price': float(self.closing_prices.mean()),
            'std_dev': float(self.closing_prices.std()),
            'current_price': float(last_price),
            'current_date': self.closing_prices.index[-1].strftime('%Y-%m-%d'),
            'initial_price': float(first_price),
            'price_change_percent': float(price_change_pct)
        }
    
    def print_statistics(self) -> None:
        """
        Print formatted statistics to console.
        
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print(f"STATISTICS FOR {stats['ticker']}")
        print("=" * 60)
        print(f"\nPeriod: {stats['start_date']} to {stats['end_date']}")
        print(f"Total records: {stats['total_records']}")
        print(f"\nPrice Range:")
        print(f"  Maximum: ${stats['max_price']:.2f} on {stats['max_price_date']}")
        print(f"  Minimum: ${stats['min_price']:.2f} on {stats['min_price_date']}")
        print(f"  Average: ${stats['mean_price']:.2f}")
        print(f"  Std Dev: ${stats['std_dev']:.2f}")
        print(f"\nCurrent Status:")
        print(f"  Initial Price: ${stats['initial_price']:.2f}")
        print(f"  Current Price: ${stats['current_price']:.2f} ({stats['current_date']})")
        print(f"  Total Change: {stats['price_change_percent']:+.2f}%")
        print("=" * 60 + "\n")
    

    def get_yearly_analysis(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze price performance by year.
        
        Returns:
            dict: Dictionary with year as key and statistics as value
            
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        if self.closing_prices is None:
            raise RuntimeError(
                "Data not downloaded yet. Call download_data() first."
            )
        
        yearly_stats = {}
        
        for year in self.closing_prices.index.year.unique():
            year_data = self.closing_prices[self.closing_prices.index.year == year]
            first_price = year_data.iloc[0]
            last_price = year_data.iloc[-1]
            year_change = ((last_price - first_price) / first_price) * 100
            
            yearly_stats[int(year)] = {
                'initial_price': float(first_price),
                'final_price': float(last_price),
                'change_percent': float(year_change),
                'max_price': float(year_data.max()),
                'min_price': float(year_data.min()),
                'mean_price': float(year_data.mean())
            }
        
        return yearly_stats
    
    def print_yearly_analysis(self) -> None:
        """
        Print formatted yearly analysis to console.
        
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        yearly_stats = self.get_yearly_analysis()
        
        print("\n" + "=" * 60)
        print("YEARLY ANALYSIS")
        print("=" * 60)
        
        for year, stats in sorted(yearly_stats.items()):
            print(f"\n{year}:")
            print(f"  Initial: ${stats['initial_price']:.2f}")
            print(f"  Final: ${stats['final_price']:.2f}")
            print(f"  Change: {stats['change_percent']:+.2f}%")
            print(f"  Max: ${stats['max_price']:.2f}")
            print(f"  Min: ${stats['min_price']:.2f}")
            print(f"  Average: ${stats['mean_price']:.2f}")
        
        print("=" * 60 + "\n")
    
    def plot_closing_prices(
        self, 
        figsize: tuple = (16, 8),
        show_average: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot closing prices with optional average line.
        
        Args:
            figsize: Figure size as (width, height) tuple
            show_average: Whether to show average price line
            save_path: Optional path to save the plot. If None, displays plot.
            
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        if self.closing_prices is None:
            raise RuntimeError(
                "Data not downloaded yet. Call download_data() first."
            )
        
        plt.figure(figsize=figsize)
        plt.plot(
            self.closing_prices.index, 
            self.closing_prices.values, 
            linewidth=2, 
            color='#007AFF', 
            label='Closing Price'
        )
        
        if show_average:
            avg_price = self.closing_prices.mean()
            plt.axhline(
                y=avg_price, 
                color='red', 
                linestyle='--', 
                linewidth=1.5, 
                alpha=0.7, 
                label=f'Average: ${avg_price:.2f}'
            )
        
        plt.title(
            f'{self.ticker} Closing Prices ({self.start_date} to {self.end_date})',
            fontsize=18,
            fontweight='bold',
            pad=20
        )
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        plt.ylabel('Closing Price (USD)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def export_to_csv(
        self, 
        filepath: str, 
        export_all_columns: bool = False
    ) -> None:
        """
        Export data to CSV file.
        
        Args:
            filepath: Path where CSV file will be saved
            export_all_columns: If True, exports all OHLCV data. 
                              If False, exports only closing prices.
                              
        Raises:
            RuntimeError: If data hasn't been downloaded yet
        """
        if self.data is None:
            raise RuntimeError(
                "Data not downloaded yet. Call download_data() first."
            )
        
        if export_all_columns:
            self.data.to_csv(filepath)
            print(f"Full data exported to {filepath}")
        else:
            self.closing_prices.to_csv(filepath, header=['close'])
            print(f"Closing prices exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Example 1: Download Apple stock data
    apple = StockPriceDownloader('AAPL', '2022-01-01')
    apple.download_data()
    apple.print_statistics()
    apple.print_yearly_analysis()
    apple.plot_closing_prices()
    
    # Example 2: Download Google stock data with custom end date
    google = StockPriceDownloader('GOOGL', '2024-01-01', '2024-12-31')
    google.download_data()
    stats = google.get_statistics()
    print(f"Google average price in 2024: ${stats['mean_price']:.2f}")
    
    # Example 3: Export to CSV
    # apple.export_to_csv('apple_closing_prices.csv')
    # apple.export_to_csv('apple_full_data.csv', export_all_columns=True)
