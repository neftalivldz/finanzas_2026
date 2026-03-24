"""
backtesting.py
==============
Clase principal de backtesting de portafolios.

Uso rápido
----------
    from backtesting import PortfolioBacktester

    bt = PortfolioBacktester(
        portfolio_path="portfolio.json",
        start_date="2023-01-03",
    )
    pv = bt.run()          # pd.DataFrame con portfolio_value
    bt.plot()              # gráfico Portfolio vs S&P 500
    df = bt.get_full_dataframe()  # DataFrame con precios, portfolio y S&P 500
"""

from __future__ import annotations

import json
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from multi_stock_downloader import MultiStockDownloader


# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------
_SP500_TICKER = "^GSPC"
_SP500_COLUMN = "SP500"
_PORTFOLIO_KEY = "portfolio"
_DATE_FORMAT = "%Y-%m-%d"


class PortfolioBacktester:
    """
    Ejecuta un backtest de buy-and-hold para un portafolio definido en JSON.

    El backtester:
    1. Lee el portafolio (tickers + monto invertido por ticker) desde un JSON.
    2. Descarga los precios históricos de cierre desde ``start_date``.
    3. Calcula cuántas acciones se pueden comprar con cada monto en ``start_date``.
    4. Evalúa el valor del portafolio en cada fecha posterior.
    5. Descarga el S&P 500 para comparación.
    6. Ofrece un gráfico normalizado a base 100.

    Attributes:
        portfolio_path (str): Ruta al archivo JSON del portafolio.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (Optional[str]): Fecha de fin; None = hoy.
        portfolio_value (Optional[pd.DataFrame]): Resultado de ``run()``.
        comparison (Optional[pd.DataFrame]): DataFrame con Portfolio y SP500.
        full_df (Optional[pd.DataFrame]): DataFrame completo con precios por
            acción, valor del portafolio y S&P 500 (disponible tras ``run()``).
    """

    def __init__(
        self,
        portfolio_path: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> None:
        """
        Inicializa el backtester.

        Args:
            portfolio_path: Ruta al JSON (ej. 'portfolio.json').
            start_date:     Fecha de compra en formato 'YYYY-MM-DD'.
            end_date:       Fecha de fin; si es None se usa la fecha de hoy.
        """
        self.portfolio_path = portfolio_path
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio_value: Optional[pd.DataFrame] = None
        self.comparison: Optional[pd.DataFrame] = None
        self.full_df: Optional[pd.DataFrame] = None

        self._tickers: list[str] = []
        self._allocations: dict[str, float] = {}
        self._prices: Optional[pd.DataFrame] = None
        self._shares: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el backtest completo.

        Returns:
            pd.DataFrame con columna ``portfolio_value`` indexada por fecha.
        """
        self._load_portfolio()
        self._download_prices()
        self._compute_shares()
        self._compute_portfolio_value()
        self._download_sp500()
        self._build_comparison()
        self._build_full_dataframe()
        return self.portfolio_value  # type: ignore[return-value]

    def plot(self, figsize: tuple[int, int] = (13, 5)) -> None:
        """
        Genera un gráfico comparando el portafolio con el S&P 500 (base 100).

        Args:
            figsize: Tamaño de la figura en pulgadas (ancho, alto).

        Raises:
            RuntimeError: Si ``run()`` no ha sido llamado antes.
        """
        self._require_run()

        df_norm = self._normalize_to_base_100(self.comparison)  # type: ignore[arg-type]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            df_norm.index,
            df_norm["Portfolio"],
            label="Portfolio",
            linewidth=1.8,
            color="#4C72B0",
        )
        ax.plot(
            df_norm.index,
            df_norm[_SP500_COLUMN],
            label="S&P 500",
            linewidth=1.8,
            color="#DD8452",
            linestyle="--",
        )

        ax.set_title("Portfolio vs S&P 500 (base 100)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Valor (base 100)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_full_dataframe(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con los precios de cada acción del portafolio,
        el valor total del portafolio y el índice S&P 500, alineados por fecha.

        Las columnas son:
        - Una columna por cada ticker (precio de cierre).
        - ``portfolio_value``: valor total del portafolio en cada fecha.
        - ``SP500``: precio de cierre del S&P 500.

        Returns:
            pd.DataFrame indexado por fecha con todas las series anteriores.

        Raises:
            RuntimeError: Si ``run()`` no ha sido llamado antes.
        """
        self._require_run()
        return self.full_df.copy()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Pasos internos del pipeline
    # ------------------------------------------------------------------

    def _load_portfolio(self) -> None:
        """Lee el JSON y extrae tickers y montos de inversión."""
        with open(self.portfolio_path, "r") as f:
            data = json.load(f)

        self._allocations = data[_PORTFOLIO_KEY]
        self._tickers = list(self._allocations.keys())

    def _download_prices(self) -> None:
        """Descarga precios de cierre para todos los tickers del portafolio."""
        downloader = MultiStockDownloader(
            tickers=self._tickers,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self._prices = downloader.download()

    def _compute_shares(self) -> None:
        """
        Calcula el número entero de acciones que se pueden comprar
        con el monto asignado, al precio de cierre de ``start_date``.
        """
        prices_at_start = self._prices.loc[self.start_date, self._tickers]
        allocations_series = pd.Series(self._allocations, dtype=float)

        shares_series = (allocations_series // prices_at_start).astype(int)
        self._shares = pd.DataFrame(
            shares_series,
            columns=[self.start_date],
        ).T
        self._shares.index = pd.DatetimeIndex([self.start_date], name="Date")

    def _compute_portfolio_value(self) -> None:
        """
        Calcula el valor del portafolio día a día desde ``start_date``.

        Guarda en ``self.portfolio_value`` un DataFrame con columna
        ``portfolio_value`` que refleja el valor total en cada fecha.
        """
        prices = self._prices.loc[self.start_date:, self._tickers].copy()

        shares_row = self._shares.iloc[0]
        shares_broadcast = pd.DataFrame(
            {ric: shares_row[ric] for ric in self._tickers},
            index=prices.index,
        )

        value_per_ticker = prices * shares_broadcast
        self.portfolio_value = pd.DataFrame(
            {"portfolio_value": value_per_ticker.sum(axis=1)}
        )

    def _download_sp500(self) -> None:
        """Descarga precios de cierre del S&P 500 desde ``start_date``."""
        ticker = yf.Ticker(_SP500_TICKER)
        df = ticker.history(start=self.start_date, end=self.end_date)[["Close"]]
        df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df.rename(columns={"Close": _SP500_COLUMN}, inplace=True)
        self._sp500: pd.DataFrame = df

    def _build_comparison(self) -> None:
        """Une portfolio_value con S&P 500 en un único DataFrame."""
        pv_flat = self.portfolio_value.copy()  # type: ignore[union-attr]

        self.comparison = pv_flat.rename(
            columns={"portfolio_value": "Portfolio"}
        ).join(self._sp500[[_SP500_COLUMN]], how="inner")

    def _build_full_dataframe(self) -> None:
        """
        Construye ``self.full_df``: precios individuales + portfolio_value + SP500,
        alineados por fecha desde ``start_date``.
        """
        prices = self._prices.loc[self.start_date:, self._tickers].copy()  # type: ignore[index]
        full = prices.join(self.portfolio_value, how="inner")  # type: ignore[arg-type]
        full = full.join(self._sp500[[_SP500_COLUMN]], how="inner")
        self.full_df = full

    # ------------------------------------------------------------------
    # Utilidades privadas
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_base_100(df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza todas las columnas del DataFrame a base 100."""
        return df / df.iloc[0] * 100

    def _require_run(self) -> None:
        """Lanza RuntimeError si run() no se ha ejecutado aún."""
        if self.portfolio_value is None or self.comparison is None:
            raise RuntimeError(
                "Debes llamar a run() antes de usar este método."
            )


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bt = PortfolioBacktester(
        portfolio_path="portfolio.json",
        start_date="2023-01-03",
    )
    pv = bt.run()
    print(pv.head())
    print(f"\nValor final del portafolio: ${pv['portfolio_value'].iloc[-1]:,.2f}")
    bt.plot()
