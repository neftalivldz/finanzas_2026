
# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional

def fill_null_with_previous_price(
    df: pd.DataFrame,
    column: str,
    inplace: bool = False,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reemplaza los valores nulos de una columna con el precio del período anterior.

    Utiliza forward fill (ffill): propaga el último valor válido hacia adelante
    para cubrir los NaN. Útil cuando faltan precios en fechas sin mercado
    (fines de semana, festivos, datos corruptos, etc.).

    Args:
        df:      DataFrame con el índice de fechas y la columna de precios.
        column:  Nombre de la columna a rellenar.
        inplace: Si True, modifica el DataFrame original; si False, devuelve
                 una copia. Por defecto False.
        limit:   Número máximo de períodos consecutivos a rellenar. Si None,
                 rellena todos los NaN hacia adelante.

    Returns:
        pd.DataFrame con los nulos de ``column`` reemplazados por el valor anterior.

    Raises:
        KeyError: Si ``column`` no existe en el DataFrame.

    Example:
        >>> prices = pd.DataFrame(
        ...     {"AAPL": [150.0, None, None, 155.0, None]},
        ...     index=pd.date_range("2024-01-01", periods=5),
        ... )
        >>> fill_null_with_previous_price(prices, "AAPL")
                     AAPL
        2024-01-01  150.0
        2024-01-02  150.0
        2024-01-03  150.0
        2024-01-04  155.0
        2024-01-05  155.0
    """
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")

    result = df if inplace else df.copy()
    result[column] = result[column].ffill(limit=limit)
    return result


def compute_log_returns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula los retornos logarítmicos de cada columna en *df*.

    El retorno logarítmico en el período *t* se define como:

        r_t = ln(P_t / P_{t-1})

    Args:
        df:      DataFrame con precios (filas = fechas, columnas = activos).
                 Puede contener precios de acciones, valor del portafolio,
                 índices bursátiles, etc.
        columns: Lista de columnas a procesar.  Si es None se procesan
                 todas las columnas numéricas del DataFrame.

    Returns:
        pd.DataFrame con los mismos índices y columnas seleccionadas,
        expresados como retornos logarítmicos.  La primera fila (NaN)
        se elimina automáticamente.

    Example:
        >>> from utils import compute_log_returns
        >>> log_ret = compute_log_returns(full_df)
        >>> log_ret = compute_log_returns(full_df, columns=["AAPL", "SP500"])
    """
    subset = df[columns] if columns is not None else df.select_dtypes(include="number")
    log_returns = np.log(subset / subset.shift(1))
    return log_returns.dropna()


def plot_pearson_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str = "Correlación de Pearson",
    cmap: str = "RdYlGn",
) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de Pearson de *df* y la visualiza
    como un heatmap anotado.

    Se recomienda pasar retornos logarítmicos (salida de ``compute_log_returns``)
    en lugar de precios para que la correlación sea significativa.

    Args:
        df:      DataFrame con las series a correlacionar (filas = fechas).
        columns: Lista opcional de columnas a incluir. Si es None se usan
                 todas las columnas numéricas.
        figsize: Tamaño de la figura en pulgadas (ancho, alto).
        title:   Título del gráfico.
        cmap:    Paleta de color de seaborn/matplotlib.
                 'RdYlGn'  → rojo (negativo) – verde (positivo).
                 'coolwarm' es otra opción popular.

    Returns:
        pd.DataFrame con la matriz de correlación (valores entre -1 y 1).

    Example:
        >>> from utils import compute_log_returns, plot_pearson_heatmap
        >>> log_ret = compute_log_returns(full_df)
        >>> corr = plot_pearson_heatmap(log_ret)
        >>> corr = plot_pearson_heatmap(log_ret, columns=["AAPL", "AMZN", "SP500"])
    """
    subset = df[columns] if columns is not None else df.select_dtypes(include="number")
    corr = subset.corr(method="pearson")

    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    for i in range(len(mask)):
        for j in range(i):
            mask.iloc[i, j] = True  # oculta el triángulo inferior

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return corr


def plot_roi_bars(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (13, 6),
    title: str = "ROI por instrumento (%)",
    portfolio_col: str = "portfolio_value",
    sp500_col: str = "SP500",
) -> pd.Series:
    """
    Calcula el ROI porcentual de cada columna en *df* y lo visualiza
    como una gráfica de barras horizontal ordenada, con S&P 500 y el
    portafolio siempre al frente de la comparación.

    El ROI se calcula como:

        ROI (%) = (precio_final - precio_inicial) / precio_inicial × 100

    donde precio_inicial es el primer valor no-nulo de cada columna y
    precio_final el último.

    Args:
        df:            DataFrame con precios (tipicamente ``get_full_dataframe()``).
                       Todas las columnas numéricas se procesan.
        figsize:       Tamaño de la figura en pulgadas (ancho, alto).
        title:         Título de la gráfica.
        portfolio_col: Nombre de la columna que representa el portafolio.
        sp500_col:     Nombre de la columna que representa el S&P 500.

    Returns:
        pd.Series con el ROI (%) de cada columna, ordenado de mayor a menor.

    Example:
        >>> from utils import plot_roi_bars
        >>> roi = plot_roi_bars(bt.get_full_dataframe())
        >>> print(roi)
    """
    numeric = df.select_dtypes(include="number")
    roi = (numeric.iloc[-1] - numeric.iloc[0]) / numeric.iloc[0] * 100

    # Ordenar: SP500 y portfolio primero, luego el resto de mayor a menor
    priority = [c for c in [sp500_col, portfolio_col] if c in roi.index]
    rest = roi.drop(labels=priority).sort_values(ascending=False)
    roi_sorted = pd.concat([roi[priority], rest])

    colors = ["#4C72B0" if c in priority else ("#2ca02c" if v >= 0 else "#d62728")
              for c, v in roi_sorted.items()]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(roi_sorted.index, roi_sorted.values, color=colors, edgecolor="white", linewidth=0.6)

    # Línea de referencia en 0
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Separador visual entre los índices de referencia y los activos
    if priority:
        ax.axvline(len(priority) - 0.5, color="gray", linewidth=1, linestyle=":", alpha=0.7)

    # Anotar valores encima/debajo de cada barra
    for bar, val in zip(bars, roi_sorted.values):
        offset = 0.5 if val >= 0 else -1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("ROI (%)")
    ax.set_xlabel("Instrumento")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return roi_sorted

def plot_sortino_bars(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    figsize: tuple[int, int] = (13, 6),
    title: str = "Sortino Ratio por instrumento",
    portfolio_col: str = "portfolio_value",
    sp500_col: str = "SP500",
) -> pd.Series:
    """
    Calcula el Sortino Ratio anualizado para cada columna de *df* y lo
    visualiza como una gráfica de barras, con el portafolio y el S&P 500
    siempre en primer lugar.

    El Sortino Ratio se define como:

        Sortino = (R_media_anual - R_libre_riesgo) / Downside_Deviation_anual

    donde Downside Deviation solo considera los retornos negativos:

        DD = sqrt(mean(min(r_t, 0)^2)) * sqrt(trading_days)

    Un Sortino > 1 se considera aceptable; > 2 es excelente.
    Valores negativos indican rendimiento por debajo de la tasa libre de riesgo.

    Args:
        df:              DataFrame con precios (resultado de ``get_full_dataframe()``).
        risk_free_rate:  Tasa libre de riesgo ANUAL en decimal (ej. 0.04 = 4%).
                         Por defecto 0.0.
        trading_days:    Días de trading por año para anualización. Default 252.
        figsize:         Tamaño de la figura en pulgadas (ancho, alto).
        title:           Título del gráfico.
        portfolio_col:   Nombre de la columna del portafolio.
        sp500_col:       Nombre de la columna del S&P 500.

    Returns:
        pd.Series con el Sortino Ratio de cada columna, en el mismo orden
        que aparece en la gráfica (portafolio + SP500 primero, resto por valor).

    Example:
        >>> from utils import plot_sortino_bars
        >>> sortino = plot_sortino_bars(bt.get_full_dataframe())
        >>> sortino = plot_sortino_bars(bt.get_full_dataframe(), risk_free_rate=0.04)
    """
    numeric = df.select_dtypes(include="number")

    # Retornos logarítmicos diarios
    log_ret = np.log(numeric / numeric.shift(1)).dropna()

    # Tasa libre de riesgo diaria
    rf_daily = risk_free_rate / trading_days

    sortino_values = {}
    for col in log_ret.columns:
        r = log_ret[col]
        excess = r - rf_daily
        mean_annual = excess.mean() * trading_days

        # Downside: solo retornos por debajo de cero (o de rf)
        downside = excess[excess < 0]
        dd = np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days)

        sortino_values[col] = mean_annual / dd if dd != 0 else np.nan

    sortino = pd.Series(sortino_values)

    # Ordenar: portfolio y sp500 primero, luego el resto de mayor a menor
    priority = [c for c in [sp500_col, portfolio_col] if c in sortino.index]
    rest = sortino.drop(labels=priority).sort_values(ascending=False)
    sortino_sorted = pd.concat([sortino[priority], rest])

    colors = [
        "#4C72B0" if c in priority else ("#2ca02c" if v >= 0 else "#d62728")
        for c, v in sortino_sorted.items()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        sortino_sorted.index,
        sortino_sorted.values,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    # Líneas de referencia
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="Sortino = 1 (umbral)")
    ax.axhline(2, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.6, label="Sortino = 2 (excelente)")

    # Separador visual entre referencia y activos
    if priority:
        ax.axvline(len(priority) - 0.5, color="gray", linewidth=1, linestyle=":", alpha=0.7)

    # Anotaciones sobre cada barra
    for bar, val in zip(bars, sortino_sorted.values):
        if np.isnan(val):
            continue
        offset = 0.03 if val >= 0 else -0.06
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Sortino Ratio (anualizado)")
    ax.set_xlabel("Instrumento")
    ax.legend(fontsize=8, loc="upper right")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return sortino_sorted
