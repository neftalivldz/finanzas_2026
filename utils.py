
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