import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eda_utils import missing_values_summary


FORMAL_COLORS = {
    "blue": "#1F4E79",
    "teal": "#2A9D8F",
    "gold": "#C99700",
    "red": "#A23E48",
    "gray": "#6C757D",
    "light_gray": "#E9ECEF"
}


def plot_missing_values(data, top_n=None, title="Valores faltantes por columna"):
    """
    Plots the percentage of missing values by column.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        top_n (int | None): number of columns to show
        title (str): plot title

    Returns:
        None
    """

    missing_table = missing_values_summary(data)

    if top_n is not None:
        missing_table = missing_table.head(top_n)

    if missing_table.empty:
        print("No hay valores faltantes")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(missing_table))))

    bars = ax.barh(
        missing_table["column"],
        missing_table["missing_percentage"],
        color=FORMAL_COLORS["blue"]
    )

    ax.invert_yaxis()

    for bar, value in zip(bars, missing_table["missing_percentage"]):
        ax.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Porcentaje de valores faltantes")
    ax.set_ylabel("Columna")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_currency_counts(data, currency_col="Moneda", title="Cantidad de publicaciones por moneda"):
    """ Plots the number of listings by currency """

    currency_counts = data[currency_col].value_counts(dropna=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(currency_counts.index.astype(str), currency_counts.values, color=FORMAL_COLORS["teal"])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Moneda")
    ax.set_ylabel("Cantidad de publicaciones")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return currency_counts


def plot_price_distribution_by_currency(data, price_col="Precio", currency_col="Moneda", bins=40, title="Distribución de precio por moneda"):
    """
    Plots price distributions separately by currency.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        price_col (str): price column name
        currency_col (str): currency column name
        bins (int): number of histogram bins
        title (str): general plot title

    Returns:
        None
    """
    plot_data = data[[price_col, currency_col]].copy()
    plot_data[price_col] = pd.to_numeric(plot_data[price_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[price_col, currency_col])

    currencies = plot_data[currency_col].unique()
    n_currencies = len(currencies)

    fig, axes = plt.subplots(1, n_currencies, figsize=(6 * n_currencies, 4.5))

    if n_currencies == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, currency in zip(axes, currencies):
        prices = plot_data.loc[plot_data[currency_col] == currency, price_col]

        ax.hist(prices, bins=bins, color=FORMAL_COLORS["blue"], edgecolor="white", alpha=0.85)
        ax.axvline(prices.median(), color=FORMAL_COLORS["gold"], linestyle="--", linewidth=2, label=f"Mediana: {prices.median():.0f}")

        ax.set_title(f"Moneda: {currency}", fontweight="bold")
        ax.set_xlabel("Precio")
        ax.set_ylabel("Cantidad de publicaciones")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


def plot_raw_numeric_distributions(data, numeric_cols=("Año", "Puertas", "Kilómetros", "Precio"), bins=35, title="Distribución raw de variables numéricas", use_percentile_range=True):
    """
    Plots raw distributions for selected numeric variables.
    If use_percentile_range is True, the plot filters extreme values only for visualization.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        numeric_cols (tuple | list): numeric columns to plot
        bins (int): number of histogram bins
        title (str): general plot title
        use_percentile_range (bool): whether to use percentiles only for visualization

    Returns:
        None
    """
    available_cols = [col for col in numeric_cols if col in data.columns]

    n_cols = 2
    n_rows = math.ceil(len(available_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.2 * n_rows))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, available_cols):
        values = pd.to_numeric(data[col], errors="coerce").dropna()

        if use_percentile_range:
            lower_limit = values.quantile(0.01)
            upper_limit = values.quantile(0.99)

            # Filters only for plotting so extreme values do not collapse the histogram
            plot_values = values[(values >= lower_limit) & (values <= upper_limit)]
        else:
            plot_values = values.copy()

        if col in ["Año", "Puertas"]:
            min_value = int(np.floor(plot_values.min()))
            max_value = int(np.ceil(plot_values.max()))
            col_bins = np.arange(min_value, max_value + 2) - 0.5
        else:
            col_bins = bins

        ax.hist(plot_values, bins=col_bins, color=FORMAL_COLORS["blue"], edgecolor="white", alpha=0.85)
        ax.axvline(values.median(), color=FORMAL_COLORS["gold"], linestyle="--", linewidth=2, label=f"Mediana: {values.median():.0f}")

        ax.set_title(col, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    for ax in axes[len(available_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_preliminary_outliers(data, numeric_cols=("Precio", "Año", "Kilómetros"), currency_col="Moneda", title="Outliers preliminares"):
    """
    Plots preliminary boxplots for selected numeric variables before preprocessing.
    If price and currency columns are available, price is shown separately by currency.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        numeric_cols (tuple | list): numeric columns to plot
        currency_col (str): currency column name
        title (str): general plot title

    Returns:
        None
    """
    available_cols = [col for col in numeric_cols if col in data.columns]

    n_cols = len(available_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.8))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, available_cols):
        plot_data = data[[col]].copy()
        plot_data[col] = pd.to_numeric(plot_data[col], errors="coerce")

        if col == "Precio" and currency_col in data.columns:
            plot_data[currency_col] = data[currency_col]
            plot_data = plot_data.dropna(subset=[col, currency_col])

            groups = []
            labels = []

            for currency, group in plot_data.groupby(currency_col):
                groups.append(group[col].dropna())
                labels.append(str(currency))

            ax.boxplot(groups, labels=labels, patch_artist=True, boxprops=dict(facecolor=FORMAL_COLORS["light_gray"], color=FORMAL_COLORS["blue"]), medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2))
            ax.set_xlabel("Moneda")

        else:
            values = plot_data[col].dropna()
            ax.boxplot(values, patch_artist=True, boxprops=dict(facecolor=FORMAL_COLORS["light_gray"], color=FORMAL_COLORS["blue"]), medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2))

        ax.set_title(col, fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


def plot_unique_values(data, top_n=None, title="Unique values by column"):
    """
    Plots the number of unique values per column.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        top_n (int | None): number of columns to show
        title (str): plot title

    Returns:
        pd.DataFrame: table with unique counts and percentages
    """

    unique_count = data.nunique(dropna=True)
    unique_percentage = unique_count / len(data) * 100

    unique_table = pd.DataFrame({
        "column": unique_count.index,
        "unique_count": unique_count.values,
        "unique_percentage": unique_percentage.values
    })

    unique_table = unique_table.sort_values(
        "unique_count",
        ascending=False
    )

    if top_n is not None:
        unique_table = unique_table.head(top_n)

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.35 * len(unique_table)))
    )

    bars = ax.barh(
        unique_table["column"],
        unique_table["unique_count"],
        color=FORMAL_COLORS["teal"]
    )

    ax.invert_yaxis()

    for bar, value in zip(bars, unique_table["unique_count"]):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {int(value)}",
            va="center",
            fontsize=9
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of unique values")
    ax.set_ylabel("Column")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return unique_table

