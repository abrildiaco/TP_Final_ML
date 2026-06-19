import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eda_utils import missing_values_summary, unique_values_summary


FORMAL_COLORS = {
    "blue": "#1F4E79",
    "teal": "#2A9D8F",
    "gold": "#77547E",
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
    """
    Plots the number of listings by currency.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        currency_col (str): currency column name
        title (str): plot title
    """

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


def plot_unique_values(data, top_n=None, title="Unique values by column"):
    """
    Plots the number of unique values per column.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        top_n (int | None): number of columns to show
        title (str): plot title
    """

    unique_table = unique_values_summary(data)

    if top_n is not None:
        unique_table = unique_table.head(top_n)

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.35 * len(unique_table)))
    )

    bars = ax.barh(
        unique_table["column"],
        unique_table["unique_values"],
        color=FORMAL_COLORS["red"]
    )

    ax.invert_yaxis()

    for bar, value in zip(bars, unique_table["unique_values"]):
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


def plot_categorical_counts(df, categorical_columns=None, ignored_columns=None, top_n=10, n_cols=2, figsize_per_plot=(7, 4),):
    """
    Plots category counts for categorical features in a single figure.

    Arguments:
        df (pd.DataFrame): dataset containing categorical features
        categorical_columns (list[str] | None): categorical columns to plot
        ignored_columns (list[str] | None): columns to exclude from the plot
        top_n (int): maximum number of categories shown per feature
        n_cols (int): number of subplot columns in the figure
        figsize_per_plot (tuple[int, int]): size multiplier for each subplot

    Returns:
        tuple[plt.Figure, np.ndarray]: figure and axes used for the plots
    """
    ignored_columns = ignored_columns or []

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    categorical_columns = [
        column for column in categorical_columns
        if column not in ignored_columns and not column.startswith("Unnamed:")
    ]

    n_plots = len(categorical_columns)

    if n_plots == 0:
        raise ValueError("No categorical columns found to plot.")

    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        constrained_layout=True,
    )

    axes = pd.Series(axes.flatten())

    for ax, column in zip(axes, categorical_columns):
        counts = df[column].fillna("Missing").astype(str).value_counts()

        # Keep the figure readable when a feature has many categories.
        if len(counts) > top_n:
            top_counts = counts.head(top_n)
            other_count = counts.iloc[top_n:].sum()
            counts = pd.concat([
                top_counts,
                pd.Series({"Other": other_count})
            ])

        counts = counts.sort_values()

        ax.barh(counts.index, counts.values, color=FORMAL_COLORS["gold"])
        ax.set_title(column)
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")

        # Add count labels next to each bar.
        for index, value in enumerate(counts.values):
            ax.text(value, index, f" {value}", va="center")

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(f"Categorical Feature Counts - Top {top_n}", fontsize=16, fontweight="bold")

    return


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


def plot_camera_missing_by_year(df, year_col="Año", camera_col="Con cámara de retroceso", title="Nulos de cámara de retroceso por año",):

    """
    Plots missing camera information by vehicle year.

    Arguments:
        df (pd.DataFrame): dataset to analyze
        year_col (str): year column name
        camera_col (str): rear camera column name
        title (str): plot title

    Returns:
        pd.DataFrame: missing summary by year
    """
    plot_data = df[[year_col, camera_col]].copy()
    plot_data[year_col] = pd.to_numeric(plot_data[year_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[year_col])
    plot_data[year_col] = plot_data[year_col].astype(int)
    plot_data["camera_missing"] = plot_data[camera_col].isna()

    summary = (
        plot_data
        .groupby(year_col)
        .agg(
            total=(camera_col, "size"),
            missing=("camera_missing", "sum"),
            missing_pct=("camera_missing", "mean"),
        )
        .reset_index()
        .sort_values(year_col)
    )
    summary["missing_pct"] = summary["missing_pct"] * 100

    fig, ax_count = plt.subplots(figsize=(12, 5))

    ax_count.bar(
        summary[year_col],
        summary["total"],
        color=FORMAL_COLORS["light_gray"],
        edgecolor=FORMAL_COLORS["gray"],
        label="Cantidad de autos",
    )
    ax_count.set_xlabel(year_col)
    ax_count.set_ylabel("Cantidad de autos")

    ax_missing = ax_count.twinx()
    ax_missing.plot(
        summary[year_col],
        summary["missing_pct"],
        color=FORMAL_COLORS["red"],
        marker="o",
        linewidth=2,
        label="% nulos",
    )
    ax_missing.set_ylabel("% nulos en cámara de retroceso")
    ax_missing.set_ylim(0, max(100, summary["missing_pct"].max() * 1.1))

    lines_1, labels_1 = ax_count.get_legend_handles_labels()
    lines_2, labels_2 = ax_missing.get_legend_handles_labels()
    ax_count.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax_count.set_title(title, fontweight="bold")
    ax_count.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return summary

 
def plot_compact_value_counts(df, columns, top_n=10, n_cols=2):
    """ Plots compact horizontal bar charts for categorical value counts """
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    bar_color = FORMAL_COLORS["teal"]

    for ax, column in zip(axes, columns):
        counts = df[column].value_counts(dropna=False).head(top_n)
        counts = counts.sort_values()

        bars = ax.barh(counts.index.astype(str), counts.values, color=bar_color, alpha=0.9)

        max_value = counts.values.max()
        ax.set_xlim(0, max_value * 1.15)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max_value * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va="center",
                ha="left",
                fontsize=9
            )

        ax.set_title(column, fontweight="bold", fontsize=12)
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(columns):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()