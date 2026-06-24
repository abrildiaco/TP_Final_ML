import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_utils import missing_values_summary, unique_values_summary


# ========================= Plot Style =========================

FORMAL_COLORS = {
    "blue": "#1F4E79",
    "teal": "#2A9D8F",
    "gold": "#77547E",
    "red": "#A23E48",
    "gray": "#6C757D",
    "light_gray": "#E9ECEF",
}


# ========================= Missing and Unique Values =========================

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
        color=FORMAL_COLORS["blue"],
    )

    ax.invert_yaxis()

    # Add percentage labels next to each bar
    for bar, value in zip(bars, missing_table["missing_percentage"]):
        ax.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Porcentaje de valores faltantes")
    ax.set_ylabel("Columna")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_unique_values(data, top_n=None, title="Unique values by column"):
    """
    Plots the number of unique values per column.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        top_n (int | None): number of columns to show
        title (str): plot title

    Returns:
        None
    """
    unique_table = unique_values_summary(data)

    if top_n is not None:
        unique_table = unique_table.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(unique_table))))

    bars = ax.barh(
        unique_table["column"],
        unique_table["unique_values"],
        color=FORMAL_COLORS["red"],
    )

    ax.invert_yaxis()

    # Add count labels next to each bar
    for bar, value in zip(bars, unique_table["unique_values"]):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {int(value)}",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of unique values")
    ax.set_ylabel("Column")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()


# ========================= Currency Plots =========================

def plot_currency_counts(data, currency_col="Moneda", title="Cantidad de publicaciones por moneda"):
    """
    Plots the number of listings by currency.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        currency_col (str): currency column name
        title (str): plot title

    Returns:
        None
    """
    currency_counts = data[currency_col].value_counts(dropna=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        currency_counts.index.astype(str),
        currency_counts.values,
        color=FORMAL_COLORS["teal"],
    )

    # Add absolute count labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Moneda")
    ax.set_ylabel("Cantidad de publicaciones")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


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

        ax.hist(
            prices,
            bins=bins,
            color=FORMAL_COLORS["blue"],
            edgecolor="white",
            alpha=0.85,
        )

        ax.axvline(
            prices.median(),
            color=FORMAL_COLORS["gold"],
            linestyle="--",
            linewidth=2,
            label=f"Mediana: {prices.median():.0f}",
        )

        ax.set_title(f"Moneda: {currency}", fontweight="bold")
        ax.set_xlabel("Precio")
        ax.set_ylabel("Cantidad de publicaciones")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


# ========================= Categorical Plots =========================

def plot_categorical_counts(df, categorical_columns=None, ignored_columns=None, top_n=10, n_cols=2, figsize_per_plot=(7, 4)):
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
        None
    """
    ignored_columns = ignored_columns or []

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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

    axes = pd.Series(np.asarray(axes).flatten())

    for ax, column in zip(axes, categorical_columns):
        counts = df[column].fillna("Missing").astype(str).value_counts()

        # Group low-frequency categories into Other to keep the plot readable
        if len(counts) > top_n:
            top_counts = counts.head(top_n)
            other_count = counts.iloc[top_n:].sum()
            counts = pd.concat([top_counts, pd.Series({"Other": other_count})])

        counts = counts.sort_values()

        ax.barh(counts.index, counts.values, color=FORMAL_COLORS["gold"])
        ax.set_title(column)
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")

        # Add count labels next to each bar
        for index, value in enumerate(counts.values):
            ax.text(value, index, f" {value}", va="center")

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(f"Categorical Feature Counts - Top {top_n}", fontsize=16, fontweight="bold")

    plt.show()


def plot_compact_value_counts(df, columns, top_n=10, n_cols=2):
    """
    Plots compact horizontal bar charts for categorical value counts.

    Arguments:
        df (pd.DataFrame): dataset containing categorical columns
        columns (list[str]): columns to plot
        top_n (int): maximum number of categories shown per column
        n_cols (int): number of subplot columns in the figure

    Returns:
        None
    """
    n_rows = math.ceil(len(columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = np.asarray(axes).flatten()

    bar_color = FORMAL_COLORS["teal"]

    for ax, column in zip(axes, columns):
        counts = df[column].value_counts(dropna=False).head(top_n)
        counts = counts.sort_values()

        bars = ax.barh(
            counts.index.astype(str),
            counts.values,
            color=bar_color,
            alpha=0.9,
        )

        max_value = counts.values.max()
        ax.set_xlim(0, max_value * 1.15)

        # Add count labels with extra space to avoid overlap
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max_value * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va="center",
                ha="left",
                fontsize=9,
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


# ========================= Numeric Distributions =========================

def plot_raw_numeric_distributions(data, numeric_cols=("Año", "Puertas", "Kilómetros", "Precio"), bins=35, title="Distribución raw de variables numéricas", use_percentile_range=True):
    """
    Plots raw distributions for selected numeric variables.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        numeric_cols (tuple | list): numeric columns to plot
        bins (int): number of histogram bins
        title (str): general plot title
        use_percentile_range (bool): whether to use percentiles only for visualization

    Returns:
        None
    """
    available_cols = [column for column in numeric_cols if column in data.columns]

    n_cols = 2
    n_rows = math.ceil(len(available_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.2 * n_rows))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes = np.asarray(axes).reshape(-1)

    for ax, column in zip(axes, available_cols):
        values = pd.to_numeric(data[column], errors="coerce").dropna()

        if use_percentile_range:
            lower_limit = values.quantile(0.01)
            upper_limit = values.quantile(0.99)

            # Filter only for plotting so extreme values do not collapse the histogram
            plot_values = values[(values >= lower_limit) & (values <= upper_limit)]
        else:
            plot_values = values.copy()

        if column in ["Año", "Puertas"]:
            min_value = int(np.floor(plot_values.min()))
            max_value = int(np.ceil(plot_values.max()))
            column_bins = np.arange(min_value, max_value + 2) - 0.5
        else:
            column_bins = bins

        ax.hist(
            plot_values,
            bins=column_bins,
            color=FORMAL_COLORS["blue"],
            edgecolor="white",
            alpha=0.85,
        )

        ax.axvline(
            values.median(),
            color=FORMAL_COLORS["gold"],
            linestyle="--",
            linewidth=2,
            label=f"Mediana: {values.median():.0f}",
        )

        ax.set_title(column, fontweight="bold")
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    for ax in axes[len(available_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


# ========================= Outlier Plots =========================

def plot_preliminary_outliers(data, numeric_cols=("Precio", "Año", "Kilómetros"), currency_col="Moneda", title="Outliers preliminares"):
    """
    Plots preliminary boxplots for selected numeric variables before preprocessing.

    Arguments:
        data (pd.DataFrame): dataset to analyze
        numeric_cols (tuple | list): numeric columns to plot
        currency_col (str): currency column name
        title (str): general plot title

    Returns:
        None
    """
    available_cols = [column for column in numeric_cols if column in data.columns]

    n_cols = len(available_cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.8))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    if n_cols == 1:
        axes = [axes]

    for ax, column in zip(axes, available_cols):
        plot_data = data[[column]].copy()
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")

        if column == "Precio" and currency_col in data.columns:
            plot_data[currency_col] = data[currency_col]
            plot_data = plot_data.dropna(subset=[column, currency_col])

            groups = []
            labels = []

            for currency, group in plot_data.groupby(currency_col):
                groups.append(group[column].dropna())
                labels.append(str(currency))

            ax.boxplot(
                groups,
                labels=labels,
                patch_artist=True,
                boxprops=dict(facecolor=FORMAL_COLORS["light_gray"], color=FORMAL_COLORS["blue"]),
                medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2),
            )

            ax.set_xlabel("Moneda")

        else:
            values = plot_data[column].dropna()

            ax.boxplot(
                values,
                patch_artist=True,
                boxprops=dict(facecolor=FORMAL_COLORS["light_gray"], color=FORMAL_COLORS["blue"]),
                medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2),
            )

        ax.set_title(column, fontweight="bold")
        ax.set_ylabel(column)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


# ========================= Missingness Analysis =========================


# ========================= Post-Preprocessing EDA =========================

def _numeric_plot_data(data, columns):
    """
    Builds a numeric-only copy for plotting selected columns.

    Arguments:
        data (pd.DataFrame): dataset containing the selected columns
        columns (tuple | list): columns to convert to numeric

    Returns:
        pd.DataFrame: numeric data with unavailable columns ignored
    """
    available_cols = [column for column in columns if column in data.columns]
    plot_data = data[available_cols].copy()

    for column in available_cols:
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")

    return plot_data


def _filter_frequent_categories(data, category_col, min_count=30, top_n=None):
    """
    Keeps categories with enough observations for readable category plots.

    Arguments:
        data (pd.DataFrame): dataset containing the category column
        category_col (str): categorical column used to filter rows
        min_count (int): minimum number of rows required per category
        top_n (int | None): maximum number of most frequent categories to keep

    Returns:
        list[str]: category labels that pass the frequency filter
    """
    counts = data[category_col].fillna("missing").astype(str).value_counts()
    counts = counts[counts >= min_count]

    if top_n is not None:
        counts = counts.head(top_n)

    return counts.index.tolist()


def _category_price_data(data, category_col, price_col="Precio", min_count=30, top_n=None):
    """
    Prepares category and price data for boxplots and median bar charts.

    Arguments:
        data (pd.DataFrame): dataset containing category and price columns
        category_col (str): categorical feature to analyze
        price_col (str): target price column
        min_count (int): minimum number of rows required per category
        top_n (int | None): maximum number of frequent categories to keep

    Returns:
        pd.DataFrame: filtered data with category and price columns
    """
    plot_data = data[[category_col, price_col]].copy()
    plot_data[category_col] = plot_data[category_col].fillna("missing").astype(str)
    plot_data[price_col] = pd.to_numeric(plot_data[price_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[price_col])

    selected_categories = _filter_frequent_categories(
        plot_data,
        category_col=category_col,
        min_count=min_count,
        top_n=top_n,
    )

    return plot_data[plot_data[category_col].isin(selected_categories)].copy()


def _add_horizontal_bar_labels(ax, values):
    """
    Adds value labels to horizontal bar plots.

    Arguments:
        ax (plt.Axes): plot axes
        values (iterable): values plotted in the bars

    Returns:
        None
    """
    max_value = max(values) if len(values) else 0

    for index, value in enumerate(values):
        ax.text(
            value + max_value * 0.01,
            index,
            f"{value:,.0f}",
            va="center",
            fontsize=9,
        )


def plot_price_distribution(data, price_col="Precio", bins=45, log_transform=False,
                            title=None):
    """
    Plots the distribution of the cleaned price variable.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        price_col (str): price column name
        bins (int): number of histogram bins
        log_transform (bool): whether to plot log1p(price) instead of raw price
        title (str | None): plot title

    Returns:
        None
    """
    prices = pd.to_numeric(data[price_col], errors="coerce").dropna()

    if log_transform:
        prices = np.log1p(prices[prices > 0])
        x_label = f"log({price_col})"
        default_title = f"Distribución de log({price_col})"
    else:
        x_label = price_col
        default_title = f"Distribución de {price_col}"

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.hist(
        prices,
        bins=bins,
        color=FORMAL_COLORS["blue"],
        edgecolor="white",
        alpha=0.85,
    )

    ax.axvline(
        prices.median(),
        color=FORMAL_COLORS["gold"],
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {prices.median():,.0f}",
    )

    ax.set_title(title or default_title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_clean_numeric_distributions(data, numeric_cols=("Año", "Kilómetros", "Puertas"),
                                     bins=35, title="Distribución de variables numéricas limpias"):
    """
    Plots distributions for cleaned numeric features after preprocessing.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        numeric_cols (tuple | list): numeric columns to plot
        bins (int): number of histogram bins
        title (str): general plot title

    Returns:
        None
    """
    plot_data = _numeric_plot_data(data, numeric_cols)
    available_cols = plot_data.columns.tolist()

    if not available_cols:
        raise ValueError("No numeric columns found to plot.")

    n_cols = 2
    n_rows = math.ceil(len(available_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.2 * n_rows))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes = np.asarray(axes).reshape(-1)

    for ax, column in zip(axes, available_cols):
        values = plot_data[column].dropna()

        if column in ["Año", "Puertas", "Grupo cilindrada", "Cilindrada"]:
            min_value = int(np.floor(values.min()))
            max_value = int(np.ceil(values.max()))
            column_bins = np.arange(min_value, max_value + 2) - 0.5
        else:
            column_bins = bins

        ax.hist(
            values,
            bins=column_bins,
            color=FORMAL_COLORS["teal"],
            edgecolor="white",
            alpha=0.85,
        )

        ax.axvline(
            values.median(),
            color=FORMAL_COLORS["gold"],
            linestyle="--",
            linewidth=2,
            label=f"Mediana: {values.median():,.0f}",
        )

        ax.set_title(column, fontweight="bold")
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    for ax in axes[len(available_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_price_vs_numeric(data, x_col, price_col="Precio", sample_size=None,
                          alpha=0.35, title=None):
    """
    Plots price against a numeric feature using a scatter plot.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        x_col (str): numeric feature on the x-axis
        price_col (str): price column name
        sample_size (int | None): optional random sample size for readability
        alpha (float): point transparency
        title (str | None): plot title

    Returns:
        None
    """
    plot_data = _numeric_plot_data(data, [x_col, price_col]).dropna()

    if sample_size is not None and len(plot_data) > sample_size:
        plot_data = plot_data.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        plot_data[x_col],
        plot_data[price_col],
        color=FORMAL_COLORS["blue"],
        alpha=alpha,
        s=18,
    )

    ax.set_title(title or f"{price_col} vs {x_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(price_col)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_year_kilometers_price_scatter(data, year_col="Año", km_col="Kilómetros",
                                       price_col="Precio", sample_size=None,
                                       km_percentile_range=(0.01, 0.99),
                                       price_color_percentile_range=(0.02, 0.98)):
    """
    Plots year vs kilometers and colors points by price.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        year_col (str): vehicle year column
        km_col (str): kilometers column
        price_col (str): price column used as color scale
        sample_size (int | None): optional random sample size for readability
        km_percentile_range (tuple[float, float] | None): percentile range used
            to filter kilometers only for visualization
        price_color_percentile_range (tuple[float, float] | None): percentile range
            used to set the color scale only for visualization

    Returns:
        None
    """
    plot_data = _numeric_plot_data(data, [year_col, km_col, price_col]).dropna()

    if km_percentile_range is not None:
        km_lower = plot_data[km_col].quantile(km_percentile_range[0])
        km_upper = plot_data[km_col].quantile(km_percentile_range[1])

        # Filter only for plotting so extreme kilometer values do not flatten the chart
        plot_data = plot_data[
            plot_data[km_col].between(km_lower, km_upper)
        ].copy()

    if plot_data.empty:
        raise ValueError("No data left to plot after applying the percentile filters.")

    if sample_size is not None and len(plot_data) > sample_size:
        plot_data = plot_data.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if price_color_percentile_range is not None:
        color_min = plot_data[price_col].quantile(price_color_percentile_range[0])
        color_max = plot_data[price_col].quantile(price_color_percentile_range[1])
    else:
        color_min = plot_data[price_col].min()
        color_max = plot_data[price_col].max()

    scatter = ax.scatter(
        plot_data[year_col],
        plot_data[km_col],
        c=plot_data[price_col],
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
        alpha=0.55,
        s=22,
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(price_col)

    ax.set_title(f"{year_col} vs {km_col} coloreado por {price_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(year_col)
    ax.set_ylabel(km_col)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_median_price_by_category(data, category_col, price_col="Precio", top_n=15,
                                  min_count=30, title=None):
    """
    Plots median price by frequent category.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        category_col (str): categorical feature to analyze
        price_col (str): price column name
        top_n (int): maximum number of frequent categories to show
        min_count (int): minimum number of rows required per category
        title (str | None): plot title

    Returns:
        None
    """
    plot_data = _category_price_data(
        data,
        category_col=category_col,
        price_col=price_col,
        min_count=min_count,
        top_n=top_n,
    )

    summary = (
        plot_data
        .groupby(category_col)[price_col]
        .median()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(summary))))

    ax.barh(summary.index, summary.values, color=FORMAL_COLORS["teal"], alpha=0.9)
    _add_horizontal_bar_labels(ax, summary.values)

    ax.set_title(title or f"Precio mediano por {category_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Mediana de {price_col}")
    ax.set_ylabel(category_col)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_price_boxplot_by_category(data, category_col, price_col="Precio", top_n=15,
                                   min_count=30, title=None):
    """
    Plots price dispersion by frequent category using boxplots.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        category_col (str): categorical feature to analyze
        price_col (str): price column name
        top_n (int): maximum number of frequent categories to show
        min_count (int): minimum number of rows required per category
        title (str | None): plot title

    Returns:
        None
    """
    plot_data = _category_price_data(
        data,
        category_col=category_col,
        price_col=price_col,
        min_count=min_count,
        top_n=top_n,
    )

    medians = plot_data.groupby(category_col)[price_col].median().sort_values()
    groups = [
        plot_data.loc[plot_data[category_col] == category, price_col]
        for category in medians.index
    ]

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(groups)), 5.5))

    ax.boxplot(
        groups,
        labels=medians.index,
        patch_artist=True,
        boxprops=dict(facecolor=FORMAL_COLORS["light_gray"], color=FORMAL_COLORS["blue"]),
        medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.25),
    )

    ax.set_title(title or f"Distribución de {price_col} por {category_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(category_col)
    ax.set_ylabel(price_col)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_category_frequency_after_cleaning(data, columns, top_n=15, n_cols=2):
    """
    Plots category frequencies after semantic cleaning.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        columns (list[str]): categorical columns to plot
        top_n (int): maximum number of categories shown per column
        n_cols (int): number of subplot columns in the figure

    Returns:
        None
    """
    plot_compact_value_counts(data, columns=columns, top_n=top_n, n_cols=n_cols)


def plot_numeric_correlation_heatmap(data, numeric_cols=("Precio", "log_Precio", "Año", "Kilómetros", "Puertas"),
                                     price_col="Precio", add_log_price=True,
                                     title="Correlación entre variables numéricas"):
    """
    Plots a correlation heatmap for numeric variables.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        numeric_cols (tuple | list): numeric columns to include
        price_col (str): price column used to create log price if requested
        add_log_price (bool): whether to create log_Precio temporarily
        title (str): plot title

    Returns:
        None
    """
    plot_data = data.copy()

    if add_log_price and price_col in plot_data.columns and "log_Precio" not in plot_data.columns:
        prices = pd.to_numeric(plot_data[price_col], errors="coerce")
        plot_data["log_Precio"] = np.where(prices > 0, np.log1p(prices), np.nan)

    corr_data = _numeric_plot_data(plot_data, numeric_cols).dropna(axis=1, how="all")
    corr = corr_data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    # Write correlation values inside each cell
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(
                col,
                row,
                f"{corr.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlación")

    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_median_price_heatmap(data, row_col="Marca", col_col="Año", price_col="Precio",
                              top_n_rows=12, min_count=30, title=None):
    """
    Plots a heatmap of median price by two categorical or discrete variables.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        row_col (str): variable shown in rows
        col_col (str): variable shown in columns
        price_col (str): price column name
        top_n_rows (int): maximum number of frequent row categories to include
        min_count (int): minimum number of observations required per row category
        title (str | None): plot title

    Returns:
        None
    """
    plot_data = data[[row_col, col_col, price_col]].copy()
    plot_data[row_col] = plot_data[row_col].fillna("missing").astype(str)
    plot_data[col_col] = plot_data[col_col].fillna("missing").astype(str)
    plot_data[price_col] = pd.to_numeric(plot_data[price_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[price_col])

    selected_rows = _filter_frequent_categories(
        plot_data,
        category_col=row_col,
        min_count=min_count,
        top_n=top_n_rows,
    )

    plot_data = plot_data[plot_data[row_col].isin(selected_rows)]

    pivot_table = (
        plot_data
        .pivot_table(index=row_col, columns=col_col, values=price_col, aggfunc="median")
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(pivot_table.columns)), 0.45 * len(pivot_table) + 3))
    image = ax.imshow(pivot_table, aspect="auto", cmap="YlGnBu")

    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot_table.index)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(f"Mediana de {price_col}")

    ax.set_title(title or f"Precio mediano por {row_col} y {col_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)

    plt.tight_layout()
    plt.show()


def plot_median_price_by_age_lines(data, group_col, age_col="Año", price_col="Precio",
                                   top_n=8, min_count=80, title=None):
    """
    Plots median price by age for frequent groups.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        group_col (str): grouping column, such as brand or model
        age_col (str): vehicle age column
        price_col (str): price column name
        top_n (int): maximum number of frequent groups to plot
        min_count (int): minimum number of rows required per group
        title (str | None): plot title

    Returns:
        None
    """
    plot_data = data[[group_col, age_col, price_col]].copy()
    plot_data[group_col] = plot_data[group_col].fillna("missing").astype(str)
    plot_data[age_col] = pd.to_numeric(plot_data[age_col], errors="coerce")
    plot_data[price_col] = pd.to_numeric(plot_data[price_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[age_col, price_col])

    selected_groups = _filter_frequent_categories(
        plot_data,
        category_col=group_col,
        min_count=min_count,
        top_n=top_n,
    )

    plot_data = plot_data[plot_data[group_col].isin(selected_groups)]

    summary = (
        plot_data
        .groupby([group_col, age_col])[price_col]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for group_name, group_data in summary.groupby(group_col):
        group_data = group_data.sort_values(age_col)
        ax.plot(
            group_data[age_col],
            group_data[price_col],
            marker="o",
            linewidth=2,
            label=group_name,
        )

    ax.set_title(title or f"Precio mediano por {age_col} y {group_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(age_col)
    ax.set_ylabel(f"Mediana de {price_col}")
    ax.grid(alpha=0.25)
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_iqr_ranking_by_category(data, category_col="Marca", price_col="Precio",
                                 top_n=15, min_count=30, title=None):
    """
    Ranks categories by price interquartile range.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        category_col (str): categorical feature to rank
        price_col (str): price column name
        top_n (int): maximum number of categories shown
        min_count (int): minimum number of rows required per category
        title (str | None): plot title

    Returns:
        pd.DataFrame: IQR ranking table used for the plot
    """
    plot_data = _category_price_data(
        data,
        category_col=category_col,
        price_col=price_col,
        min_count=min_count,
        top_n=None,
    )

    summary = (
        plot_data
        .groupby(category_col)[price_col]
        .agg(
            count="count",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )

    summary["iqr"] = summary["q3"] - summary["q1"]
    summary = summary.sort_values("iqr", ascending=False).head(top_n)
    plot_summary = summary.sort_values("iqr")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_summary))))

    ax.barh(plot_summary[category_col], plot_summary["iqr"], color=FORMAL_COLORS["red"], alpha=0.85)
    _add_horizontal_bar_labels(ax, plot_summary["iqr"].values)

    ax.set_title(title or f"Ranking de IQR de {price_col} por {category_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel("IQR")
    ax.set_ylabel(category_col)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return summary.reset_index(drop=True)
