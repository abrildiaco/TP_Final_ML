import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ========================= Plot Style =========================

FORMAL_COLORS = {
    "blue": "#1F4E79",
    "teal": "#2A9D8F",
    "gold": "#77547E",
    "red": "#A23E48",
    "gray": "#6C757D",
    "light_gray": "#E9ECEF",
}


def _resolve_price_currency(data=None, currency=None, currency_col="Moneda"):
    """
    Resolves a currency label only when it is explicit or unambiguous.

    Arguments:
        data (pd.DataFrame | None): dataset used in the plot
        currency (str | None): explicit currency label
        currency_col (str): currency column name

    Returns:
        str | None: currency label when it can be safely shown
    """
    if currency is not None and str(currency).strip():
        return str(currency)

    if data is not None and currency_col in data.columns:
        currencies = data[currency_col].dropna().astype(str).unique()

        if len(currencies) == 1:
            return currencies[0]

    return None


def _display_label(column, data=None, currency=None, currency_col="Moneda"):
    """
    Returns a readable plot label for project columns.

    Arguments:
        column (str): column name used in a plot
        data (pd.DataFrame | None): dataset used to infer currency when possible
        currency (str | None): explicit currency label
        currency_col (str): currency column name

    Returns:
        str: display label
    """
    resolved_currency = _resolve_price_currency(
        data=data,
        currency=currency,
        currency_col=currency_col,
    )

    if column == "Precio":
        return f"Precio ({resolved_currency})" if resolved_currency else "Precio"

    if column in ["log_Precio", "log(Precio)"]:
        return f"log({_display_label('Precio', data=data, currency=currency, currency_col=currency_col)})"

    return str(column)


def _log_display_label(column, data=None, currency=None, currency_col="Moneda"):
    """
    Returns a readable log-scale label for project columns.

    Arguments:
        column (str): column name used in a plot
        data (pd.DataFrame | None): dataset used to infer currency when possible
        currency (str | None): explicit currency label
        currency_col (str): currency column name

    Returns:
        str: display label for the log-transformed column
    """
    if column == "Precio":
        return f"log({_display_label(column, data=data, currency=currency, currency_col=currency_col)})"

    return f"log({_display_label(column, data=data, currency=currency, currency_col=currency_col)})"


def _add_top_n_to_title(title, top_n, item_label="categorías"):
    """
    Adds a top-N note to a plot title when the plot is filtered by frequency.

    Arguments:
        title (str): base plot title
        top_n (int | None): number of items shown
        item_label (str): item description used in the title

    Returns:
        str: title with top-N note when applicable
    """
    if top_n is None:
        return title

    if "top" in title.lower():
        return title

    return f"{title} (top {top_n} {item_label})"


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

    bars = ax.bar(currency_counts.index.astype(str), currency_counts.values, color=FORMAL_COLORS["teal"])

    # Add absolute count labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Moneda")
    ax.set_ylabel("Cantidad de publicaciones")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_price_distribution_by_currency(data, price_col="Precio", currency_col="Moneda", bins=40, title=None):
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

    fig.suptitle(
        title or f"Distribución de {_display_label(price_col, data=plot_data, currency_col=currency_col)} por moneda",
        fontsize=15,
        fontweight="bold",
    )

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
        ax.set_xlabel(_display_label(price_col, currency=currency))
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows), constrained_layout=True)
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

    _, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = np.asarray(axes).flatten()

    bar_color = FORMAL_COLORS["teal"]

    for ax, column in zip(axes, columns):
        counts = df[column].value_counts(dropna=False).head(top_n)
        counts = counts.sort_values()

        labels = ["Missing" if pd.isna(value) else str(value) for value in counts.index]
        y_positions = np.arange(len(labels))

        bars = ax.barh(y_positions, counts.values, color=bar_color, alpha=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)

        max_value = counts.values.max()
        ax.set_xlim(0, max_value * 1.15)

        # Add count labels with extra space to avoid overlap
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max_value * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", ha="left", fontsize=9)

        ax.set_title(_add_top_n_to_title(str(column), top_n, "categorías"), fontweight="bold", fontsize=12)
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(columns):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ========================= Numeric Distributions =========================

def plot_raw_numeric_distributions(data, numeric_cols=("Año", "Kilómetros"), bins=35, title="Distribución raw de variables numéricas", use_percentile_range=True):
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

        if column == "Año":
            min_value = int(np.floor(plot_values.min()))
            max_value = int(np.ceil(plot_values.max()))
            column_bins = np.arange(min_value, max_value + 2) - 0.5
        else:
            column_bins = bins

        ax.hist(plot_values, bins=column_bins, color=FORMAL_COLORS["blue"], edgecolor="white", alpha=0.85)
        ax.axvline( values.median(), color=FORMAL_COLORS["gold"], linestyle="--", linewidth=2, label=f"Mediana: {values.median():.0f}")

        ax.set_title(_display_label(column, data=data), fontweight="bold")
        ax.set_xlabel(_display_label(column, data=data))
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    for ax in axes[len(available_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


# ========================= Outlier Plots =========================

def plot_preliminary_outliers(data, numeric_cols=("Precio", "Año", "Puertas"), currency_col="Moneda", title="Outliers preliminares"):
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
                medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2)
            )

            ax.set_xlabel("Moneda")

        else:
            values = plot_data[column].dropna()

            ax.boxplot(
                values,
                patch_artist=True,
                boxprops=dict(facecolor=FORMAL_COLORS["light_gray"], color=FORMAL_COLORS["blue"]),
                medianprops=dict(color=FORMAL_COLORS["red"], linewidth=2)
            )

        ax.set_title(_display_label(column, data=plot_data, currency_col=currency_col), fontweight="bold")
        ax.set_ylabel(_display_label(column, data=plot_data, currency_col=currency_col))
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


def iqr_bounds(values, iqr_multiplier=1.5):
    """
    Computes IQR lower and upper bounds for a numeric series.

    Arguments:
        values (pd.Series): numeric values
        iqr_multiplier (float): multiplier applied to the IQR

    Returns:
        tuple[float, float, float]: lower bound, upper bound and IQR
    """
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    return lower_bound, upper_bound, iqr


def resolve_context_columns(data, value_col, group_col=None, context_cols=None):
    """
    Resolves which columns should be attached to an outlier audit table.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        value_col (str): numeric column used to detect outliers
        group_col (str | None): optional group column
        context_cols (list[str] | tuple[str] | str | None): extra columns to keep.
            Use "all" to keep every available feature except value_col and group_col

    Returns:
        list[str]: context columns present in the dataset
    """
    excluded_cols = {value_col, group_col, None}

    if context_cols == "all":
        return [column for column in data.columns if column not in excluded_cols]

    context_cols = context_cols or []

    return [
        column for column in context_cols
        if column in data.columns and column not in excluded_cols
    ]


def detect_iqr_outliers(data, value_col, group_col=None, context_cols=None,
                        iqr_multiplier=1.5, min_group_size=30,
                        side="both"):
    """
    Detects outliers using the IQR rule and returns row-level context.

    If group_col is provided, IQR bounds are computed within each group. This is
    useful for variables such as price, where a value can be extreme globally
    but reasonable inside a brand or model.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        value_col (str): numeric column used to detect outliers
        group_col (str | None): optional group column for grouped IQR bounds
        context_cols (list[str] | tuple[str] | str | None): additional columns
            included in the returned outlier table. Use "all" to include every
            available feature
        iqr_multiplier (float): multiplier applied to the IQR
        min_group_size (int): minimum group size required when group_col is used
        side (str): one of "both", "high" or "low"

    Returns:
        pd.DataFrame: detected outliers with bounds and context columns
    """
    valid_sides = {"both", "high", "low"}

    if side not in valid_sides:
        raise ValueError(f"side must be one of {valid_sides}.")

    context_cols = resolve_context_columns(data, value_col, group_col, context_cols)
    columns_to_keep = [value_col]

    if group_col is not None:
        columns_to_keep.append(group_col)

    columns_to_keep.extend(context_cols)

    plot_data = data[columns_to_keep].copy()
    plot_data["row_index"] = data.index
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[value_col])

    outlier_tables = []

    if group_col is None:
        lower_bound, upper_bound, iqr = iqr_bounds(plot_data[value_col], iqr_multiplier)
        group_bounds = [(None, plot_data, lower_bound, upper_bound, iqr)]
    else:
        group_bounds = []

        for group_name, group_data in plot_data.groupby(group_col, dropna=False):
            if len(group_data) < min_group_size:
                continue

            lower_bound, upper_bound, iqr = iqr_bounds(group_data[value_col], iqr_multiplier)
            group_bounds.append((group_name, group_data, lower_bound, upper_bound, iqr))

    for group_name, group_data, lower_bound, upper_bound, iqr in group_bounds:
        high_mask = group_data[value_col] > upper_bound
        low_mask = group_data[value_col] < lower_bound

        if side == "high":
            outlier_mask = high_mask
        elif side == "low":
            outlier_mask = low_mask
        else:
            outlier_mask = high_mask | low_mask

        outliers = group_data[outlier_mask].copy()

        if outliers.empty:
            continue

        outliers["lower_bound"] = lower_bound
        outliers["upper_bound"] = upper_bound
        outliers["iqr"] = iqr
        outliers["outlier_type"] = np.where(
            outliers[value_col] > upper_bound,
            "high",
            "low",
        )

        # Score is scaled by IQR so outliers from different groups are comparable
        denominator = iqr if iqr != 0 else 1
        outliers["outlier_score"] = np.where(
            outliers["outlier_type"] == "high",
            (outliers[value_col] - upper_bound) / denominator,
            (lower_bound - outliers[value_col]) / denominator,
        )

        outlier_tables.append(outliers)

    if not outlier_tables:
        output_cols = [
            "row_index",
            value_col,
            "lower_bound",
            "upper_bound",
            "iqr",
            "outlier_type",
            "outlier_score",
        ]

        if group_col is not None:
            output_cols.insert(2, group_col)

        output_cols.extend([
            column for column in context_cols
            if column in data.columns and column not in output_cols
        ])

        return pd.DataFrame(columns=output_cols)

    outlier_table = pd.concat(outlier_tables, axis=0)

    output_cols = [
        "row_index",
        value_col,
        "lower_bound",
        "upper_bound",
        "iqr",
        "outlier_type",
        "outlier_score",
    ]

    if group_col is not None:
        output_cols.insert(2, group_col)

    output_cols.extend([
        column for column in context_cols
        if column in outlier_table.columns and column not in output_cols
    ])

    return (
        outlier_table[output_cols]
        .sort_values("outlier_score", ascending=False)
        .reset_index(drop=True)
    )


def plot_iqr_outliers(data, value_col, group_col=None, context_cols=None,
                      iqr_multiplier=1.5, min_group_size=30, side="both",
                      max_groups=20, top_n_labels=12, title=None):
    """
    Plots a numeric variable and highlights IQR outliers in red.

    The function also returns a table with the detected outliers and selected
    context columns, so each highlighted point can be inspected.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        value_col (str): numeric column used to detect outliers
        group_col (str | None): optional group column for grouped outlier plots
        context_cols (list[str] | None): additional columns returned for context
        iqr_multiplier (float): multiplier applied to the IQR
        min_group_size (int): minimum group size required when group_col is used
        side (str): one of "both", "high" or "low"
        max_groups (int): maximum number of groups shown when group_col is used
        top_n_labels (int): number of strongest outliers annotated in the plot
        title (str | None): plot title

    Returns:
        pd.DataFrame: detected outliers with context columns
    """
    outliers = detect_iqr_outliers(
        data,
        value_col=value_col,
        group_col=group_col,
        context_cols=context_cols,
        iqr_multiplier=iqr_multiplier,
        min_group_size=min_group_size,
        side=side,
    )

    plot_cols = [value_col]

    if group_col is not None:
        plot_cols.append(group_col)

    plot_data = data[plot_cols].copy()
    plot_data["row_index"] = data.index
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[value_col])

    if group_col is None:
        plot_data["_x_position"] = 0
        x_labels = [_display_label(value_col, data=data)]
        figsize = (6, 5)
    else:
        if outliers.empty:
            selected_groups = plot_data[group_col].value_counts().head(max_groups).index.tolist()
        else:
            selected_groups = outliers[group_col].value_counts().head(max_groups).index.tolist()

        plot_data = plot_data[plot_data[group_col].isin(selected_groups)].copy()
        x_positions = {group: index for index, group in enumerate(selected_groups)}
        plot_data["_x_position"] = plot_data[group_col].map(x_positions)
        x_labels = [str(group) for group in selected_groups]
        figsize = (max(8, 0.55 * len(selected_groups)), 5.5)

    outlier_indices = set(outliers["row_index"])
    plot_data["is_outlier"] = plot_data["row_index"].isin(outlier_indices)

    fig, ax = plt.subplots(figsize=figsize)

    normal_data = plot_data[~plot_data["is_outlier"]]
    outlier_plot_data = plot_data[plot_data["is_outlier"]]

    rng = np.random.default_rng(42)

    normal_x = normal_data["_x_position"] + rng.uniform(-0.08, 0.08, size=len(normal_data))
    outlier_x = outlier_plot_data["_x_position"] + rng.uniform(-0.08, 0.08, size=len(outlier_plot_data))

    ax.scatter(
        normal_x,
        normal_data[value_col],
        color=FORMAL_COLORS["gray"],
        alpha=0.25,
        s=18,
        label="Normal rows",
    )

    ax.scatter(
        outlier_x,
        outlier_plot_data[value_col],
        color=FORMAL_COLORS["red"],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.95,
        s=42,
        label="Outliers",
    )

    label_rows = outliers.head(top_n_labels)

    for row in label_rows.itertuples():
        match = outlier_plot_data[outlier_plot_data["row_index"] == row.row_index]

        if match.empty:
            continue

        x_value = match["_x_position"].iloc[0]
        y_value = match[value_col].iloc[0]

        ax.annotate(
            str(row.row_index),
            xy=(x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=FORMAL_COLORS["red"],
        )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel(_display_label(value_col, data=data))
    default_title = f"Outliers de {_display_label(value_col, data=data)}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n_labels, "outliers etiquetados"), fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return outliers


def plot_iqr_outlier_scatter(data, x_col, y_col, outlier_col=None,
                             context_cols=None, group_col=None,
                             iqr_multiplier=1.5, min_group_size=30,
                             side="both", top_n_labels=12,
                             sample_size=None, title=None):
    """
    Plots a scatter plot and highlights IQR outliers in red.

    Outliers are detected using outlier_col. For example, x_col can be
    Kilómetros, y_col can be Precio, and outlier_col can be Precio.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        x_col (str): x-axis numeric column
        y_col (str): y-axis numeric column
        outlier_col (str | None): column used to detect outliers. If None, y_col
            is used
        context_cols (list[str] | None): additional columns returned for context
        group_col (str | None): optional group column for grouped IQR bounds
        iqr_multiplier (float): multiplier applied to the IQR
        min_group_size (int): minimum group size required when group_col is used
        side (str): one of "both", "high" or "low"
        top_n_labels (int): number of strongest outliers annotated in the plot
        sample_size (int | None): optional sample size for normal rows
        title (str | None): plot title

    Returns:
        pd.DataFrame: detected outliers with context columns
    """
    outlier_col = outlier_col or y_col

    outliers = detect_iqr_outliers(
        data,
        value_col=outlier_col,
        group_col=group_col,
        context_cols=context_cols,
        iqr_multiplier=iqr_multiplier,
        min_group_size=min_group_size,
        side=side,
    )

    plot_data = data[[x_col, y_col]].copy()
    plot_data["row_index"] = data.index
    plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors="coerce")
    plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[x_col, y_col])
    plot_data["is_outlier"] = plot_data["row_index"].isin(set(outliers["row_index"]))

    normal_data = plot_data[~plot_data["is_outlier"]]
    outlier_plot_data = plot_data[plot_data["is_outlier"]]

    if sample_size is not None and len(normal_data) > sample_size:
        normal_data = normal_data.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.scatter(
        normal_data[x_col],
        normal_data[y_col],
        color=FORMAL_COLORS["gray"],
        alpha=0.25,
        s=18,
        label="Normal rows",
    )

    ax.scatter(
        outlier_plot_data[x_col],
        outlier_plot_data[y_col],
        color=FORMAL_COLORS["red"],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.95,
        s=45,
        label="Outliers",
    )

    label_rows = outliers.head(top_n_labels)

    for row in label_rows.itertuples():
        match = outlier_plot_data[outlier_plot_data["row_index"] == row.row_index]

        if match.empty:
            continue

        ax.annotate(
            str(row.row_index),
            xy=(match[x_col].iloc[0], match[y_col].iloc[0]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=FORMAL_COLORS["red"],
        )

    default_title = f"{_display_label(y_col, data=data)} vs {_display_label(x_col, data=data)} con outliers resaltados"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n_labels, "outliers etiquetados"), fontsize=14, fontweight="bold")
    ax.set_xlabel(_display_label(x_col, data=data))
    ax.set_ylabel(_display_label(y_col, data=data))
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return outliers


def plot_outliers_by_category(df, category_col, price_col="Precio", 
                            min_count=10, n_splits=2, figsize=(14, 6)):
    """
    Plots price distribution by category using boxplots, filtering out
    categories with too few observations and splitting the chart into
    multiple subplots ordered by median price. Useful for datasets with
    many categories where a single plot would be unreadable.

    Arguments:
        df (pd.DataFrame): dataset to analyze
        category_col (str): categorical column to group by (e.g. "Marca")
        price_col (str): numeric column to plot, defaults to "Precio"
        min_count (int): minimum number of rows a category must have to be
            included, defaults to 10
        n_splits (int): number of subplots to split the categories into,
            defaults to 2
        figsize (tuple): size of each individual subplot, defaults to (14, 6)

    Returns:
        None
    """
    # Filter categories with enough observations
    valid_categories = (
        df[category_col]
        .value_counts()
        .loc[lambda counts: counts >= min_count]
        .index
    )

    df_filtered = df[df[category_col].isin(valid_categories)].copy()

    n_filtered = df[category_col].nunique() - len(valid_categories)
    print(f"Filtered out {n_filtered} categories with fewer than {min_count} rows.")
    print(f"Plotting {len(valid_categories)} categories split into {n_splits} charts.")

    # Sort categories by median price descending
    order = (
        df_filtered.groupby(category_col)[price_col]
        .median()
        .sort_values(ascending=False)
        .index
    )

    # Split sorted categories evenly across subplots
    splits = np.array_split(order, n_splits)

    fig, axes = plt.subplots(n_splits, 1, figsize=(figsize[0], figsize[1] * n_splits))

    if n_splits == 1:
        axes = [axes]

    for i, (ax, split_categories) in enumerate(zip(axes, splits)):
        split_data = df_filtered[df_filtered[category_col].isin(split_categories)]

        sns.boxplot(
            data=split_data,
            x=category_col,
            y=price_col,
            order=split_categories,
            ax=ax,
            color = FORMAL_COLORS["red"]
        )

        ax.set_title(
            f"{price_col} by {category_col} — "
            f"ranked {i + 1} of {n_splits} (highest to lowest median)",
            fontweight="bold"
        )
        
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.show()


# ========================= Post-Preprocessing EDA =========================

def numeric_plot_data(data, columns):
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


def filter_frequent_categories(data, category_col, min_count=30, top_n=None):
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

    selected_categories = filter_frequent_categories(
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
        x_label = _log_display_label(price_col, data=data)
        default_title = f"Distribución de {_log_display_label(price_col, data=data)}"
    else:
        x_label = _display_label(price_col, data=data)
        default_title = f"Distribución de {_display_label(price_col, data=data)}"

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
    plot_data = numeric_plot_data(data, numeric_cols)
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

        ax.set_title(_display_label(column, data=data), fontweight="bold")
        ax.set_xlabel(_display_label(column, data=data))
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
    plot_data = numeric_plot_data(data, [x_col, price_col]).dropna()

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

    ax.set_title(title or f"{_display_label(price_col, data=data)} vs {_display_label(x_col, data=data)}", fontsize=14, fontweight="bold")
    ax.set_xlabel(_display_label(x_col, data=data))
    ax.set_ylabel(_display_label(price_col, data=data))
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
    plot_data = numeric_plot_data(data, [year_col, km_col, price_col]).dropna()

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
    colorbar.set_label(_display_label(price_col, data=data))

    ax.set_title(f"{year_col} vs {km_col} coloreado por {_display_label(price_col, data=data)}", fontsize=14, fontweight="bold")
    ax.set_xlabel(_display_label(year_col, data=data))
    ax.set_ylabel(_display_label(km_col, data=data))
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

    default_title = f"{_display_label(price_col, data=data)} mediano por {category_col}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n, "categorías"), fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Mediana de {_display_label(price_col, data=data)}")
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

    default_title = f"Distribución de {_display_label(price_col, data=data)} por {category_col}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n, "categorías"), fontsize=14, fontweight="bold")
    ax.set_xlabel(category_col)
    ax.set_ylabel(_display_label(price_col, data=data))
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


def build_plot_dataset(features, target, target_col="Precio"):
    """
    Adds the target column to a feature dataframe for EDA plots.

    Arguments:
        features (pd.DataFrame): feature dataset
        target (pd.Series | np.ndarray | list): target values
        target_col (str): name assigned to the target column

    Returns:
        pd.DataFrame: copy of features with the target column added
    """
    plot_data = features.copy()

    if isinstance(target, pd.Series):
        plot_data[target_col] = target.reindex(plot_data.index).values
    else:
        plot_data[target_col] = target

    return plot_data


def get_feature_columns(data, feature_type="numeric", target_col="Precio",
                        include_target=False, include_log_target=False,
                        exclude_cols=None):
    """
    Selects numeric, binary or both types of columns from a dataset.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        feature_type (str): one of "numeric", "binary" or "both"
        target_col (str | None): target column name
        include_target (bool): whether to include the target column first
        include_log_target (bool): whether to include log target if it exists
        exclude_cols (list[str] | None): columns to ignore

    Returns:
        list[str]: selected column names
    """
    valid_feature_types = {"numeric", "binary", "both"}

    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of {valid_feature_types}.")

    exclude_cols = set(exclude_cols or [])

    if target_col is not None:
        exclude_cols.add(target_col)
        exclude_cols.add(f"log_{target_col}")

    numeric_candidates = [
        column for column in data.select_dtypes(include="number").columns
        if column not in exclude_cols
    ]

    binary_cols = []

    for column in numeric_candidates:
        values = data[column].dropna().unique()

        if len(values) > 0 and set(values).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(column)

    numeric_cols = [
        column for column in numeric_candidates
        if column not in binary_cols
    ]

    if feature_type == "numeric":
        selected_cols = numeric_cols
    elif feature_type == "binary":
        selected_cols = binary_cols
    else:
        selected_cols = numeric_cols + binary_cols

    output_cols = []

    if include_target and target_col in data.columns:
        output_cols.append(target_col)

    log_target_col = f"log_{target_col}"

    if include_log_target and log_target_col in data.columns:
        output_cols.append(log_target_col)

    output_cols.extend(selected_cols)

    return list(dict.fromkeys(output_cols))


def get_non_one_hot_binary_columns(data, target_col="Precio", one_hot_prefixes=None,
                                   exclude_cols=None):
    """
    Selects binary columns that are not one-hot encoded categorical columns.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        target_col (str | None): target column name
        one_hot_prefixes (list[str] | None): prefixes used by one-hot columns.
            If None, common project prefixes are used
        exclude_cols (list[str] | None): additional columns to ignore

    Returns:
        list[str]: binary columns that do not start with one-hot prefixes
    """
    if one_hot_prefixes is None:
        one_hot_prefixes = [
            "Marca_",
            "Modelo_",
            "Color_",
            "Transmisión_",
            "Tipo de combustible_",
            "Tipo de vendedor_",
        ]

    binary_cols = get_feature_columns(
        data,
        feature_type="binary",
        target_col=target_col,
        exclude_cols=exclude_cols,
    )

    return [
        column for column in binary_cols
        if not any(column.startswith(prefix) for prefix in one_hot_prefixes)
    ]


def plot_numeric_and_binary_correlation_heatmap(data, target_col="Precio",
                                                one_hot_prefixes=None,
                                                include_cols=None,
                                                include_one_hot_prefixes=None,
                                                top_n_one_hot=None,
                                                min_one_hot_frequency=0.0,
                                                max_one_hot_frequency=1.0,
                                                include_log_target=True,
                                                title="Correlación de variables numéricas y binarias no one-hot"):
    """
    Plots a compact heatmap with numeric columns and selected binary columns.

    This is useful after one-hot encoding when we want to keep engineered binary
    signals, such as turbo or backup camera, but avoid hundreds of dummy columns.
    One-hot groups can be included explicitly when needed, for example Color.

    Arguments:
        data (pd.DataFrame): dataset including the target column
        target_col (str): target column name
        one_hot_prefixes (list[str] | None): prefixes excluded from the default
            non-one-hot binary selection
        include_cols (list[str] | None): specific columns to add to the heatmap
        include_one_hot_prefixes (list[str] | None): one-hot groups to add,
            such as ["Color"] or ["Color_"]
        top_n_one_hot (int | None): maximum number of most frequent one-hot
            columns added per prefix
        min_one_hot_frequency (float): minimum one-hot column mean required
        max_one_hot_frequency (float): maximum one-hot column mean allowed
        include_log_target (bool): whether to include log target in the heatmap
        title (str): plot title

    Returns:
        None
    """
    numeric_cols = get_feature_columns(
        data,
        feature_type="numeric",
        target_col=target_col,
        include_target=True,
        include_log_target=include_log_target,
    )

    binary_cols = get_non_one_hot_binary_columns(
        data,
        target_col=target_col,
        one_hot_prefixes=one_hot_prefixes,
    )

    include_cols = include_cols or []
    explicit_cols = [
        column for column in include_cols
        if column in data.columns
    ]

    one_hot_cols = []

    for prefix in include_one_hot_prefixes or []:
        prefix_text = prefix if str(prefix).endswith("_") else f"{prefix}_"
        prefix_cols = []

        for column in data.columns:
            if not column.startswith(prefix_text):
                continue

            values = pd.to_numeric(data[column], errors="coerce")
            frequency = values.mean()

            if pd.isna(frequency):
                continue

            if frequency < min_one_hot_frequency or frequency > max_one_hot_frequency:
                continue

            prefix_cols.append((column, frequency))

        prefix_cols = sorted(prefix_cols, key=lambda item: item[1], reverse=True)

        if top_n_one_hot is not None:
            prefix_cols = prefix_cols[:top_n_one_hot]

        one_hot_cols.extend([column for column, _ in prefix_cols])

    selected_cols = list(dict.fromkeys(numeric_cols + binary_cols + explicit_cols + one_hot_cols))

    plot_numeric_correlation_heatmap(
        data,
        numeric_cols=selected_cols,
        price_col=target_col,
        add_log_price=include_log_target,
        include_target=False,
        include_log_target=False,
        title=title,
    )


def plot_numeric_correlation_heatmap(data, numeric_cols=None, feature_type="numeric",
                                     price_col="Precio", add_log_price=True,
                                     include_target=True, include_log_target=True,
                                     title="Correlación entre variables numéricas",
                                     figsize=None, cell_size=0.6, annotate=None):
    """
    Plots a correlation heatmap for numeric variables.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        numeric_cols (tuple | list | None): columns to include. If None, columns
            are selected automatically using feature_type
        feature_type (str): one of "numeric", "binary" or "both"
        price_col (str): price column used to create log price if requested
        add_log_price (bool): whether to create log_Precio temporarily
        include_target (bool): whether to include price in the heatmap
        include_log_target (bool): whether to include log price in the heatmap
        title (str): plot title
        figsize (tuple[float, float] | None): custom figure size
        cell_size (float): size multiplier used when figsize is computed
        annotate (bool | None): whether to write correlation values in cells.
            If None, annotation is used only for small heatmaps

    Returns:
        None
    """
    plot_data = data.copy()

    if add_log_price and price_col in plot_data.columns and "log_Precio" not in plot_data.columns:
        prices = pd.to_numeric(plot_data[price_col], errors="coerce")
        plot_data["log_Precio"] = np.where(prices > 0, np.log1p(prices), np.nan)

    if numeric_cols is None:
        numeric_cols = get_feature_columns(
            plot_data,
            feature_type=feature_type,
            target_col=price_col,
            include_target=include_target,
            include_log_target=include_log_target,
        )

    corr_data = numeric_plot_data(plot_data, numeric_cols).dropna(axis=1, how="all")
    corr = corr_data.corr()

    if corr.empty:
        raise ValueError("No numeric columns available for the correlation heatmap.")

    n_features = len(corr.columns)

    if figsize is None:
        figsize = (
            max(8, n_features * cell_size + 2),
            max(6, n_features * cell_size + 1.5),
        )

    if annotate is None:
        annotate = n_features <= 18

    label_fontsize = 9 if n_features <= 18 else 7 if n_features <= 40 else 5

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([_display_label(column, data=plot_data) for column in corr.columns], rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticklabels([_display_label(column, data=plot_data) for column in corr.index], fontsize=label_fontsize)

    if annotate:
        # Write correlation values only when the heatmap is small enough to read
        for row in range(len(corr.index)):
            for col in range(len(corr.columns)):
                ax.text(
                    col,
                    row,
                    f"{corr.iloc[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlación")

    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def binary_feature_columns(data, exclude_cols=None):
    """
    Finds binary numeric columns in a dataset.

    Arguments:
        data (pd.DataFrame): dataset to inspect
        exclude_cols (list[str] | None): columns to ignore

    Returns:
        list[str]: columns that only contain 0/1 values after dropping missing values
    """
    return get_feature_columns(
        data,
        feature_type="binary",
        target_col=None,
        exclude_cols=exclude_cols,
    )


def feature_group_name(column):
    """
    Gets a readable group name from an encoded feature name.

    Arguments:
        column (str): feature name

    Returns:
        str: group name before the first underscore, or Binary features
    """
    if "_" in column:
        return column.split("_", 1)[0]

    return "Binary features"


def encoded_feature_target_correlation_table(data, target_col="Precio", feature_cols=None,
                                             feature_type="binary", use_log_target=True,
                                             min_frequency=None, max_frequency=None):
    """
    Builds a table with feature frequency and correlation against the target.

    This is safer than plotting a full correlation matrix after one-hot encoding,
    because it compares each encoded feature directly with the target instead of
    comparing every feature against every other feature.

    Arguments:
        data (pd.DataFrame): encoded dataset including the target column
        target_col (str): target column name
        feature_cols (list[str] | None): features to analyze. If None, columns
            are selected automatically using feature_type
        feature_type (str): one of "numeric", "binary" or "both"
        use_log_target (bool): whether to correlate features with log1p(target)
        min_frequency (float | None): minimum feature mean required
        max_frequency (float | None): maximum feature mean allowed

    Returns:
        pd.DataFrame: feature-level correlation summary
    """
    plot_data = data.copy()
    target = pd.to_numeric(plot_data[target_col], errors="coerce")

    if use_log_target:
        target = np.where(target > 0, np.log1p(target), np.nan)

    if feature_cols is None:
        feature_cols = get_feature_columns(
            plot_data,
            feature_type=feature_type,
            target_col=target_col,
        )

    binary_cols = set(get_feature_columns(
        plot_data,
        feature_type="binary",
        target_col=target_col,
    ))

    rows = []

    for feature in feature_cols:
        values = pd.to_numeric(plot_data[feature], errors="coerce")
        is_binary = feature in binary_cols
        frequency = values.mean() if is_binary else np.nan

        if is_binary and min_frequency is not None and frequency < min_frequency:
            continue

        if is_binary and max_frequency is not None and frequency > max_frequency:
            continue

        valid_mask = values.notna() & pd.notna(target)

        if valid_mask.sum() < 2 or values.loc[valid_mask].nunique() < 2:
            correlation = np.nan
        else:
            correlation = np.corrcoef(values.loc[valid_mask], target[valid_mask])[0, 1]

        rows.append({
            "feature": feature,
            "group": feature_group_name(feature),
            "feature_type": "binary" if is_binary else "numeric",
            "frequency": frequency,
            "target_correlation": correlation,
            "abs_target_correlation": abs(correlation) if pd.notna(correlation) else np.nan,
        })

    return (
        pd.DataFrame(rows)
        .dropna(subset=["target_correlation"])
        .sort_values("abs_target_correlation", ascending=False)
        .reset_index(drop=True)
    )


def plot_encoded_feature_correlation_heatmaps(data, target_col="Precio", feature_cols=None,
                                              feature_type="binary",
                                              groups=None, top_n_per_group=20,
                                              min_frequency=0.01, max_frequency=0.99,
                                              use_log_target=True, n_cols=2):
    """
    Plots grouped heatmaps of encoded binary feature correlation with the target.

    Each heatmap shows only one group of encoded features, such as Marca, Modelo
    or Color. This avoids building a huge feature-by-feature matrix after one-hot
    encoding.

    Arguments:
        data (pd.DataFrame): encoded dataset including the target column
        target_col (str): target column name
        feature_cols (list[str] | None): features to analyze. If None, columns
            are selected automatically using feature_type
        feature_type (str): one of "numeric", "binary" or "both"
        groups (list[str] | None): encoded groups to plot, such as ["Marca",
            "Modelo", "Color"]. If None, the most common groups are used
        top_n_per_group (int): maximum number of features shown per group
        min_frequency (float): minimum feature frequency shown
        max_frequency (float): maximum feature frequency shown
        use_log_target (bool): whether to correlate features with log1p(target)
        n_cols (int): number of subplot columns

    Returns:
        pd.DataFrame: correlation table used for the plots
    """
    correlation_table = encoded_feature_target_correlation_table(
        data,
        target_col=target_col,
        feature_cols=feature_cols,
        feature_type=feature_type,
        use_log_target=use_log_target,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )

    if correlation_table.empty:
        raise ValueError("No encoded binary features available after applying the filters.")

    if groups is None:
        groups = correlation_table["group"].value_counts().head(6).index.tolist()

    groups = [group for group in groups if group in correlation_table["group"].unique()]

    if not groups:
        raise ValueError("No feature groups available to plot.")

    n_plots = len(groups)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 0.45 * top_n_per_group * n_rows + 2),
    )

    axes = np.asarray(axes).reshape(-1)
    target_label = _log_display_label(target_col, data=data) if use_log_target else _display_label(target_col, data=data)

    for ax, group in zip(axes, groups):
        group_data = (
            correlation_table[correlation_table["group"] == group]
            .head(top_n_per_group)
            .sort_values("target_correlation")
        )

        heatmap_values = group_data[["target_correlation"]].values

        image = ax.imshow(heatmap_values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        labels = []

        for row in group_data.itertuples():
            if pd.notna(row.frequency):
                labels.append(f"{row.feature}\n(freq={row.frequency:.2f})")
            else:
                labels.append(row.feature)

        ax.set_yticks(range(len(group_data)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels([target_label], fontsize=9)
        ax.set_title(group, fontsize=12, fontweight="bold")

        for row_index, value in enumerate(group_data["target_correlation"]):
            ax.text(
                0,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    for ax in axes[n_plots:]:
        ax.axis("off")

    colorbar = fig.colorbar(image, ax=axes[:n_plots], shrink=0.85)
    colorbar.set_label(f"Correlación con {target_label}")

    suptitle = _add_top_n_to_title(
        f"Correlación de features encoded con {_display_label(target_col, data=data)}",
        top_n_per_group,
        "features por grupo",
    )
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return correlation_table


def plot_top_target_correlations(data, target_col="Precio", feature_cols=None,
                                 feature_type="both", top_n=25, use_log_target=True,
                                 exclude_binary=False, title=None):
    """
    Plots the strongest feature correlations with the target as a bar chart.

    This is useful after one-hot encoding to summarize the strongest signals
    without plotting a huge correlation matrix.

    Arguments:
        data (pd.DataFrame): dataset including the target column
        target_col (str): target column name
        feature_cols (list[str] | None): features to analyze. If None, columns
            are selected automatically using feature_type
        feature_type (str): one of "numeric", "binary" or "both"
        top_n (int): number of strongest positive and negative correlations shown
        use_log_target (bool): whether to correlate features with log1p(target)
        exclude_binary (bool): backward-compatible way to force numeric-only
            features when feature_cols is None
        title (str | None): plot title

    Returns:
        pd.DataFrame: sorted correlation table used for the plot
    """
    plot_data = data.copy()
    target = pd.to_numeric(plot_data[target_col], errors="coerce")

    if use_log_target:
        target = np.where(target > 0, np.log1p(target), np.nan)

    if feature_cols is None:
        selected_feature_type = "numeric" if exclude_binary else feature_type
        feature_cols = get_feature_columns(
            plot_data,
            feature_type=selected_feature_type,
            target_col=target_col,
        )
    elif exclude_binary:
        binary_cols = set(binary_feature_columns(plot_data, exclude_cols=[target_col]))
        feature_cols = [column for column in feature_cols if column not in binary_cols]

    rows = []

    for feature in feature_cols:
        values = pd.to_numeric(plot_data[feature], errors="coerce")
        valid_mask = values.notna() & pd.notna(target)

        if valid_mask.sum() < 2 or values.loc[valid_mask].nunique() < 2:
            continue

        correlation = np.corrcoef(values.loc[valid_mask], target[valid_mask])[0, 1]

        rows.append({
            "feature": feature,
            "target_correlation": correlation,
            "abs_target_correlation": abs(correlation),
        })

    correlation_table = (
        pd.DataFrame(rows)
        .sort_values("abs_target_correlation", ascending=False)
        .reset_index(drop=True)
    )

    plot_data = correlation_table.head(top_n).sort_values("target_correlation")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_data))))
    colors = [
        FORMAL_COLORS["red"] if value < 0 else FORMAL_COLORS["teal"]
        for value in plot_data["target_correlation"]
    ]

    ax.barh(plot_data["feature"], plot_data["target_correlation"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)
    target_label = _log_display_label(target_col, data=data) if use_log_target else _display_label(target_col, data=data)
    default_title = f"Features más correlacionadas con {target_label}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n, "features"), fontsize=14, fontweight="bold")
    ax.set_xlabel("Correlación")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return correlation_table


def one_hot_group_columns(data, prefix, min_frequency=0.0, max_frequency=1.0):
    """
    Selects one-hot encoded columns from a specific feature group.

    Arguments:
        data (pd.DataFrame): encoded dataset
        prefix (str): one-hot prefix, for example "Marca" or "Modelo"
        min_frequency (float): minimum column mean required
        max_frequency (float): maximum column mean allowed

    Returns:
        list[str]: one-hot columns that match the prefix and frequency filters
    """
    columns = []
    prefix_text = f"{prefix}_"

    for column in data.columns:
        if not column.startswith(prefix_text):
            continue

        values = pd.to_numeric(data[column], errors="coerce")
        frequency = values.mean()

        if pd.isna(frequency):
            continue

        if frequency < min_frequency or frequency > max_frequency:
            continue

        columns.append(column)

    return columns


def _clean_one_hot_label(column, prefix):
    """
    Removes the one-hot prefix from a column name for plot labels.

    Arguments:
        column (str): one-hot encoded column name
        prefix (str): encoded group prefix

    Returns:
        str: readable category label
    """
    return str(column).replace(f"{prefix}_", "", 1)


def one_hot_group_correlation_table(data, row_prefix="Marca", col_prefix="Modelo",
                                    min_frequency=0.005, max_frequency=0.995):
    """
    Computes correlations between two groups of one-hot encoded columns.

    This is useful for relationships such as Marca_* vs Modelo_*. Since both
    sides are binary columns, the correlation is the phi correlation: positive
    values mean that the two categories appear together more often, while
    negative values mean they rarely appear together.

    Arguments:
        data (pd.DataFrame): encoded dataset
        row_prefix (str): first one-hot prefix, used as heatmap rows
        col_prefix (str): second one-hot prefix, used as heatmap columns
        min_frequency (float): minimum frequency required for each one-hot column
        max_frequency (float): maximum frequency allowed for each one-hot column

    Returns:
        pd.DataFrame: sorted table with one row per pair of encoded categories
    """
    row_cols = one_hot_group_columns(
        data,
        prefix=row_prefix,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )
    col_cols = one_hot_group_columns(
        data,
        prefix=col_prefix,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )

    if not row_cols:
        raise ValueError(f"No one-hot columns found for prefix '{row_prefix}'.")

    if not col_cols:
        raise ValueError(f"No one-hot columns found for prefix '{col_prefix}'.")

    encoded_data = data[row_cols + col_cols].apply(pd.to_numeric, errors="coerce")
    correlation_matrix = encoded_data.corr().loc[row_cols, col_cols]

    rows = []

    for row_col in row_cols:
        for col_col in col_cols:
            correlation = correlation_matrix.loc[row_col, col_col]

            if pd.isna(correlation):
                continue

            rows.append({
                "row_feature": row_col,
                "col_feature": col_col,
                row_prefix: _clean_one_hot_label(row_col, row_prefix),
                col_prefix: _clean_one_hot_label(col_col, col_prefix),
                "row_frequency": encoded_data[row_col].mean(),
                "col_frequency": encoded_data[col_col].mean(),
                "correlation": correlation,
                "abs_correlation": abs(correlation),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )


def plot_one_hot_group_correlation_heatmap(data, row_prefix="Marca", col_prefix="Modelo",
                                           top_n_pairs=40, top_n_rows=15, top_n_cols=20,
                                           min_frequency=0.005, max_frequency=0.995,
                                           title=None, annotate=None):
    """
    Plots a compact heatmap between two one-hot encoded feature groups.

    The function first finds the strongest pairwise correlations between both
    groups, then plots only the rows and columns involved in those strongest
    pairs. This keeps large groups like Marca and Modelo readable.

    Arguments:
        data (pd.DataFrame): encoded dataset
        row_prefix (str): one-hot prefix shown in rows
        col_prefix (str): one-hot prefix shown in columns
        top_n_pairs (int): number of strongest pairs used to choose rows/columns
        top_n_rows (int): maximum number of row categories shown
        top_n_cols (int): maximum number of column categories shown
        min_frequency (float): minimum one-hot frequency required
        max_frequency (float): maximum one-hot frequency allowed
        title (str | None): plot title
        annotate (bool | None): whether to write correlation values inside cells.
            If None, annotation is used only for small heatmaps

    Returns:
        pd.DataFrame: sorted pairwise correlation table
    """
    correlation_table = one_hot_group_correlation_table(
        data,
        row_prefix=row_prefix,
        col_prefix=col_prefix,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )

    selected_pairs = correlation_table.head(top_n_pairs)
    selected_rows = list(dict.fromkeys(selected_pairs["row_feature"]))[:top_n_rows]
    selected_cols = list(dict.fromkeys(selected_pairs["col_feature"]))[:top_n_cols]

    if not selected_rows or not selected_cols:
        raise ValueError("No one-hot pairs available to plot after filtering.")

    encoded_data = data[selected_rows + selected_cols].apply(pd.to_numeric, errors="coerce")
    correlation_matrix = encoded_data.corr().loc[selected_rows, selected_cols]

    row_labels = [_clean_one_hot_label(column, row_prefix) for column in selected_rows]
    col_labels = [_clean_one_hot_label(column, col_prefix) for column in selected_cols]

    if annotate is None:
        annotate = correlation_matrix.size <= 120

    fig_width = max(8, 0.45 * len(selected_cols) + 3)
    fig_height = max(5, 0.35 * len(selected_rows) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(
        correlation_matrix.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(selected_cols)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(selected_rows)))
    ax.set_yticklabels(row_labels)

    if annotate:
        for row_index in range(len(selected_rows)):
            for col_index in range(len(selected_cols)):
                value = correlation_matrix.iloc[row_index, col_index]

                if pd.isna(value):
                    continue

                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlación")

    default_title = (
        f"Correlación entre {row_prefix} y {col_prefix} one-hot "
        f"(top {top_n_pairs} pares)"
    )
    ax.set_title(title or default_title, fontsize=14, fontweight="bold")
    ax.set_xlabel(col_prefix)
    ax.set_ylabel(row_prefix)

    plt.tight_layout()
    plt.show()

    return correlation_table


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

    selected_rows = filter_frequent_categories(
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
    colorbar.set_label(f"Mediana de {_display_label(price_col, data=data)}")

    default_title = f"{_display_label(price_col, data=data)} mediano por {row_col} y {col_col}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n_rows, "filas"), fontsize=14, fontweight="bold")
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)

    plt.tight_layout()
    plt.show()


def plot_median_price_by_year_lines(data, group_col, year_col="Año", price_col="Precio",
                                    top_n=8, min_count=80, title=None):
    """
    Plots median price by vehicle year for frequent groups.

    Arguments:
        data (pd.DataFrame): preprocessed dataset
        group_col (str): grouping column, such as brand or model
        year_col (str): vehicle year column
        price_col (str): price column name
        top_n (int): maximum number of frequent groups to plot
        min_count (int): minimum number of rows required per group
        title (str | None): plot title

    Returns:
        None
    """
    plot_data = data[[group_col, year_col, price_col]].copy()
    plot_data[group_col] = plot_data[group_col].fillna("missing").astype(str)
    plot_data[year_col] = pd.to_numeric(plot_data[year_col], errors="coerce")
    plot_data[price_col] = pd.to_numeric(plot_data[price_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[year_col, price_col])

    selected_groups = filter_frequent_categories(
        plot_data,
        category_col=group_col,
        min_count=min_count,
        top_n=top_n,
    )

    plot_data = plot_data[plot_data[group_col].isin(selected_groups)]

    summary = (
        plot_data
        .groupby([group_col, year_col])[price_col]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for group_name, group_data in summary.groupby(group_col):
        group_data = group_data.sort_values(year_col)
        ax.plot(
            group_data[year_col],
            group_data[price_col],
            marker="o",
            linewidth=2,
            label=group_name,
        )

    default_title = f"{_display_label(price_col, data=data)} mediano por {year_col} y {group_col}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n, "grupos"), fontsize=14, fontweight="bold")
    ax.set_xlabel(year_col)
    ax.set_ylabel(f"Mediana de {_display_label(price_col, data=data)}")
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
    plot_data = _category_price_data(data, category_col=category_col, price_col=price_col,
                                     min_count=min_count, top_n=None,)

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

    default_title = f"Ranking de IQR de {_display_label(price_col, data=data)} por {category_col}"
    ax.set_title(_add_top_n_to_title(title or default_title, top_n, "categorías"), fontsize=14, fontweight="bold")
    ax.set_xlabel(f"IQR de {_display_label(price_col, data=data)}")
    ax.set_ylabel(category_col)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return summary.reset_index(drop=True)


# ========================= Modeling Graphics =========================
def plot_regression_metrics(metrics_df, model_name="Linear Regression", metrics=None,
                            n_cols=2, label_col=None):
    """
    Plots selected train and validation metrics for a regression model.

    Arguments:
        metrics_df (pd.DataFrame): model metrics with train_* and val_* columns
        model_name (str): model name shown in the figure title
        metrics (list[str] | None): metrics to plot. Accepted values are
            "mse", "rmse", "mae" and "r2". If None, all available metrics are
            plotted
        n_cols (int): number of subplot columns
        label_col (str | None): column used to label rows when metrics_df has
            more than one row. If None, common labels such as "model",
            "segment" or "variant" are detected automatically

    Returns:
        None
    """
    metric_groups = {
        "mse": ("MSE", "train_mse", "val_mse", FORMAL_COLORS["blue"]),
        "rmse": ("RMSE", "train_rmse", "val_rmse", FORMAL_COLORS["teal"]),
        "mae": ("MAE", "train_mae", "val_mae", FORMAL_COLORS["gold"]),
        "r2": ("R²", "train_r2", "val_r2", FORMAL_COLORS["red"]),
    }

    if metrics is None:
        selected_metrics = list(metric_groups.keys())
    elif isinstance(metrics, str):
        selected_metrics = [metrics.lower()]
    else:
        selected_metrics = [str(metric).lower() for metric in metrics]

    invalid_metrics = [
        metric for metric in selected_metrics
        if metric not in metric_groups
    ]

    if invalid_metrics:
        raise ValueError(f"Unknown metrics: {invalid_metrics}.")

    selected_metrics = [
        metric for metric in selected_metrics
        if metric_groups[metric][1] in metrics_df.columns
        and metric_groups[metric][2] in metrics_df.columns
    ]

    if not selected_metrics:
        raise ValueError("No selected metrics are present in metrics_df.")

    if label_col is None:
        for candidate in ["model", "segment", "variant"]:
            if candidate in metrics_df.columns:
                label_col = candidate
                break

    if len(metrics_df) > 1 and label_col is None:
        row_labels = [f"model_{index + 1}" for index in range(len(metrics_df))]
    elif label_col is None:
        row_labels = None
    else:
        row_labels = metrics_df[label_col].astype(str).tolist()

    n_plots = len(selected_metrics)
    n_cols = min(n_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4 * n_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    fig.suptitle(f"{model_name} - Metrics", fontsize=16, fontweight="bold")

    for ax, metric_key in zip(axes, selected_metrics):
        metric_name, train_col, val_col, color = metric_groups[metric_key]

        if len(metrics_df) == 1:
            x_labels = ["Train", "Validation"]
            values = [metrics_df[train_col].iloc[0], metrics_df[val_col].iloc[0]]
            bars = ax.bar(x_labels, values, color=color, alpha=0.9)

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:,.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            x = np.arange(len(metrics_df))
            width = 0.36
            train_values = metrics_df[train_col].values
            val_values = metrics_df[val_col].values

            train_bars = ax.bar(
                x - width / 2,
                train_values,
                width=width,
                color=color,
                alpha=0.55,
                label="Train",
            )
            val_bars = ax.bar(
                x + width / 2,
                val_values,
                width=width,
                color=color,
                alpha=0.95,
                label="Validation",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(row_labels, rotation=20, ha="right")
            ax.legend()

            for bars, values in [(train_bars, train_values), (val_bars, val_values)]:
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:,.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_title(metric_name, fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def plot_regression_metrics_comparison(metrics_dict, title="Regression Metrics Comparison"):
    """
    Plots train and validation metrics for multiple versions of the same model.

    Arguments:
        metrics_dict (dict): dictionary where keys are model labels and values are metrics dataframes
        title (str): general plot title

    Returns:
        None
    """
    metric_groups = {
        "MSE": ("train_mse", "val_mse"),
        "RMSE": ("train_rmse", "val_rmse"),
        "MAE": ("train_mae", "val_mae"),
        "R²": ("train_r2", "val_r2"),
    }

    colors = [
        FORMAL_COLORS["blue"],
        FORMAL_COLORS["teal"],
        FORMAL_COLORS["gold"],
        FORMAL_COLORS["red"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    fig.suptitle(title, fontsize=16, fontweight="bold")

    model_names = list(metrics_dict.keys())
    x_labels = ["Train", "Validation"]
    x = np.arange(len(x_labels))
    bar_width = 0.8 / len(model_names)

    for ax, (metric_name, (train_col, val_col)) in zip(axes, metric_groups.items()):

        for i, model_name in enumerate(model_names):
            metrics_df = metrics_dict[model_name]

            values = [
                metrics_df[train_col].iloc[0],
                metrics_df[val_col].iloc[0],
            ]

            offset = (i - (len(model_names) - 1) / 2) * bar_width

            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=model_name,
                color=colors[i % len(colors)],
                alpha=0.9,
            )

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:,.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()
