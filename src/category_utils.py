from pathlib import Path
import pandas as pd


def get_project_root():
    """
    Finds the project root from the current working directory.

    Arguments:
        None

    Returns:
        Path: project root path
    """
    current_path = Path.cwd().resolve()

    if current_path.name == "notebooks":
        return current_path.parent

    return current_path


def load_dataset(file_name="pf_suvs.csv"):
    """
    Loads the SUV dataset.

    Arguments:
        file_name (str): dataset file name inside the data folder

    Returns:
        pd.DataFrame: loaded dataset without generated index columns
    """
    project_root = get_project_root()
    data_path = project_root / "data" / file_name

    df = pd.read_csv(data_path)

    # Remove CSV-generated index columns.
    index_columns = [
        column for column in df.columns
        if column.startswith("Unnamed:")
    ]

    return df.drop(columns=index_columns)


def get_categorical_columns(df, ignored_columns=None):
    """
    Gets categorical columns from a DataFrame.

    Arguments:
        df (pd.DataFrame): dataset to analyze
        ignored_columns (list[str] | None): columns to exclude

    Returns:
        list[str]: categorical column names
    """
    ignored_columns = ignored_columns or []

    return [
        column
        for column in df.select_dtypes(include=["object", "category", "bool"]).columns
        if column not in ignored_columns
    ]


def build_category_table(df, categorical_columns=None):
    """
    Builds a table with all categories and their frequencies.

    Arguments:
        df (pd.DataFrame): dataset containing categorical features
        categorical_columns (list[str] | None): columns to summarize

    Returns:
        pd.DataFrame: category counts and percentages by feature
    """
    if categorical_columns is None:
        categorical_columns = get_categorical_columns(df)

    rows = []

    for column in categorical_columns:
        counts = df[column].fillna("Missing").astype(str).value_counts(dropna=False)
        total = counts.sum()

        for rank, (category, count) in enumerate(counts.items(), start=1):
            rows.append({
                "feature": column,
                "rank": rank,
                "category": category,
                "count": count,
                "percentage": round(count / total * 100, 2),
            })

    return pd.DataFrame(rows)


def build_category_summary(df, categorical_columns=None):
    """
    Builds a summary with the number of categories per categorical feature.

    Arguments:
        df (pd.DataFrame): dataset containing categorical features
        categorical_columns (list[str] | None): columns to summarize

    Returns:
        pd.DataFrame: number of unique categories by feature
    """
    if categorical_columns is None:
        categorical_columns = get_categorical_columns(df)

    summary = (
        df[categorical_columns]
        .nunique(dropna=False)
        .sort_values(ascending=False)
        .reset_index()
    )

    summary.columns = ["feature", "unique_categories"]

    return summary


def save_category_audit_report(category_summary, category_table, output_path="html/category_audit_report.html",):
    """
    Saves a complete HTML report with all categorical values.

    Arguments:
        category_summary (pd.DataFrame): summary of unique categories by feature
        category_table (pd.DataFrame): full category frequency table
        output_path (str): report path relative to the project root

    Returns:
        Path: saved report path
    """
    project_root = get_project_root()
    report_path = project_root / output_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 32px; }",
        "h1 { color: #1F4E79; }",
        "h2 { margin-top: 0; color: #1F4E79; }",
        ".summary-table { border-collapse: collapse; width: auto; margin-bottom: 32px; }",
        ".feature-grid { display: grid; grid-template-columns: repeat(2, minmax(360px, max-content)); gap: 28px; align-items: start; }",
        ".feature-card { width: fit-content; max-width: 100%; overflow-x: auto; }",
        "table { border-collapse: collapse; width: auto; margin-bottom: 12px; }",
        "th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; white-space: nowrap; }",
        "th { background-color: #E9ECEF; text-align: left; }",
        "tr:nth-child(even) { background-color: #F8F9FA; }",
        ".category-table td:nth-child(2) { max-width: 360px; white-space: normal; word-break: break-word; }",
        "@media (max-width: 900px) { .feature-grid { grid-template-columns: 1fr; } }",
        "</style></head><body>",
        "<h1>Categorical Features Audit</h1>",
        "<p>Full list of categories, counts, and percentages by feature.</p>",
        "<h2>Summary</h2>",
        category_summary.to_html(index=False, classes="summary-table"),
        "<div class='feature-grid'>",
    ]

    for feature in category_summary["feature"]:
        feature_table = (
            category_table[category_table["feature"] == feature]
            [["rank", "category", "count", "percentage"]]
            .copy()
        )

        html_parts.append("<div class='feature-card'>")
        html_parts.append(f"<h2>{feature}</h2>")
        html_parts.append(feature_table.to_html(index=False, classes="category-table"))
        html_parts.append("</div>")

    html_parts.extend([
        "</div>",
        "</body></html>",
    ])

    report_path.write_text("\n".join(html_parts), encoding="utf-8")

    return report_path
