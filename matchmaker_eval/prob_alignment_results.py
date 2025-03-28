import os
import json
import re
import pandas as pd
from pathlib import Path

# Directory containing the JSON files
exp_dir = "../oltw_arzt_results5/"

# Regex pattern to extract perf_name, model, and features
# Assumes filenames end with: _<model>_<features>_results.json
filename_pattern = re.compile(
    r"^(?P<perf_name>.+)_(?P<model>[^_]+)_(?P<features>[^_]+)_results_perf\.json$"
)

records = []

for filepath in Path(exp_dir).rglob("*_results_perf.json"):
    filename = filepath.name
    match = filename_pattern.match(filename)
    if match:
        perf_name = match.group("perf_name")
        model = match.group("model")

        # Extract features as the third-to-last part of the path
        features = filepath.parts[-2]

        # Extract dataset from the path parts
        parts = filepath.parts
        dataset = None
        for part in parts:
            if part in {"vienna", "batik", "asap"}:
                dataset = part
                break

        with open(filepath, "r") as f:
            data = json.load(f)

        data["perf_name"] = perf_name
        data["model"] = model
        data["features"] = features
        data["dataset"] = dataset

        records.append(data)
    else:
        print(f"Filename did not match pattern: {filename}")

# Create DataFrame
df = pd.DataFrame(records)

# Create summary tables per feature type
summary_tables = {}
for feature in df["features"].unique():
    df_feat = df[df["features"] == feature]
    grouped = df_feat.drop(columns=["perf_name", "model", "features"]).groupby("dataset").agg(
        {col: ("sum" if "count" in col else "mean") for col in df_feat.columns if col not in ["perf_name", "model", "features", "dataset"]}
    )
    summary_tables[feature] = grouped

# Display each table
for feature, table in summary_tables.items():
    print(f"\nFeature: {feature}")
    # Ensure the required columns are present
    required_columns = [
        "mean_perf", "std_perf", "median_perf", "skewness_perf", "kurtosis_perf",
        "50ms_perf", "100ms_perf", "500ms_perf", "1000ms_perf", "2000ms_perf", "count_perf"
    ]
    if all(col in table.columns for col in required_columns):
        for dataset, row in table.iterrows():
            result = (
                f'{row["mean_perf"]:.2f}±{row["std_perf"]:.2f}, '
                f'{row["median_perf"]:.2f}, {row["skewness_perf"]:.2f}, {row["kurtosis_perf"]:.2f}, '
                f'{row["50ms_perf"]:.2f}, {row["100ms_perf"]:.2f}, {row["500ms_perf"]:.2f}, '
                f'{row["1000ms_perf"]:.2f}, {row["2000ms_perf"]:.2f}, {row["count_perf"]:.0f}'
            )
            print(f"Dataset: {dataset}, Results: {result}")
    else:
        print(f"Table for feature '{feature}' is missing required columns.")


if exp_dir == "../oltw_artzt_results/":

    # Aggregate results for "median_perf" across all datasets
    features_of_interest = ["chroma", "mel", "cqt", "lse"]
    median_perf_table = {}

    for feature in features_of_interest:
        if feature in df["features"].unique():
            df_feat = df[df["features"] == feature]
            median_perf_table[feature] = df_feat["median_perf"].mean().round(2)

    # Convert to a DataFrame
    median_perf_df = pd.DataFrame.from_dict(
        median_perf_table, orient="index", columns=["median_perf"]
    )

    # Display the aggregated table
    print("\nAggregated Median Performance Table:")
    print(median_perf_df)
