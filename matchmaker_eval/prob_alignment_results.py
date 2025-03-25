import os
import json
import re
import pandas as pd
from pathlib import Path

# Directory containing the JSON files
exp_dir = "../pitchtempohmm_results"

# Regex pattern to extract perf_name, model, and features
# Assumes filenames end with: _<model>_<features>_results.json
filename_pattern = re.compile(
    r"^(?P<perf_name>.+)_(?P<model>[^_]+)_(?P<features>[^_]+)_results\.json$"
)

records = []

for filepath in Path(exp_dir).rglob("*_results.json"):
    filename = filepath.name
    match = filename_pattern.match(filename)
    if match:
        perf_name = match.group("perf_name")
        model = match.group("model")
        features = match.group("features")

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
        {col: ("sum" if col == "count" else "mean") for col in df_feat.columns if col not in ["perf_name", "model", "features", "dataset"]}
    )
    summary_tables[feature] = grouped

# Display each table
for feature, table in summary_tables.items():
    print(f"\nFeature: {feature}")
    print(table)
