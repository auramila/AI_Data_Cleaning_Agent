import pandas as pd
from report import generate_report

def clean_data(df: pd.DataFrame):
    summary = []
    # Drop duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    summary.append(f"Dropped {dup_count} duplicate rows.")

    # Handle missing values
    missing_count = df.isnull().sum().sum()
    df = df.fillna("MISSING")  # You can later add options to impute
    summary.append(f"Replaced {missing_count} missing values.")

    # Basic outlier detection
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        outliers = df[(df[col] < col_mean - 3*col_std) | (df[col] > col_mean + 3*col_std)]
        summary.append(f"Detected {len(outliers)} outliers in '{col}' (not removed).")

    return df, "\n".join(summary)
