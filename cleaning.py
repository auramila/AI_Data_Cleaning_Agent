import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def clean_data(df: pd.DataFrame, missing_strategy: str = "Drop rows with missing data"):
    summary = []

    missing_count = df.isnull().sum().sum()

    if missing_strategy == "Drop rows with missing data":
        df = df.dropna()
        summary.append(f"🗑 Dropped {missing_count} missing values.")
    elif missing_strategy == "Impute with mean (numeric) or mode (categorical)":
        for col in df.columns:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
        summary.append(f"🧠 Imputed {missing_count} missing values.")
    else:
        df = df.fillna("MISSING")
        summary.append(f"🔄 Replaced {missing_count} missing values with 'MISSING'.")

    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    summary.append(f"🚮 Removed {dup_count} duplicate rows.")

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = z_scores.abs() > 3
        summary.append(f"⚠️ Found {outliers.sum()} outliers in '{col}' (not removed).")

    return df, "\n".join(summary)

def generate_report(summary: str) -> str:
    return f"🧼 Data Cleaning Report\n\n{summary}\n\nThank you for using the AI Assistant!"

def visualize_missing_data(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.bar_chart(missing)
    else:
        st.success("✅ No missing values detected.")
