import streamlit as st
import pandas as pd
from cleaning import clean_data
from report import generate_report

st.set_page_config(page_title="Data Cleaning Agent", layout="wide")

st.title("🧼 Data Cleaning Agent")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df)

        if st.button("🧼 Clean Data"):
            cleaned_df, summary = clean_data(df)
            st.subheader("✅ Cleaned Data")
            st.dataframe(cleaned_df)

            st.subheader("📄 Cleaning Summary")
            st.text(summary)

            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

