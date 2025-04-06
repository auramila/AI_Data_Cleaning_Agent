import streamlit as st
import pandas as pd
from cleaning import clean_data
from ai_agent import suggest_cleaning_steps
from dotenv import load_dotenv
import os

load_dotenv()  # Load OpenAI API key from .env

st.set_page_config(page_title="🧼 Data Cleaning Agent", layout="wide")

st.title("🧼 Data Cleaning Agent")
st.markdown("Upload a dataset, clean it automatically, or ask the AI assistant for smart suggestions.")

uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df)

        # Button: AI Suggestions
        if st.button("🤖 Suggest Cleaning Steps with AI"):
            with st.spinner("Asking the AI assistant..."):
                suggestions = suggest_cleaning_steps(df)
            st.subheader("🧠 AI Suggestions")
            st.markdown(suggestions)

        # Button: Run Cleaning
        if st.button("🧼 Clean Data"):
            cleaned_df, summary = clean_data(df)

            st.subheader("✅ Cleaned Data")
            st.dataframe(cleaned_df)

            st.subheader("📄 Cleaning Summary")
            st.text(summary)

            # Download cleaned file
            cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned CSV",
                data=cleaned_csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.info("Please upload a CSV or Excel file to begin.")
