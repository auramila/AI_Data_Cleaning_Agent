import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from cleaning import clean_data, visualize_missing_data, generate_report
from ai_agent import suggest_cleaning_steps
from agent import run_agent_with_tools

# Load environment variables (like OpenAI key)
load_dotenv()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="\U0001f9fc Data Cleaning Agent", layout="wide")
st.title("\U0001f9fc Smart Data Cleaning Agent")
st.markdown("Upload your dataset, clean it automatically, or let the AI agent help you!")

# --- Sidebar: AI Configuration ---
st.sidebar.header("\U0001f916 AI Settings")
model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 500)

# --- File Upload ---
uploaded_file = st.file_uploader("\U0001f4c1 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.size > 100 * 1024 * 1024:  # 100MB
            st.error("❌ File too large. Please upload a file under 100MB.")
            st.stop()

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("\U0001f50d Raw Data Preview")
        st.dataframe(df)

        # --- Visualization: Missing Values ---
        st.subheader("\U0001f4ca Missing Values Overview")
        visualize_missing_data(df)

        # --- Missing Value Strategy ---
        strategy = st.radio(
            "How should we handle missing values?",
            options=[
                "Impute with mean (numeric) or mode (categorical)",
                "Drop rows with missing data",
                "Replace with 'MISSING'",
            ],
        )

        # --- AI Suggestions ---
        if st.button("\U0001f9e0 Suggest Cleaning Steps with AI"):
            with st.spinner("Thinking..."):
                suggestions = suggest_cleaning_steps(df, model, temperature, max_tokens)
            st.subheader("\U0001f4a1 AI Suggestions")
            st.markdown(suggestions)

        # --- Clean Data ---
        if st.button("\U0001f9fc Run Basic Cleaning"):
            cleaned_df, summary = clean_data(df, missing_strategy=strategy)
            st.subheader("\u2705 Cleaned Data")
            st.dataframe(cleaned_df)
            st.subheader("\U0001f4cb Cleaning Report")
            st.text(summary)
            st.download_button("Download Cleaned CSV", cleaned_df.to_csv(index=False), "cleaned_data.csv")
            report = generate_report(summary)
            st.download_button("Download Report", report.encode("utf-8"), "cleaning_report.txt")

        # --- LangChain Agent ---
        st.subheader("\U0001f9ea LangChain Agent")
        user_prompt = st.text_input("Ask the agent (e.g., 'Clean this dataset')", value="Clean this dataset")

        if st.button("\U0001f916 Run LangChain Agent"):
            with st.spinner("LangChain Agent running..."):
                try:
                    result = run_agent_with_tools(df, user_prompt)
                    st.subheader("\U0001f4cc Agent Result")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Agent failed. Reason: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("Upload a valid CSV or Excel file to begin.")
