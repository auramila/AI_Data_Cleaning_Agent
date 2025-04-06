import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from cleaning import clean_data, visualize_missing_data, generate_report
from ai_agent import suggest_cleaning_steps_with_personality as suggest_cleaning_steps
from agent import run_multi_agent_cleaning, get_cleaned_df, get_cleaning_summary, get_original_df

load_dotenv()

st.set_page_config(page_title="🧼 Multi-Agent Data Cleaning App", layout="wide")
st.title("🧼 AI-Powered Data Cleaning Assistant")

# Sidebar AI config
st.sidebar.header("🤖 AI Settings")
model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 500)

# Agent personality options
personality = st.sidebar.selectbox("🧠 Agent Personality", [
    "Professional Analyst",
    "Friendly Coach",
    "Playful Assistant"
])

# Toggle auto-clean vs manual decision
auto_clean = st.sidebar.toggle("⚙️ Auto-clean mode", value=True)

# Upload file
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("🔍 Raw Dataset Preview")
        st.dataframe(df)

        # Visualization: Missing values
        st.subheader("📊 Missing Values Overview")
        visualize_missing_data(df)

        # Missing value strategy
        strategy = st.radio("Missing Value Strategy", [
            "Impute with mean (numeric) or mode (categorical)",
            "Drop rows with missing data",
            "Replace with 'MISSING'"
        ])

        # Manual Cleaning
        if st.button("🧼 Run Manual Cleaning"):
            cleaned_df, summary = clean_data(df, missing_strategy=strategy)
            st.subheader("✅ Cleaned Data Preview")
            st.dataframe(cleaned_df)
            st.subheader("📋 Cleaning Summary")
            st.text(summary)
            st.download_button("📥 Download Cleaned CSV", cleaned_df.to_csv(index=False), "cleaned_data.csv")
            report = generate_report(summary)
            st.download_button("📄 Download Report", report.encode("utf-8"), "cleaning_report.txt")

        # AI Suggestions
        if st.button("🧠 Get AI Cleaning Suggestions"):
            with st.spinner("Thinking..."):
                if personality == "Professional Analyst":
                    tone = "You are a highly experienced data analyst."
                elif personality == "Friendly Coach":
                    tone = "You are a supportive and friendly coach helping someone clean their dataset."
                else:
                    tone = "You are a playful assistant who enjoys cleaning up messy data!"

                suggestions = suggest_cleaning_steps(df, model, temperature, max_tokens, system_message=tone)

            st.subheader("💡 AI Suggestions")
            st.markdown(suggestions)

        # Multi-Agent Section
        st.subheader("🧪 Multi-Agent Auto Clean")
        user_prompt = st.text_input("What would you like the AI to do?", value="Suggest and clean this dataset using multi-agent reasoning.")

        if st.button("🤖 Run Multi-Agent Cleaning"):
            with st.spinner("Running LangChain Multi-Agent..."):
                try:
                    run_prompt = user_prompt
                    if auto_clean:
                        run_prompt += " Please clean the dataset automatically."
                    else:
                        run_prompt += " Just suggest and explain cleaning steps, don't clean automatically."

                    result = run_multi_agent_cleaning(df, run_prompt)
                    st.subheader("📋 Multi-Agent Result")
                    st.markdown(result)

                    cleaned_df = get_cleaned_df()
                    original_df = get_original_df()
                    summary = get_cleaning_summary()

                    if cleaned_df is not None:
                        st.subheader("🪞 Side-by-Side Comparison")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Before Cleaning**")
                            st.dataframe(original_df.head(20))
                        with col2:
                            st.markdown("**After Cleaning**")
                            st.dataframe(cleaned_df.head(20))

                        st.download_button("📥 Download Agent-Cleaned CSV", cleaned_df.to_csv(index=False), "agent_cleaned_data.csv")
                    else:
                        st.warning("⚠️ No cleaned data was returned by the agent.")

                    if summary:
                        st.subheader("📄 Agent Cleaning Summary")
                        st.text(summary)

                except Exception as e:
                    st.error(f"Agent error: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")
