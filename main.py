import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv
import os

# Local modules
from cleaning import clean_data, visualize_missing_data, generate_report
from ai_agent import suggest_cleaning_steps_with_personality as suggest_cleaning_steps
from agent import (
    run_multi_agent_cleaning,
    get_cleaned_df,
    get_cleaning_summary,
    get_original_df
)

load_dotenv()

st.set_page_config(page_title="🧼 Multi-Agent Data Cleaning App", layout="wide")
st.title("🧼 AI-Powered Data Cleaning Assistant")

# ----------------------------
# Helper: Visualize outliers using Altair
# ----------------------------
def visualize_outliers(df: pd.DataFrame, title="Outlier Distributions"):
    """Show boxplots for numeric columns to illustrate outliers."""
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        st.info("No numeric columns available for outlier visualization.")
        return

    st.markdown(f"### {title}")
    for col in numeric_cols:
        st.write(f"**Boxplot for `{col}`**")
        chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(x=alt.X(col, type='quantitative'))
            .properties(width=600)
        )
        st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Sidebar AI Settings
# ----------------------------
st.sidebar.header("🤖 AI Settings")
model = st.sidebar.selectbox("Model", [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-1106-preview"
])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 500)
frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0)

personality = st.sidebar.selectbox("🧠 Agent Personality", [
    "Professional Analyst",
    "Friendly Coach",
    "Playful Assistant"
])

# ----------------------------
# 1️⃣ Upload & Preview
# ----------------------------
st.header("1️⃣ Upload File and See Preview")
uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False)

if uploaded_file:
    # Basic file checks
    if uploaded_file.size > 1_000_000_000:
        st.error("🚫 File too large. Please upload under 1GB.")
        st.stop()

    if not uploaded_file.name.lower().endswith(('.csv', '.xlsx')):
        st.error("🚫 Invalid file type. Only CSV or Excel supported.")
        st.stop()

    if uploaded_file.size == 0:
        st.error("🚫 Uploaded file is empty.")
        st.stop()

    try:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            chunks = pd.read_csv(uploaded_file, chunksize=100000, engine='python')
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_excel(uploaded_file)

        # Show data
        st.subheader("🔍 Raw Dataset Preview")
        st.dataframe(df)

        # Missing Values
        st.subheader("📊 Missing Values Overview")
        visualize_missing_data(df)

        # Outlier Distribution (Raw)
        with st.expander("📈 View Outlier Distributions (Raw)", expanded=False):
            visualize_outliers(df, title="Outlier Distributions (Raw Data)")

        # ----------------------------
        # 2️⃣ AI Suggestions
        # ----------------------------
        st.header("2️⃣ Get AI Suggestions")
        if st.button("🧠 Get AI Cleaning Suggestions"):
            with st.spinner("Thinking..."):
                # Tone
                if personality == "Professional Analyst":
                    tone = "You are a highly experienced data analyst."
                elif personality == "Friendly Coach":
                    tone = "You are a supportive and friendly coach helping someone clean their dataset."
                else:
                    tone = "You are a playful assistant who enjoys cleaning up messy data!"

                suggestions, usage = suggest_cleaning_steps(
                    df,
                    model,
                    temperature,
                    max_tokens,
                    system_message=tone,
                    frequency_penalty=frequency_penalty
                )

            st.subheader("💡 AI Suggestions")
            st.markdown(suggestions)
            if usage:
                # Show token usage & cost
                cost_per_1k = {
                    'gpt-3.5-turbo': 0.0015,
                    'gpt-3.5-turbo-16k': 0.003,
                    'gpt-4': 0.03,
                    'gpt-4-1106-preview': 0.01
                }.get(model, 0.0015)
                total_cost = (usage['total_tokens'] / 1000) * cost_per_1k
                st.caption(
                    f"🔢 Tokens used: {usage['total_tokens']} "
                    f"(Prompt: {usage['prompt_tokens']}, Completion: {usage['completion_tokens']})"
                )
                st.caption(f"💸 Estimated Cost: ${total_cost:.4f} USD")

        # ----------------------------
        # 3 & 4: Tabs for Multi-Agent vs Manual
        # ----------------------------
        st.header("3️⃣ & 4️⃣ Choose Your Cleaning Mode")
        tabs = st.tabs(["🤖 AI Multi-Agent Clean", "🛠️ Manual Cleaning"])

        # Tab 0: AI Multi-Agent
        with tabs[0]:
            st.subheader("🤖 Multi-Agent Auto Clean")
            user_prompt = st.text_input(
                "What would you like the AI to do?",
                value="Suggest and clean this dataset using multi-agent reasoning."
            )

            if st.button("▶️ Run Multi-Agent Cleaning"):
                with st.spinner("Running LangChain Multi-Agent..."):
                    try:
                        run_prompt = user_prompt
                        run_prompt += " First, suggest and explain cleaning steps. Then clean the dataset automatically."
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

                            # Outlier distribution post multi-agent
                            with st.expander("📈 Outlier Distributions (Multi-Agent Cleaned)", expanded=False):
                                visualize_outliers(cleaned_df, "Outlier Dist. (Post Multi-Agent)")

                            st.download_button(
                                "📥 Download Agent-Cleaned CSV",
                                cleaned_df.to_csv(index=False),
                                "agent_cleaned_data.csv"
                            )
                        else:
                            st.warning("⚠️ No cleaned data was returned by the agent.")

                        if summary:
                            st.subheader("📄 Agent Cleaning Summary")
                            st.text(summary.replace(
                                "\n\n💡 The AI followed the cleaning plan it suggested earlier.",
                                "[AI plan omitted for brevity]"
                            ))

                    except Exception as e:
                        st.error(f"Agent error: {e}")

        # Tab 1: Manual
        with tabs[1]:
            st.subheader("🛠️ Manual Cleaning with Configuration")
            strategy = st.radio("Missing Value Strategy", [
                "Impute with mean (numeric) or mode (categorical)",
                "Drop rows with missing data",
                "Replace with 'MISSING'"
            ])

            st.markdown("### 🧹 Advanced Cleaning Options")
            standardize_columns = st.checkbox("Standardize column names", value=True)
            correct_dtypes = st.checkbox("Correct data types (strings to dates/numbers)", value=True)
            remove_low_variance = st.checkbox("Remove columns with very low variance", value=True)

            if st.button("🧼 Run Manual Cleaning"):
                cleaned_df, summary = clean_data(
                    df,
                    missing_strategy=strategy,
                    standardize_columns=standardize_columns,
                    correct_dtypes=correct_dtypes,
                    remove_low_variance=remove_low_variance
                )
                st.subheader("✅ Cleaned Data Preview")
                st.dataframe(cleaned_df)

                with st.expander("📈 Outlier Distributions (Manually Cleaned)", expanded=False):
                    visualize_outliers(cleaned_df, "Outlier Dist. (Post Manual Cleaning)")

                st.subheader("📋 Cleaning Summary")
                st.text(summary)
                st.download_button(
                    "📥 Download Cleaned CSV",
                    cleaned_df.to_csv(index=False),
                    "cleaned_data.csv"
                )
                report = generate_report(summary)
                st.download_button("📄 Download Report", report.encode("utf-8"), "cleaning_report.txt")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")
