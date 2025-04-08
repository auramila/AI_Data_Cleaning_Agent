import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from cleaning import clean_data
from ai_agent import suggest_cleaning_steps_with_personality

TEMP_DF = None
TEMP_SUGGESTION = None
CLEANED_DF = None
CLEANING_SUMMARY = None

AGENT_LOG = {
    "assessment": "",
    "suggestion": "",
    "execution": ""
}

def assess_data(_: str):
    global TEMP_DF, AGENT_LOG
    issues = []
    missing = TEMP_DF.isnull().sum().sum()
    if missing > 0:
        issues.append(f"🔎 Detected {missing} missing values")
    dups = TEMP_DF.duplicated().sum()
    if dups > 0:
        issues.append(f"♻️ {dups} duplicate rows found")
    AGENT_LOG["assessment"] = "\n".join(issues) if issues else "✅ No major data issues detected."
    return AGENT_LOG["assessment"]

def suggest_steps(_: str):
    global TEMP_DF, TEMP_SUGGESTION, AGENT_LOG
    TEMP_SUGGESTION = suggest_cleaning_steps_with_personality(
        TEMP_DF,
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=500,
        system_message="You are a helpful and encouraging data coach helping users clean their dataset."
    )[0]
    AGENT_LOG["suggestion"] = TEMP_SUGGESTION
    return TEMP_SUGGESTION

def execute_cleaning(_: str):
    global TEMP_DF, CLEANED_DF, CLEANING_SUMMARY, AGENT_LOG, TEMP_SUGGESTION
    suggestion_note = "\n\n📝 AI suggestions were considered during cleaning, but only summarized here for brevity."
    cleaned_df, summary = clean_data(
        TEMP_DF,
        missing_strategy="Drop rows with missing data",
        standardize_columns=True,
        correct_dtypes=True,
        remove_low_variance=True
    )
    summary += suggestion_note
    CLEANED_DF = cleaned_df
    CLEANING_SUMMARY = summary
    TEMP_DF = cleaned_df
    AGENT_LOG["execution"] = summary
    return summary

multi_tools = [
    Tool(name="AssessDataQuality", func=assess_data, description="Evaluates the dataset for missing values, duplicates, and potential issues."),
    Tool(name="SuggestCleaningSteps", func=suggest_steps, description="Uses GPT to suggest intelligent data cleaning steps."),
    Tool(name="ExecuteCleaning", func=execute_cleaning, description="Executes cleaning based on AI-generated suggestions.")
]

def run_multi_agent_cleaning(df: pd.DataFrame, prompt: str) -> str:
    global TEMP_DF, TEMP_SUGGESTION
    TEMP_DF = df.copy()
    TEMP_SUGGESTION = None

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    agent = initialize_agent(
        tools=multi_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message="You are a friendly multi-agent orchestrator that coordinates AI tools for data cleaning."
    )

    return agent.run(prompt)

def get_cleaned_df():
    return CLEANED_DF

def get_cleaning_summary():
    global AGENT_LOG
    return f"--- Assessment ---\n{AGENT_LOG['assessment']}\n\n--- Suggestions ---\n{AGENT_LOG['suggestion']}\n\n--- Cleaning Performed ---\n{AGENT_LOG['execution']}"

def get_original_df():
    return TEMP_DF
