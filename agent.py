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

def suggest_steps(_: str):
    global TEMP_SUGGESTION
    TEMP_SUGGESTION = suggest_cleaning_steps_with_personality(
        TEMP_DF,
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=500,
        system_message="You are a helpful and encouraging data coach helping users clean their dataset."
    )
    return TEMP_SUGGESTION

def decide_cleaning(_: str):
    if not TEMP_SUGGESTION:
        return "No suggestion generated yet."
    return f"Decision: Proceeding with cleaning based on AI suggestion.\n\n{TEMP_SUGGESTION}"

def execute_cleaning(_: str):
    global TEMP_DF, CLEANED_DF, CLEANING_SUMMARY
    cleaned_df, summary = clean_data(TEMP_DF, missing_strategy="Drop rows with missing data")
    CLEANED_DF = cleaned_df
    CLEANING_SUMMARY = summary
    TEMP_DF = cleaned_df
    return summary

multi_tools = [
    Tool(name="SuggestCleaningSteps", func=suggest_steps, description="Suggests steps for cleaning the dataset."),
    Tool(name="DecideCleaningAction", func=decide_cleaning, description="Decides what cleaning actions to take based on suggestion."),
    Tool(name="ExecuteCleaning", func=execute_cleaning, description="Cleans the dataset using standard cleaning functions.")
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
        system_message="You are a helpful and friendly assistant guiding users through cleaning their data."
    )

    return agent.run(prompt)

def get_cleaned_df():
    return CLEANED_DF

def get_cleaning_summary():
    return CLEANING_SUMMARY

def get_original_df():
    return TEMP_DF
