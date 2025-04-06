import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from cleaning import clean_data

TEMP_DF = None

def preview_data(_: str):
    return TEMP_DF.head(5).to_string()

def clean_data_agent(_: str):
    cleaned_df, summary = clean_data(TEMP_DF, missing_strategy="Drop rows with missing data")
    return summary

tools = [
    Tool(name="PreviewData", func=preview_data, description="Preview the uploaded dataset"),
    Tool(name="CleanData", func=clean_data_agent, description="Cleans the dataset and gives a cleaning summary")
]

def run_agent_with_tools(df: pd.DataFrame, user_prompt: str) -> str:
    global TEMP_DF
    TEMP_DF = df

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent.run(user_prompt)
