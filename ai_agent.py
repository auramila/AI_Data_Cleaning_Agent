import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def suggest_cleaning_steps(df: pd.DataFrame) -> str:
    preview = df.head(3).to_csv(index=False)
    prompt = f"""
You are a data cleaning assistant.

Here is a preview of the dataset:
{preview}

Suggest intelligent cleaning steps to improve the dataset.
Include missing value handling, outlier detection, column type corrections, etc.
Be specific and return suggestions as numbered steps.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content
