import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def suggest_cleaning_steps(df: pd.DataFrame, model: str, temperature: float, max_tokens: int) -> str:
    preview = df.head(3).to_csv(index=False)

    prompt = f"""
You are a data cleaning assistant.
Here is a preview of a dataset:
{preview}

Suggest intelligent and detailed cleaning steps to improve the dataset.
Include handling for:
- missing values
- outlier detection
- column renaming or standardization
- data type corrections (e.g. convert to date, int)
- irrelevant or low-variance column removal

Respond with clear numbered steps and a short explanation for each.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert data analyst helping with data cleaning."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content
