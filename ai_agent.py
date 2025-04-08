import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def suggest_cleaning_steps_with_personality(
    df: pd.DataFrame,
    model: str,
    temperature: float,
    max_tokens: int,
    system_message: str = "You are an expert data analyst helping with data cleaning.",
    frequency_penalty: float = 0.0
) -> tuple[str, dict]:
    preview = df.head(3).to_csv(index=False)

    prompt = f"""
Here is a preview of a dataset:
{preview}

Suggest intelligent and detailed cleaning steps to improve the dataset. Consider:
- handling missing values
- detecting and flagging outliers
- standardizing column names
- converting data types
- removing irrelevant or low-variance columns

Be clear, professional, and concise.
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )

    suggestion = response.choices[0].message.content
    usage = response.usage.model_dump() if hasattr(response, "usage") else None

    return suggestion, usage
