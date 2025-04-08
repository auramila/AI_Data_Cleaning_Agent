# 🧼 AI-Powered Data Cleaning App

## Introduction

An intelligent Streamlit application for cleaning and preprocessing messy datasets. Use either an AI multi-agent approach (powered by OpenAI + LangChain) or manual cleaning with advanced options. Visualize missing values, outliers, and easily download cleaned results.

✨ Features

- CSV/Excel File Upload – Supports large files up to ~1GB-
- AI Settings – Tune model, temperature, token limits, and frequency penalty
- Personality Tones – Switch between “Professional Analyst,” “Friendly Coach,” or “Playful Assistant”
- AI Suggestions – Summarizes recommended cleaning steps, estimates token usage + cost
- Multi-Agent Auto Clean – Automatic pipeline to assess data, propose steps, and clean
- Manual Cleaning – Impute or drop missing values, rename columns, fix dtypes, remove low variance columns
- Before/After Comparison – See data side by side, track changes made
- Downloadable – Export cleaned dataset and summary report
- Missing Values – Quick bar chart overview
- Agent Summary – Check the AI reasoning and final steps

## Technologies Used

- Streamlit – Fast UI
- LangChain – Multi-agent orchestrations
- OpenAI GPT – AI suggestions + auto cleaning logic
- Pandas – Data manipulation
- Altair – Outlier visualizations

## Installation and Setup

1. Clone the Repository

```markdown
git clone git@github.com:TuringCollegeSubmissions/aumilas-AE.3.5.git
```

2. Install [Poetry](https://python-poetry.org/docs/#installation)

3. Install dependencies

```markdown
poetry install
```

4. Set Up API Keys

Create a .env file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_key
```

5. Run the Application

```markdown
poetry run streamlit run main.py
```

## Authors

**Aura Milasiute** - [GitHub](https://github.com/auramila)

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

## Acknowledgments

Thank you [Turing College](https://www.turingcollege.com)
