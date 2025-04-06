# 🧼 AI-Powered Data Cleaning App

## Introduction

An intelligent Streamlit application that helps you clean messy datasets for analysis. Users can upload CSV or Excel files, visualize issues like missing values and outliers, and clean data using manual tools or AI-powered workflows. Choose between receiving smart suggestions or letting a multi-agent system handle everything automatically.

💡 What It Does

- ✅ Interactive UI built with Streamlit
- 🧠 Uses OpenAI (GPT) to suggest intelligent cleaning steps
- 🛠️ LangChain agents preview, reason, and clean the dataset
- 📊 Visualize missing values instantly
- 🧼 Clean data manually or through agents
- 🧠 Choose from agent personalities (Analyst, Coach, Playful)
- 🔁 Side-by-side before/after comparison
- 📥 Download cleaned data and reports

## Technologies Used

- Streamlit
- OpenAI API
- LangChain
- Pandas

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
