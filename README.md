# 🧼 Data Cleaning Agent

## Introduction

Data Cleaning Agent is built using Streamlit and LangChain. This agent can process CSV or Excel files, detect common issues like missing values or outliers, and provide users with a clean dataset ready for analysis.

An AI-powered Streamlit app that helps you clean messy datasets using:
- ✅ Streamlit for an interactive UI
- 🧠 OpenAI (GPT) to suggest intelligent cleaning steps
- 🛠️ LangChain agents to preview and clean the dataset using tools
- 📊 Visualizations for missing values
- 📥 File download and report generation

## ✨ Features

- Upload CSV or Excel files
- View data preview and missing values
- Choose how to handle missing values (drop, impute, fill)
- Automatically detect duplicates and outliers
- Get AI-generated cleaning suggestions
- Use a LangChain agent to run cleaning tasks
- Download cleaned dataset and summary report

## Technologies Used

- Streamlit
- OpenAI API (langchain)

## Installation and Setup

1. Clone the Repository

```markdown
git clone git@github.com:TuringCollegeSubmissions/aumilas-AE.3.5.git
```

2. Create a Virtual Environment (Optional)

```markdown
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```markdown
pip install -r requirements.txt
```

4. Set Up API Keys

Create a .env file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_key
```

Also, ensure you have your Firebase credentials (firebase_credentials.json) in the root directory.

5. Run the Application

```markdown
streatlit run Cleaning_App.py
```

## Authors

**Aura Milasiute** - [GitHub](https://github.com/auramila)

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

## Acknowledgments

Thank you [Turing College](https://www.turingcollege.com)
