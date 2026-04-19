# Intelligent News Credibility Analyzer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Milestone2](https://img.shields.io/badge/Agentic_AI-LangChain-green)
![LLM](https://img.shields.io/badge/LLM-Gemini_1.5_Flash-orange)

A two-phase capstone project moving from classical Machine Learning to a modern, Agentic AI retrieval-augmented verification system.

## 🚀 Two-Phase Architecture

This project evaluates the credibility of news content using two radically different paradigms, available side-by-side in the Streamlit app.

### Milestone 1: Classical ML (Statistical Pre-screening)
- **Methodology**: TF-IDF Vectorization -> Logistic Regression
- **Training Data**: 38,644 cleaned articles (True vs Fake)
- **Performance**: ~97% Accuracy
- **Pros**: Lightning fast, offline, reliable statistical baseline.
- **Cons**: Cannot fact-check novel claims not in its training distribution.

### Milestone 2: Agentic AI Fact-Checking (RAG + Web Search)
- **Methodology**: LangChain ReAct Agent loop using Groq Llama 3 1.5 Flash.
- **Agent Workflow**:
  1. Calls the M1 ML model for a quick statistical baseline.
  2. Extracts key factual claims using the LLM.
  3. Uses **DuckDuckGo Web Search** tools to retrieve recent evidence.
  4. Synthesizes findings using the LLM and outputs a grounded Verdict + Confidence + Cited Sources.
- **Pros**: Handles zero-shot claims, grounds answers in current factual data, highly interpretable.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Agent Framework** | LangChain (`create_react_agent`) |
| **LLM Backend** | Groq Llama 3 1.5 Flash |
| **Web Search capability**| DuckDuckGo Search API (`duckduckgo-search`) |
| **ML Models (M1)** | Logistic Regression, Decision Trees, Scikit-Learn |
| **NLP Pipeline** | NLTK (Lemmatization), TF-IDF |
| **UI Framework** | Streamlit |

---

## 💻 Local Setup & Execution

### 1. Prerequisites
You need Python 3.8+ and a Google API key for the Gemini LLM.
Get a free key here: [Groq Console](https://console.groq.com/keys)

### 2. Environment Variables
Clone the repo and create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Edit `.env` and add your key: `GROQ_API_KEY=your_key_here`

### 3. Install Dependencies
Create a virtual environment and install the required modules:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
This will launch a local server, opening a two-tab UI where you can perform Fast ML checks or Deep Agentic Fact-Checks.

---

## 👥 Team
- Aditya Mathur
- Om Kar Shukla
- Yachna Khanna

> **Built for the University End-Semester Capstone Project**
