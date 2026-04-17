"""
News Credibility Analyzer — Streamlit Application (Milestone 2)
---------------------------------------------------
Accepts user text input.
Tab 1: Fast ML check (Logistic Regression + TF-IDF)
Tab 2: Deep Agentic check (ReAct Agent + DuckDuckGo Web Search)
"""

import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv

load_dotenv()

# Try to import agent (might fail if API keys aren't set)
try:
    from agent.orchestrator import run_agent
    AGENT_AVAILABLE = True
except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

# ── Setup ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_model():
    """Load the trained model and TF-IDF vectorizer (cached across sessions)."""
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def download_nltk_data():
    """Download NLTK resources once."""
    for res in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        nltk.download(res, quiet=True)

def preprocess_text(text: str) -> str:
    """Apply the same preprocessing pipeline used during ML training."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(cleaned_tokens)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="News Credibility Analyzer", page_icon="📰", layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main-title { text-align: center; padding: 0.5rem 0 0.2rem 0; }
    .subtitle { text-align: center; color: #888; font-size: 1.05rem; margin-bottom: 0.5rem; }
    .result-box { padding: 1.5rem; border-radius: 12px; text-align: center; margin-top: 1.5rem; }
    .high-cred { background-color: #172a1e; border: 1px solid #28a745; color: #d4edda; }
    .low-cred { background-color: #3b1619; border: 1px solid #dc3545; color: #f8d7da; }
    .cred-label { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
    .confidence { font-size: 1.1rem; color: #a1a1a1; }
    .divider { margin: 2rem 0 1rem 0; border-top: 1px solid #e0e0e0; }
    
    /* Tweaks for agent trace */
    .agent-trace { background-color: #1a1a1a; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9em; max-height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4;}
</style>
""",
    unsafe_allow_html=True,
)

# ── App ──────────────────────────────────────────────────────────────────────

# Initialize resources
download_nltk_data()
ml_model, ml_vectorizer = load_ml_model()

# Header
st.markdown('<h1 class="main-title">📰 News Credibility Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a news article or headline to check its credibility.</p>', unsafe_allow_html=True)

# Main Input
user_text = st.text_area(
    "Enter news text",
    height=150,
    placeholder="Paste a news article or headline here...",
    label_visibility="collapsed",
)

# Tabs
tab1, tab2 = st.tabs(["⚡ Fast ML Check (Milestone 1)", "🧠 Deep Agent Fact-Check (Milestone 2)"])

# ── Tab 1: ML Check ─────────────────────────────────────────────────────────

with tab1:
    st.write("Uses **Logistic Regression + TF-IDF** (Tested on 38K samples, ~97% Accuracy)")
    
    if st.button("🔍 Run Fast ML Check", key="btn_ml", type="primary", use_container_width=True):
        if not user_text or user_text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Classifying..."):
                cleaned = preprocess_text(user_text)
                features = ml_vectorizer.transform([cleaned])
                prediction = ml_model.predict(features)[0]
                probabilities = ml_model.predict_proba(features)[0]
                confidence = probabilities[prediction] * 100

                if prediction == 1:
                    st.markdown(
                        f"""
                    <div class="result-box high-cred">
                        <div class="cred-label" style="color: #4ade80;">✅ High Credibility</div>
                        <div class="confidence">ML Confidence: {confidence:.1f}%</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="result-box low-cred">
                        <div class="cred-label" style="color: #f87171;">⚠️ Low Credibility</div>
                        <div class="confidence">ML Confidence: {confidence:.1f}%</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )


# ── Tab 2: Deep Agent Check ─────────────────────────────────────────────────

with tab2:
    st.write("Uses a **ReAct Agent** with **Web Search** and **Gemini Flash** LLM to verify factual claims.")
    
    # Check if API key is set
    has_api_key = bool(os.getenv("GOOGLE_API_KEY")) or ("GOOGLE_API_KEY" in st.secrets)
    
    if not has_api_key:
         st.error("Google API key is missing! Please set `GOOGLE_API_KEY` in environment variables or `.env`.")
    else:
        if st.button("🕵️ Run Deep Fact-Check", key="btn_agent", type="primary", use_container_width=True):
            if not user_text or user_text.strip() == "":
                st.warning("⚠️ Please enter some text to verify.")
            else:
                if not AGENT_AVAILABLE:
                    st.error(f"Agent failed to load. Details: {AGENT_ERROR}")
                else:
                    with st.spinner("Agent is planning, searching the web, and reasoning..."):
                        
                        try:
                            verdict = run_agent(user_text)
                            st.success("Analysis Complete!")
                            
                            # Render the Final Answer as Markdown
                            st.markdown("### Agent Findings")
                            st.markdown(verdict)
                            
                        except Exception as e:
                            st.error(f"An error occurred during agent verification: {str(e)}")


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("Capstone Project • Milestone 1: Classical ML • Milestone 2: Agentic GenAI")
