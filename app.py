"""
News Credibility Analyzer — Streamlit Application
---------------------------------------------------
Accepts user text input, applies the trained preprocessing + TF-IDF pipeline,
and predicts credibility (High / Low) with a confidence score.
"""

import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Setup ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained model and TF-IDF vectorizer (cached across sessions)."""
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def download_nltk_data():
    """Download NLTK resources once."""
    for res in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
        nltk.download(res, quiet=True)

def preprocess_text(text: str) -> str:
    """Apply the same preprocessing pipeline used during training."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="News Credibility Analyzer",
    page_icon="📰",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .high-cred {
        background-color: #d4edda;
        border: 1px solid #28a745;
    }
    .low-cred {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
    }
    .cred-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .confidence {
        font-size: 1.1rem;
        color: #555;
    }
    .divider {
        margin: 2rem 0 1rem 0;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── App ──────────────────────────────────────────────────────────────────────

# Initialize resources
download_nltk_data()
model, vectorizer = load_model()

# Header
st.markdown('<h1 class="main-title">📰 News Credibility Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Paste a news article or headline below to check its credibility.</p>', unsafe_allow_html=True)

# Input
user_text = st.text_area(
    "Enter news text",
    height=180,
    placeholder="Paste the news article or headline here …",
    label_visibility="collapsed"
)

# Predict
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

if analyze_btn:
    if not user_text or user_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing …"):
            # 1. Preprocess
            cleaned = preprocess_text(user_text)

            # 2. Vectorize
            features = vectorizer.transform([cleaned])

            # 3. Predict
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[prediction] * 100

            # 4. Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="result-box high-cred">
                    <div class="cred-label" style="color: #28a745;">✅ High Credibility</div>
                    <div class="confidence">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box low-cred">
                    <div class="cred-label" style="color: #dc3545;">⚠️ Low Credibility</div>
                    <div class="confidence">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("Built with Streamlit • Logistic Regression + TF-IDF • Classical ML only")
