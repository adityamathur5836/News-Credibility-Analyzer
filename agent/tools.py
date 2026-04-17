"""
Tools for the ReAct Agent.
Contains the ML Pre-screener, Web Search, and RAG retrieval mechanisms.
"""
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from duckduckgo_search import DDGS
import joblib
import os
from langchain.tools import tool

# ML Pre-screener tool
def preprocess_text(text: str) -> str:
    """Apply the same preprocessing pipeline used during training."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words("english"))
        
    lemmatizer = WordNetLemmatizer()
    try:
        cleaned_tokens = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]
    except:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        cleaned_tokens = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

    return " ".join(cleaned_tokens)

@tool
def ml_prescreener(text: str) -> str:
    """
    Use this tool FIRST to get a fast first-pass credibility verdict from a Logistic Regression model.
    Input: The full text of the news claim or article.
    Output: A string indicating the ML prediction (High Credibility or Low Credibility) and its confidence score.
    """
    try:
        # Assumes this is run from the project root
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        return f"Error loading ML model. Proceed with deeper agentic search. Details: {str(e)}"
        
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        cleaned = preprocess_text(text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100
        
        if prediction == 1:
            return f"ML Model Verdict: High Credibility (Confidence: {confidence:.1f}%)"
        else:
            return f"ML Model Verdict: Low Credibility (Confidence: {confidence:.1f}%)"
    except Exception as e:
         return f"ML processing error. Details: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Use this tool to search the internet for recent news and evidence.
    Input: Specific search query based on the key claims in the text.
    Output: Relevant snippets and source URLs.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No search results found."
            formatted_results = "\\n".join([f"Source: {res.get('href')} \\nSnippet: {res.get('body')}" for res in results])
            return formatted_results
    except Exception as e:
        return f"Web search failed. Details: {str(e)}"
