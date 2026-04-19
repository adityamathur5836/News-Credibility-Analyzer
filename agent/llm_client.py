"""
LLM Client to connect to Groq.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Returns a ChatGroq instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Please set it in .env"
        )

    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.0)
