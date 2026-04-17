# Agent Module

This directory contains the core logic for the **Milestone 2 Agentic Fact-Checking System**.

It uses a **ReAct** (Reasoning + Acting) architecture powered by LangChain and Google's Gemini 1.5 Flash.

## Files
- `orchestrator.py`: The main LangChain `AgentExecutor`. Defines the prompt and runs the ReAct loop.
- `tools.py`: Defines the tools exposed to the agent:
  - `ml_prescreener`: Hooks into the existing Logistic Regression model from M1.
  - `web_search`: Uses the DuckDuckGo Search package to retrieve live evidence without needing an API key.
- `llm_client.py`: Initializes the Google Gemini connector mapping.
- `rag_pipeline.py`: (Modular capability) Defines FAISS + sentence-transformer retrieval methods.
