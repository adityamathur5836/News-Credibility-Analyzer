"""
Agent Orchestrator: Combines tools and LLM into a ReAct workflow.
"""

from langchain.agents import create_agent
from agent.llm_client import get_llm
from agent.tools import ml_prescreener, web_search


def get_agent_executor():
    """Configures and runs the agent graph."""
    llm = get_llm()
    # We provide the ML pre-screener and the DuckDuckGo web search
    tools = [ml_prescreener, web_search]

    # Custom instructions for the agent
    prompt = """You are a highly intelligent fact-checking agent designed to evaluate the credibility of news articles.
    
Your goal is to determine if a claim or article is Credible, Suspicious, or Fake based on the evidence.

You have access to tools. Follow this exact workflow:
1. First, use `ml_prescreener` with the exact text to get a baseline ML prediction.
2. Next, extract 1-2 core factual claims from the text.
3. Use `web_search` to find recent facts or debunking articles for these claims.
4. Synthesize all findings into a Final Answer.

Give a structured response using Markdown with: 
**Verdict**: (Credible / Suspicious / Fake)
**ML Confidence**: (From the ML prescreener)
**Agent Confidence**: (Your confidence in the web evidence, e.g., High/Medium/Low)
**Sources Cited**: (List of URLs from web search)
**Detailed Reasoning**: (Step-by-step logic explaining the verdict)"""

    agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
    return agent


def run_agent(text_claim: str):
    executor = get_agent_executor()
    result = executor.invoke({"messages": [{"role": "user", "content": text_claim}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    claim = "NASA faked the moon landing"
    if len(sys.argv) > 1:
        claim = sys.argv[1]
    res = run_agent(claim)
    print("\\n--- FINAL VERDICT ---\\n")
    print(res)
