# Veritas: AI Fact-Checker Agent

## Overview
This project is an AI-powered fact-checking system that uses **LangGraph**, **OpenAI GPT-4o**, and **Perplexity AI** to verify factual claims. It builds a verification workflow graph that plans, queries, and adjudicates claims via multiple agents, then displays results in a Streamlit dashboard.

## ⚙️ Features
- ✅ Graph-based LLM workflow via LangGraph
- ✅ Multi-agent fact-checking using GPT-4o and Perplexity
- ✅ Adjudicator node decides final verdict
- ✅ Citations returned in JSON format
- ✅ User-friendly Streamlit dashboard

## Tech Stack
- Python
- LangGraph
- OpenAI (gpt-4o)
- Perplexity (sonar-reasoning)
- Streamlit
- Pydantic

## 🚀 How It Works
1. **Plan**: GPT-4o creates a plan
2. **Parallel Agent Execution**: OpenAI and Perplexity run the plan
3. **Adjudication**: GPT-4o decides if the claim is true/false
4. **Retry**: If verdict is uncertain, retry up to 3 times

## 🛠️ Usage
Set your API keys:
```bash
export OPENAI_API_KEY=your_key
export PERPLEXITY_API_KEY=your_key
```

Then run:
```bash
streamlit run fact-checker.py
```
