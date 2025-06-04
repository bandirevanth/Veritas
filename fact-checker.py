import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI  
from langchain_community.chat_models.perplexity import ChatPerplexity
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError
from typing import TypedDict, List, Literal
import os, json, logging

logging.basicConfig(level=logging.INFO)


# (Section 1: Set API Keys and Begin LLMs) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
if not PERPLEXITY_API_KEY:
    raise RuntimeError("PERPLEXITY_API_KEY environment variable is not set.")

planner_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
adjudicator_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)

openai_agent_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
tool = {"type": "web_search_preview"}
openai_agent_llm = openai_agent_llm.bind_tools([tool])

perplexity_agent_llm = ChatPerplexity(model="sonar-reasoning", temperature=0.2, pplx_api_key=PERPLEXITY_API_KEY)


# (Section 2: Define Pydantic Model and State Structure) ----------
class AdjudicationResult(BaseModel):
    final_verdict: Literal["Claim is True", "Claim is False", "Re-run verification"]
    evidence_summary: str = ""
    citations: dict = {}

class FactCheckState(TypedDict):
    query: str
    plan: str
    perplexity_result: str
    openai_result: str
    adjudicator_result: str
    final_verdict: str
    attempts: int
    evidence_summary: str
    messages: List[str]


# (Section 3: Define Node Functions) ----------
communal_prompt = (
    "When providing citations, return them in JSON format under the key 'citations'. "
    "Critically evaluate sources when coming to a final verdict. "
    "Even sources that generally have a track-record of accuracy or legitimacy or authority are often biased. "
    "Evaluate True and False literally. A claim that is a metaphor or an exaggeration is not true. "
    "You must also use a counter-source to fully evaluate and understand whether a claim is true or false."
)

def plan_fact_check(state: FactCheckState) -> FactCheckState:
    prompt = f"Create a simple plan to fact-check: '{state['query']}'"
    try:
        plan = planner_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        logging.error(f"Error in planner_llm: {e}")
        plan = "Unable to create plan due to error."
    state["plan"] = plan
    state["messages"].append("Plan: " + plan)
    return state

def run_perplexity_agent(state: FactCheckState) -> FactCheckState:
    prompt = f"Using this plan, fact-check the claim:\n{state['plan']}\n{communal_prompt}"
    try:
        response = perplexity_agent_llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()
    except Exception as e:
        logging.error(f"Error in perplexity_agent_llm: {e}")
        result = "Error during Perplexity agent invocation."
    state["perplexity_result"] = result
    state["messages"].append("---------- Perplexity: " + result)
    return state

def run_openai_agent(state: FactCheckState) -> FactCheckState:
    prompt = f"Using this plan, fact-check the claim:\n{state['plan']}\n{communal_prompt}"
    try:
        responses = openai_agent_llm.invoke([HumanMessage(content=prompt)])
        response = responses[0] if isinstance(responses, list) and responses else responses
        content = response.content
        result = "\n".join(str(item) for item in content) if isinstance(content, list) else content
        result = result.strip() if isinstance(result, str) else str(result)
    except Exception as e:
        logging.error(f"Error in openai_agent_llm: {e}")
        result = "Error during OpenAI agent invocation."
    state["openai_result"] = result
    state["messages"].append("---------- OpenAI result: " + result)
    return state

def adjudicate(state: FactCheckState) -> FactCheckState:
    prompt = (
        "You are a fact-check adjudicator. Two agents provided these outputs\n"
        f"following these criteria: {communal_prompt}.\n"
        "If there is no agreement in true/true or false/false,"
        "you must send back the task to the agents to re-run verification.\n"
        "Here are the results of the Perplexity and OpenAI Agents:\n"
        f"Perplexity: {state['perplexity_result']}\n"
        f"OpenAI: {state['openai_result']}\n"
        "Return a JSON object with keys:\n"
        '"final_verdict": "Claim is True"|"Claim is False"|"Re-run verification",\n'
        '"evidence_summary": "Summary",\n'
        '"citations": {"perplexity": [], "openai": []}'
    )
    try:
        raw = adjudicator_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        raw = raw.replace("```json", "").replace("```", "")
        adjud = AdjudicationResult.model_validate_json(raw)
        verdict = adjud.final_verdict
        evidence = adjud.evidence_summary
    except ValidationError as ve:
        logging.error(f"ValidationError in adjudication result: {ve}")
        verdict, evidence = "Re-run verification", ""
        raw = "{}"
    except Exception as e:
        logging.error(f"Error in adjudicator_llm invocation: {e}")
        verdict, evidence = "Re-run verification", ""
        raw = "{}"

    state["adjudicator_result"] = raw
    state["final_verdict"] = verdict
    state["evidence_summary"] = evidence  
    state["messages"].append("---------- Adjudicator: " + raw)
    state["attempts"] += 1
    return state

def needs_recheck(state: FactCheckState) -> bool:
    return state["final_verdict"] == "Re-run verification"


# (Section 4: Build and Compile the LangGraph) ----------
graph = (
    StateGraph(FactCheckState)
    .add_node("plan_node", plan_fact_check)
    .add_node("perplexity", run_perplexity_agent)
    .add_node("openai", run_openai_agent)
    .add_node("adjudicate", adjudicate)
    .add_edge(START, "plan_node")
    .add_edge("plan_node", "perplexity")
    .add_edge("perplexity", "openai")
    .add_edge("openai", "adjudicate")
    .add_conditional_edges(
        "adjudicate",
        lambda s: "recheck" if needs_recheck(s) and s["attempts"] < 3 else "finish",
        {"recheck": "perplexity", "finish": END}
    )
    .compile()
)


# (Section 5: Streamlit Dashboard) ----------
st.title("Fact-Checking AI Agent Dashboard")

claim = st.text_input("Enter your claim:")
if st.button("Submit") and claim:
    state: FactCheckState = {
        "query": claim,
        "plan": "",
        "perplexity_result": "",
        "openai_result": "",
        "adjudicator_result": "",
        "final_verdict": "",
        "attempts": 0,
        "evidence_summary": "",
        "messages": []
    }
    
    with st.spinner("Processing your claim..."):
        state = graph.invoke(state)
    
    if state["final_verdict"] == "Claim is True":
        st.success(state["final_verdict"])
    elif state["final_verdict"] == "Claim is False":
        st.error(state["final_verdict"])
    else:
        st.info(state["final_verdict"])
    
    st.markdown("**Evidence Summary:**")
    st.write(state["evidence_summary"])
    
    st.markdown("**Detailed Log:**")
    st.markdown("# Adjudicator Result")
    st.markdown(state["adjudicator_result"])
    st.markdown("# OpenAI Result")
    st.markdown(state["openai_result"])
    st.markdown("# Perplexity Result")
    st.markdown(state["perplexity_result"])