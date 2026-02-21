from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

from src.agents.agents import (
    finance_qa_runnable,
    portfolio_analysis_runnable,
    market_analysis_runnable,
    goal_planning_runnable,
    news_synthesizer_runnable,
    tax_education_runnable,
    router_runnable,
)
from src.core.tools import get_stock_price, get_company_info, query_knowledge_base, analyze_portfolio, get_market_data, calculate_investment_projection

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    force_agent: str # Optional: Force routing to specific agent

# Helper to create agent node
def create_agent_node(agent_runnable, name):
    def agent_node(state):
        print(f"--- Agent Invoked: {name} ---")
        result = agent_runnable.invoke(state)
        # Handle AIMessage which might have tool_calls
        return {"messages": [result], "sender": name}
    return agent_node

# Define agent nodes
finance_qa_node = create_agent_node(finance_qa_runnable, "finance_qa")
portfolio_analysis_node = create_agent_node(portfolio_analysis_runnable, "portfolio_analysis")
market_analysis_node = create_agent_node(market_analysis_runnable, "market_analysis")
goal_planning_node = create_agent_node(goal_planning_runnable, "goal_planning")
news_synthesizer_node = create_agent_node(news_synthesizer_runnable, "news_synthesizer")
tax_education_node = create_agent_node(tax_education_runnable, "tax_education")

# Tool Node
tools = [get_stock_price, get_company_info, query_knowledge_base, analyze_portfolio, get_market_data, calculate_investment_projection]
tool_node = ToolNode(tools)

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("finance_qa", finance_qa_node)
workflow.add_node("portfolio_analysis", portfolio_analysis_node)
workflow.add_node("market_analysis", market_analysis_node)
workflow.add_node("goal_planning", goal_planning_node)
workflow.add_node("news_synthesizer", news_synthesizer_node)
workflow.add_node("tax_education", tax_education_node)
workflow.add_node("tools", tool_node)

# Routing Logic (Conditional Edge)
def router_logic(state):
    # Check if forced routing is enabled (e.g., from specific agent tab)
    if state.get("force_agent"):
        print(f"--- Force Routing: {state['force_agent']} ---")
        return state["force_agent"]

    messages = state["messages"]
    # Invoke the router runnable
    response = router_runnable.invoke({"messages": messages})
    next_agent = response.content.strip().lower()
    print(f"--- Router Decision: {next_agent} ---")
    
    AGENT_MAPPING = {
        "finance_qa": "finance_qa",
        "portfolio_analysis": "portfolio_analysis",
        "market_analysis": "market_analysis",
        "goal_planning": "goal_planning",
        "news_synthesizer": "news_synthesizer",
        "tax_education": "tax_education",
    }
    
    # Fallback to fuzzy substring match if exact match fails
    if next_agent not in AGENT_MAPPING:
        if "finance" in next_agent: next_agent = "finance_qa"
        elif "portfolio" in next_agent: next_agent = "portfolio_analysis"
        elif "market" in next_agent: next_agent = "market_analysis"
        elif "goal" in next_agent: next_agent = "goal_planning"
        elif "news" in next_agent: next_agent = "news_synthesizer"
        elif "tax" in next_agent: next_agent = "tax_education"
        
    return AGENT_MAPPING.get(next_agent, "finance_qa")

# Set entry point
workflow.set_conditional_entry_point(
    router_logic,
    {
        "finance_qa": "finance_qa",
        "portfolio_analysis": "portfolio_analysis",
        "market_analysis": "market_analysis",
        "goal_planning": "goal_planning",
        "news_synthesizer": "news_synthesizer",
        "tax_education": "tax_education",
    }
)

# Agent -> Tools or End Logic
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        print("--- Agent requesting tool capability ---")
        return "tools"
    print("--- Agent finishing ---")
    return END

for agent in ["finance_qa", "portfolio_analysis", "market_analysis", "goal_planning", "news_synthesizer", "tax_education"]:
    workflow.add_conditional_edges(agent, should_continue, ["tools", END])

# Tools -> Back to Agent Logic
def route_back_to_agent(state):
    sender = state["sender"]
    print(f"--- Tool output returning to: {sender} ---")
    return sender

workflow.add_conditional_edges(
    "tools",
    route_back_to_agent,
    {
        "finance_qa": "finance_qa",
        "portfolio_analysis": "portfolio_analysis",
        "market_analysis": "market_analysis",
        "goal_planning": "goal_planning",
        "news_synthesizer": "news_synthesizer",
        "tax_education": "tax_education",
    }
)

# Compile graph
graph = workflow.compile()
