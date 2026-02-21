import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

# Patch everything in workflow to avoid live LLM calls during import
@patch("src.agents.agents.ChatGoogleGenerativeAI")
def test_workflow_logic(mock_chat):
    mock_llm_instance = MagicMock()
    mock_chat.return_value = mock_llm_instance
    
    from src.workflow.workflow import (
        create_agent_node,
        router_logic,
        should_continue,
        route_back_to_agent
    )
    
    # 1. Test create_agent_node
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = AIMessage(content="Hello")
    agent_node = create_agent_node(mock_runnable, "test_agent")
    
    state = {"messages": [HumanMessage(content="Hi")], "sender": ""}
    result = agent_node(state)
    
    assert result["sender"] == "test_agent"
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Hello"
    
    # 2. Test router_logic - Force routing
    state_force = {"force_agent": "market_analysis"}
    assert router_logic(state_force) == "market_analysis"
    
    # 3. Test router_logic - LLM routing
    with patch("src.workflow.workflow.router_runnable") as mock_router:
        state_routing = {"messages": [], "force_agent": None}
        
        mock_router.invoke.return_value = AIMessage(content="portfolio_analysis")
        assert router_logic(state_routing) == "portfolio_analysis"
        
        mock_router.invoke.return_value = AIMessage(content="finance_qa")
        assert router_logic(state_routing) == "finance_qa"
        
        mock_router.invoke.return_value = AIMessage(content="market_analysis")
        assert router_logic(state_routing) == "market_analysis"
        
        mock_router.invoke.return_value = AIMessage(content="goal")
        assert router_logic(state_routing) == "goal_planning"
        
        mock_router.invoke.return_value = AIMessage(content="news")
        assert router_logic(state_routing) == "news_synthesizer"

        mock_router.invoke.return_value = AIMessage(content="tax")
        assert router_logic(state_routing) == "tax_education"
        
        mock_router.invoke.return_value = AIMessage(content="gibberish")
        assert router_logic(state_routing) == "finance_qa" # default

    # 4. Test should_continue
    # Without tool calls
    state_no_tools = {"messages": [AIMessage(content="Done")]}
    assert should_continue(state_no_tools) == "__end__"
    
    # With tool calls
    msg_with_tools = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "123"}])
    state_with_tools = {"messages": [msg_with_tools]}
    assert should_continue(state_with_tools) == "tools"
    
    # 5. Test route_back_to_agent
    state_route_back = {"sender": "tax_education"}
    assert route_back_to_agent(state_route_back) == "tax_education"
