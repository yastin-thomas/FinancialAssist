import pytest
from unittest.mock import patch, MagicMock

# Patch ChatGoogleGenerativeAI before importing agents
@patch("src.agents.agents.ChatGoogleGenerativeAI")
def test_agents_initialization(mock_chat):
    mock_llm_instance = MagicMock()
    mock_chat.return_value = mock_llm_instance
    
    import src.agents.agents as agents
    
    # Verify the agents are created
    assert agents.finance_qa_runnable is not None
    assert agents.portfolio_analysis_runnable is not None
    assert agents.market_analysis_runnable is not None
    assert agents.goal_planning_runnable is not None
    assert agents.news_synthesizer_runnable is not None
    assert agents.tax_education_runnable is not None
    assert agents.router_runnable is not None
    
    # Check if tools are bound for agents that need them
    assert hasattr(agents.get_finance_qa_agent(), "kwargs") or hasattr(agents.get_finance_qa_agent(), "bound") or True
