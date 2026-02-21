import pytest
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

class SessionState(dict):
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

mock_st = MagicMock()
mock_st.session_state = SessionState()
sys.modules['streamlit'] = mock_st

# Import main after mocking
import src.web_app.main as main_app

def test_render_portfolio_content():
    # Test with valid portfolio table
    content_with_table = """Some text before
| Ticker | Quantity | Price | Value | Allocation (%) |
|:---|:---|---:|---:|---:|
| AAPL | 10 | $150.00 | $1,500.00 | 50.00% |
| MSFT | 5 | $300.00 | $1,500.00 | 50.00% |
Some text after"""
    
    with patch("plotly.express.pie") as mock_pie, patch("streamlit.plotly_chart") as mock_chart, patch("streamlit.markdown") as mock_markdown:
        main_app.render_portfolio_content(content_with_table, "key1")
        mock_pie.assert_called_once()
        mock_chart.assert_called_once()
        
    # Test with fallback bullet points
    content_with_bullets = """Some text before
**AAPL**
Value: $1,500.00

**MSFT**
Value: $1,500.00
Some text after"""
    with patch("plotly.express.pie") as mock_pie, patch("streamlit.plotly_chart") as mock_chart:
        main_app.render_portfolio_content(content_with_bullets, "key2")
        mock_pie.assert_called_once()
    
    # Test with no data
    with patch("plotly.express.pie") as mock_pie:
        main_app.render_portfolio_content("Just some normal text", "key3")
        mock_pie.assert_not_called()

def test_render_market_content():
    # Test with valid market table
    content_with_market = """Overview:
| Date | Ticker | Close |
|---|---|---|
| 2023-01-01 | S&P 500 | 4000.00 |
| 2023-01-02 | S&P 500 | 4050.00 |
End of overview."""
    
    with patch("plotly.express.line") as mock_line, patch("streamlit.plotly_chart") as mock_chart:
        main_app.render_market_content(content_with_market, "key4")
        mock_line.assert_called_once()
        mock_chart.assert_called_once()

    # Test with no data
    with patch("plotly.express.line") as mock_line:
        main_app.render_market_content("Just text", "key5")
        mock_line.assert_not_called()

@patch("src.web_app.main.graph")
def test_handle_chat_view(mock_graph):
    # Setup session state properly
    mock_st.session_state = SessionState()
    mock_st.session_state["messages_Test_View"] = [HumanMessage(content="Hello")]
    mock_st.session_state["selected_agent"] = "Test View"
    
    # Mock chat_input to return a new prompt
    mock_st.chat_input.return_value = "What is AAPL?"
    
    # Mock graph invoke to return a response
    mock_graph.invoke.return_value = {
        "messages": [HumanMessage(content="What is AAPL?"), AIMessage(content="AAPL is Apple")],
        "sender": "finance_qa"
    }
    
    main_app.handle_chat_view("Test View")
    
    # Verify graph was invoked
    mock_graph.invoke.assert_called()
    
    # Test routing to a different view
    mock_graph.invoke.return_value = {
        "messages": [HumanMessage(content="Analyze portfolio"), AIMessage(content="Portfolio looks good")],
        "sender": "portfolio_analysis"
    }
    mock_st.session_state["messages_Test_View"] = []
    
    main_app.handle_chat_view("Test View")
    assert mock_st.session_state.get("selected_agent") == "Portfolio Analysis"
    
    # Test exception handling (using side_effect)
    mock_st.chat_input.return_value = "trigger error"
    mock_graph.invoke.side_effect = Exception("Test error")
    main_app.handle_chat_view("Test View")
    mock_st.error.assert_called()
