import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.core.tools import (
    get_stock_price,
    get_company_info,
    query_knowledge_base,
    analyze_portfolio,
    get_market_data,
    _fetch_market_data
)

@patch("src.core.tools.yf.Ticker")
def test_get_stock_price_success(mock_ticker):
    # Setup mock
    mock_stock = MagicMock()
    mock_df = pd.DataFrame({"Close": [150.0]})
    mock_stock.history.return_value = mock_df
    mock_ticker.return_value = mock_stock

    # Call function
    result = get_stock_price.invoke({"ticker": "AAPL"})
    
    # Assert
    assert "current price of AAPL is $150.00" in result
    mock_ticker.assert_called_with("AAPL")

@patch("src.core.tools.yf.Ticker")
def test_get_stock_price_empty(mock_ticker):
    mock_stock = MagicMock()
    mock_stock.history.return_value = pd.DataFrame() # empty
    mock_ticker.return_value = mock_stock

    result = get_stock_price.invoke({"ticker": "INVALID"})
    assert "Could not find price data" in result

@patch("src.core.tools.yf.Ticker")
def test_get_stock_price_error(mock_ticker):
    mock_ticker.side_effect = Exception("API error")
    result = get_stock_price.invoke({"ticker": "AAPL"})
    assert "Error fetching stock price" in result

@patch("src.core.tools.yf.Ticker")
def test_get_company_info_success(mock_ticker):
    mock_stock = MagicMock()
    mock_stock.info = {
        "longName": "Apple Inc",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "longBusinessSummary": "Designs and manufactures devices..."
    }
    mock_ticker.return_value = mock_stock

    result = get_company_info.invoke({"ticker": "AAPL"})
    assert "Apple Inc" in result
    assert "Technology" in result
    assert "Consumer Electronics" in result
    assert "Designs and manufactures devices..." in result

@patch("src.core.tools.yf.Ticker")
def test_get_company_info_not_found(mock_ticker):
    mock_stock = MagicMock()
    mock_stock.info = {}
    mock_ticker.return_value = mock_stock

    result = get_company_info.invoke({"ticker": "INVALID"})
    assert "Could not find information" in result

@patch("src.core.tools.yf.Ticker")
def test_get_company_info_error(mock_ticker):
    mock_ticker.side_effect = Exception("API Error")
    result = get_company_info.invoke({"ticker": "AAPL"})
    assert "Error fetching company info" in result

@patch("src.core.tools.os.path.exists")
def test_query_knowledge_base_not_init(mock_exists):
    import src.core.tools
    mock_exists.return_value = False
    src.core.tools._vector_db = None
    result = query_knowledge_base.invoke({"query": "What is AI?"})
    assert "Knowledge base not available" in result

@patch("src.core.tools.get_vector_db")
def test_query_knowledge_base_success(mock_get_db):
    import src.core.tools
    mock_doc = MagicMock()
    mock_doc.page_content = "AI is Artificial Intelligence"
    mock_doc.metadata = {"source": "test.txt"}
    
    mock_db_instance = MagicMock()
    mock_db_instance.similarity_search.return_value = [mock_doc]
    mock_get_db.return_value = mock_db_instance

    result = query_knowledge_base.invoke({"query": "What is AI?"})
    assert "From Knowledge Base" in result
    assert "AI is Artificial Intelligence" in result

@patch("src.core.tools.get_vector_db")
def test_query_knowledge_base_no_docs(mock_get_db):
    import src.core.tools
    mock_db_instance = MagicMock()
    mock_db_instance.similarity_search.return_value = []
    mock_get_db.return_value = mock_db_instance

    result = query_knowledge_base.invoke({"query": "What is AI?"})
    assert "No relevant information found" in result

@patch("src.core.tools.get_vector_db")
def test_query_knowledge_base_error(mock_get_db):
    import src.core.tools
    mock_db_instance = MagicMock()
    mock_db_instance.similarity_search.side_effect = Exception("DB Error")
    mock_get_db.return_value = mock_db_instance

    result = query_knowledge_base.invoke({"query": "What is AI?"})
    assert "Error querying knowledge base" in result

@patch("src.core.tools.yf.Ticker")
def test_analyze_portfolio_empty(mock_ticker):
    result = analyze_portfolio.invoke({"positions": "{}"})
    assert "Empty portfolio provided" in result

@patch("src.core.tools.yf.Ticker")
def test_analyze_portfolio_success(mock_ticker):
    mock_stock = MagicMock()
    mock_df = pd.DataFrame({"Close": [150.0]})
    mock_stock.history.return_value = mock_df
    mock_ticker.return_value = mock_stock

    result = analyze_portfolio.invoke({"positions": '{"AAPL": 10, "MSFT": 5}'})
    assert "Total Portfolio Value" in result
    assert "Allocation (%)" in result

@patch("src.core.tools.yf.Ticker")
def test_analyze_portfolio_empty_history(mock_ticker):
    mock_stock = MagicMock()
    mock_df = pd.DataFrame()
    mock_stock.history.return_value = mock_df
    mock_ticker.return_value = mock_stock

    result = analyze_portfolio.invoke({"positions": '{"INVALID": 10}'})
    assert "Total Portfolio Value: $0.00" in result

@patch("src.core.tools.yf.Ticker")
def test_analyze_portfolio_error(mock_ticker):
    result = analyze_portfolio.invoke({"positions": "invalid json"})
    assert "Error analyzing portfolio" in result

@patch("src.core.tools.yf.Ticker")
def test_fetch_market_data_success(mock_ticker):
    import src.core.tools
    src.core.tools._market_data_cache.clear()
    
    import datetime
    mock_stock = MagicMock()
    # Mocking date index
    dates = [datetime.datetime(2023, 1, 1)]
    mock_df = pd.DataFrame({"Close": [4000.0]}, index=dates)
    mock_df.index.name = "Date"
    mock_stock.history.return_value = mock_df
    mock_ticker.return_value = mock_stock

    result = _fetch_market_data()
    assert len(result) == 3
    assert result[0]["Close"] == 4000.0

@patch("src.core.tools.yf.Ticker")
def test_fetch_market_data_empty(mock_ticker):
    import src.core.tools
    src.core.tools._market_data_cache.clear()
    
    mock_stock = MagicMock()
    mock_stock.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_stock

    result = _fetch_market_data()
    assert len(result) == 0

@patch("src.core.tools.yf.Ticker")
def test_fetch_market_data_error(mock_ticker):
    import src.core.tools
    src.core.tools._market_data_cache.clear()
    
    mock_ticker.side_effect = Exception("API error")
    result = _fetch_market_data()
    assert len(result) == 0

@patch("src.core.tools._fetch_market_data")
def test_get_market_data_success(mock_fetch):
    mock_fetch.return_value = [{"Date": "2023-01-01", "Ticker": "S&P 500", "Close": 4000.0}]
    result = get_market_data.invoke({"period": "3mo"})
    assert "S&P 500" in result

@patch("src.core.tools._fetch_market_data")
def test_get_market_data_empty(mock_fetch):
    mock_fetch.return_value = []
    result = get_market_data.invoke({"period": "3mo"})
    assert "Could not fetch market data" in result
