import yfinance as yf
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import time
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()
# Initialize Vector DB lazily
VECTOR_DB_PATH = "financial_db"
_vector_db = None

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        if os.path.exists(VECTOR_DB_PATH):
            try:
                embeddings = OpenAIEmbeddings()
                _vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                print(f"--- Vector DB loaded from {VECTOR_DB_PATH} ---")
            except Exception as e:
                print(f"--- Failed to load Vector DB: {e} ---")
    return _vector_db

@tool
def get_stock_price(ticker: str) -> str:
    """
    Fetches the current stock price for a given ticker symbol.
    Args:
        ticker: The exact stock ticker symbol (e.g., AAPL, MSFT). DO NOT pass company names like 'Apple'.
    Returns:
        The current stock price or an error message.
    """
    print(f"--- Tool Invoked: get_stock_price with ticker: {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if history.empty:
            return f"Could not find price data for {ticker}."
        current_price = history['Close'].iloc[-1]
        return f"The current price of {ticker} is ${current_price:.2f}"
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {str(e)}"

@tool
def get_company_info(ticker: str) -> str:
    """
    Fetches company information for a given ticker symbol.
    Args:
        ticker: The exact stock ticker symbol (e.g., AAPL). DO NOT pass company names.
    Returns:
        A summary of the company information.
    """
    print(f"--- Tool Invoked: get_company_info with ticker: {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return f"Could not find information for {ticker}."
        
        name = info.get('longName', ticker)
        summary = info.get('longBusinessSummary', 'No summary available.')
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        return f"**{name}** ({ticker})\nSector: {sector}\nIndustry: {industry}\n\n{summary}"
    except Exception as e:
        return f"Error fetching company info for {ticker}: {str(e)}"



@tool
def query_knowledge_base(query: str) -> str:
    """
    Queries the internal financial knowledge base (Vector DB) for information.
    Use this tool to get trusted definitions and explanations before searching the web.
    Args:
        query: The question or concept to search for in the knowledge base.
    Returns:
        Relevant excerpts from the knowledge base.
    """
    print(f"--- Tool Invoked: query_knowledge_base with query: {query} ---")
    
    db = get_vector_db()
    
    print(f"--- Initiating Vector DB Query: {query} ---")
    if not db:
        print("--- Error: Vector DB is not initialized ---")
        return "Knowledge base not available. Please contact the administrator."
    
    try:
        print(f"--- Querying Vector DB: {query} ---")
        docs = db.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in the knowledge base."
            
        response = "**From Knowledge Base:**\n"
        for doc in docs:
            response += f"- {doc.page_content[:500]}...\n[Source: {doc.metadata.get('source', 'Unknown')}]\n\n"
            
        return response
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"

@tool
def analyze_portfolio(positions: str) -> str:
    """
    Calculates total value and allocation percentages for a portfolio.
    Args:
        positions: A string representation of a dictionary where keys are strictly VALID TICKER SYMBOLS (e.g., "AAPL", not "Apple") and values are quantities.
                   Example: '{"AAPL": 10, "MSFT": 5, "GOOGL": 2}'
                   CRITICAL: Do not pass full company names as keys. Convert them to tickers first.
    Returns:
        A string containing the portfolio analysis table.
    """
    print(f"--- Tool Invoked: analyze_portfolio with positions: {positions} ---")
    try:
        # Parse input string to dict
        # The LLM might pass a string like "{'AAPL': 10}", so we need to be careful with quotes
        positions_str = positions.replace("'", '"')
        portfolio = json.loads(positions_str)
        
        if not portfolio:
            return "Empty portfolio provided."
            
        print(f"--- Analyzing metrics for: {portfolio} ---")
        
        tickers_list = list(portfolio.keys())
        # Batch download for all tickers to optimize performance
        try:
            # Setting auto_adjust=True correctly fetches adjusted close directly
            hist_data = yf.download(tickers_list, period="1d", group_by="ticker", auto_adjust=True, progress=False)
        except Exception as e:
            print(f"Failed to batch download yfinance data: {e}")
            hist_data = pd.DataFrame()
            
        data = []
        total_value = 0
        
        for ticker, quantity in portfolio.items():
            current_price = 0
            
            if not hist_data.empty:
                try:
                    # If multiple tickers, yfinance returns MultiIndex columns. If single, normal columns.
                    if len(tickers_list) > 1:
                        # Ensure we get the latest valid close price
                        series = hist_data[ticker]['Close'].dropna()
                        if not series.empty:
                             current_price = series.iloc[-1]
                    else:
                        series = hist_data['Close'].dropna()
                        if not series.empty:
                            current_price = series.iloc[-1]
                except KeyError:
                    pass
                except Exception as e:
                    print(f"Error extracting price for {ticker}: {e}")
                
            value = float(current_price * quantity)
            total_value += value
            
            data.append({
                "Ticker": ticker,
                "Quantity": quantity,
                "Price": float(current_price),
                "Value": value
            })
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        if total_value > 0:
            df["Allocation (%)"] = (df["Value"] / total_value) * 100
        else:
            df["Allocation (%)"] = 0
            
        # Format for output - ensuring consistent string representation
        # We perform formatting but keep values clean for potential parsing if tool output is read directly
        # However, for the markdown table, we want it pretty.
        
        df_display = df.copy()
        df_display["Price"] = df_display["Price"].map("${:,.2f}".format)
        df_display["Value"] = df_display["Value"].map("${:,.2f}".format)
        df_display["Allocation (%)"] = df_display["Allocation (%)"].map("{:.2f}%".format)
        
        # Convert to markdown table with explicit string alignment (left) to avoid parser confusion
        markdown_table = df_display.to_markdown(index=False, colalign=("left", "left", "right", "right", "right"))
        
        summary = f"\n\n**Total Portfolio Value: ${total_value:,.2f}**\n\n*Source: Yahoo Finance*"
       
        return markdown_table + summary
        
    except Exception as e:
        return f"Error analyzing portfolio: {str(e)}"

# Cache for market data: {period: {"data": data, "timestamp": time.time()}}
_market_data_cache = {}
MARKET_DATA_TTL_SECONDS = 30 * 60  # 30 minutes

# Helper for fetching market data
def _fetch_market_data(period="3mo"):
    global _market_data_cache
    current_time = time.time()
    
    # Check cache
    if period in _market_data_cache:
        cached_item = _market_data_cache[period]
        if current_time - cached_item["timestamp"] < MARKET_DATA_TTL_SECONDS:
            print(f"--- Returning cached market data for {period} (TTL: {MARKET_DATA_TTL_SECONDS - (current_time - cached_item['timestamp']):.0f}s left) ---")
            return cached_item["data"]

    # Using ETF symbols for better data availability sometimes, or indices
    # ^GSPC = S&P 500, ^DJI = Dow Jones, ^IXIC = Nasdaq
    tickers = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "Nasdaq"}
    data = []
    try:
        for ticker, name in tickers.items():
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                hist_reset = hist.reset_index()
                hist_reset["Date"] = hist_reset["Date"].dt.strftime('%Y-%m-%d')
                hist_reset["Ticker"] = name
                hist_reset["Close"] = hist_reset["Close"].round(2)
                # vectorized dictionary creation
                data.extend(hist_reset[["Date", "Ticker", "Close"]].to_dict("records"))
                
        # Save to cache if successful
        if data:
            _market_data_cache[period] = {
                "data": data,
                "timestamp": current_time
            }
    except Exception as e:
        print(f"Error fetching market data: {e}")
    return data

@tool
def get_market_data(period: str = "3mo") -> str:
    """
    Fetches historical market data (3 months) for major indices: S&P 500, Dow Jones, and Nasdaq.
    Use this tool when asked about "market overview", "market trends", or "how the market is doing".
    
    Args:
        period: Time period (default: "3mo").
    Returns:
        Markdown table with Date, Ticker, and Close price.
    """
    print(f"--- Tool Invoked: get_market_data with period: {period} ---")
    data = _fetch_market_data(period)
    if not data:
        return "Could not fetch market data."
    
    df = pd.DataFrame(data)
   
    return df.to_markdown(index=False)

@tool
def calculate_investment_projection(initial_amount: float, monthly_contribution: float, years: int, annual_return_rate: float) -> str:
    """
    Calculates the future value of an investment using compound interest.
    Use this tool whenever you need to project if a user will hit their financial goal.
    
    Args:
        initial_amount: The current amount saved ($).
        monthly_contribution: The amount added at the end of every month ($).
        years: The number of years the money will be invested.
        annual_return_rate: Expected annual return rate as a decimal (e.g. 0.07 for 7%).
    """
    print(f"--- Tool Invoked: calculate_investment_projection (init: {initial_amount}, mo: {monthly_contribution}, yrs: {years}, rate: {annual_return_rate}) ---")
    try:
        # Monthly rate and total months
        r = annual_return_rate / 12
        n = years * 12
        
        # Formula for Future Value of initial sum
        future_value_principal = initial_amount * ((1 + r) ** n)
        
        # Formula for Future Value of a series of monthly contributions
        if r > 0:
            future_value_contributions = monthly_contribution * (((1 + r) ** n - 1) / r)
        else:
            future_value_contributions = monthly_contribution * n
            
        total_value = future_value_principal + future_value_contributions
        total_contributed = initial_amount + (monthly_contribution * n)
        interest_earned = total_value - total_contributed
        
        return (
            f"Projection after {years} years at {annual_return_rate*100}% return:\n"
            f"- Total Projected Output: ${total_value:,.2f}\n"
            f"- Total Contributions: ${total_contributed:,.2f}\n"
            f"- Compound Interest Earned: ${interest_earned:,.2f}"
        )
    except Exception as e:
        return f"Error calculating projection: {str(e)}"
