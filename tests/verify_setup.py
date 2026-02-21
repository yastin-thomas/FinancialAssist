import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()

try:
    print("Verifying imports...")
    import langchain
    import langgraph
    import streamlit
    import yfinance
    import duckduckgo_search
    import dotenv
    print("Core dependencies imported successfully.")
    
    # Verify project modules
    from src.core.tools import get_stock_price, search_investopedia
    print("Project tools imported successfully.")
    
    from src.agents.agents import finance_qa_runnable
    print("Project agents imported successfully.")
    
    from src.workflow.workflow import graph
    print("Project workflow imported successfully.")
    
    print("\nALL CHECKS PASSED. The environment is ready.")
except ImportError as e:
    print(f"\nIMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: {e}")
    sys.exit(1)
