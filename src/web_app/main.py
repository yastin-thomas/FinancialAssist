import streamlit as st
import os
import pandas as pd
import re
from io import StringIO
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.workflow.workflow import graph
from src.core.tools import get_market_data, _fetch_market_data
from src.agents.agents import DAILY_LLM_RATE_LIMIT
from dotenv import load_dotenv

# NOTE: We removed the @st.cache_data monkeypatch here because Streamlit's 
# threading models and caching decorators conflict with LangChain's async 
# tool execution loops, causing "Event loop is closed" errors.

load_dotenv()

st.set_page_config(page_title="AI Finance Assistant", layout="wide")

st.title("AI Finance Assistant")

# Modern Premium CSS Injection
st.markdown("""
<style>
    /* Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Title */
    h1 {
        background: -webkit-linear-gradient(45deg, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    /* Premium Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        box-shadow: inset -1px 0px 0px rgba(0,0,0,0.05);
    }
    
    /* Chat Message Aesthetic Enhancements */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03);
        border: 1px solid rgba(128,128,128,0.1);
        background-color: transparent;
    }
    
    [data-testid="stChatInput"] {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for active agent (navigation)
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "General Assistant"

# Define Agent Options
agent_options = [
    "General Assistant", 
    "Finance Q&A", 
    "Portfolio Analysis", 
    "Market Analysis", 
    "Goal Planning", 
    "News Synthesizer", 
    "Tax Education"
]

current_selection = st.session_state.get("selected_agent", "General Assistant")
try:
    index = agent_options.index(current_selection)
except ValueError:
    index = 0

# Sidebar for configuration and navigation
with st.sidebar:
    st.header("Navigation")
    
    # Use native st.pills for a row-wise menu without CSS hacks
    selection = st.pills(
        "Choose an Agent",
        options=agent_options,
        default=current_selection,
        label_visibility="collapsed"
    )

    # Enforce selection if user un-clicks the active pill
    if not selection:
        selection = current_selection

    st.markdown("---")
    
    st.header("Configuration")
    if "GOOGLE_API_KEY" not in os.environ:
        api_key = st.text_input("Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.success("Google API Key provided.")

if selection != current_selection:
    st.session_state.selected_agent = selection
    st.rerun()

agent_map = {
    "General Assistant": None,
    "Finance Q&A": "finance_qa",
    "Portfolio Analysis": "portfolio_analysis",
    "Market Analysis": "market_analysis",
    "Goal Planning": "goal_planning",
    "News Synthesizer": "news_synthesizer",
    "Tax Education": "tax_education"
}
reverse_agent_map = {v: k for k, v in agent_map.items() if v is not None}

def render_portfolio_content(content, unique_key):
    """
    Parses agent content to extract portfolio data.
    If data is found:
        1. Strips the raw markdown table from the text to avoid duplication.
        2. Renders the Pie Chart (First).
        3. Renders the cleaned text (Second).
    If no data is found, simply renders the original content.
    """
    df = None
    cleaned_content = content
    
    # 1. Try passing Markdown Table
    # Regex to capture a markdown table block
    # Looks for lines starting with | and ending with | eventually
    
    # Debug: Print content len
    print(f"DEBUG: render_portfolio_content called. Content len: {len(content)}")
    
    # Robust check for table presence using regex
    # Matches lines like: | Ticker | Quantity | ...
    table_pattern = re.compile(r'\|.*Ticker.*\|.*Value.*\|', re.IGNORECASE)
    
    if table_pattern.search(content):
        # print("DEBUG: Table pattern found.")
        try:
            lines = content.split('\n')
            # Identify table lines more loosely (containing | and at least 3 chars)
            table_lines = [line for line in lines if "|" in line and len(line.strip()) > 3]
            
            if len(table_lines) > 2:
                table_str = "\n".join(table_lines)
                # print(f"DEBUG: Attempting to parse table string:\n{table_str[:100]}...")
                
                df = pd.read_csv(StringIO(table_str), sep="|", skipinitialspace=True)
                df = df.dropna(axis=1, how='all')
                df.columns = [c.strip() for c in df.columns]
                
                # Normalize column names
                for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip()
                
                if 'Value' in df.columns and 'Ticker' in df.columns:
                    # Clean data
                    df = df[~df['Value'].str.contains('---', na=False)] # Remove separator lines
                    df['ValueClean'] = df['Value'].astype(str).str.replace('$', '').str.replace(',', '')
                    df['ValueClean'] = pd.to_numeric(df['ValueClean'], errors='coerce')
                    df = df.dropna(subset=['ValueClean'])
                    
                    if not df.empty:
                        # print("DEBUG: DataFrame parsed successfully.")
                        # Successfully parsed table.
                        # Now strip the table lines from content.
                        # We remove any line that looks like a table row
                        cleaned_lines = [line for line in lines if not ("|" in line and len(line.strip()) > 3)]
                        # Remove excessive newlines
                        cleaned_content = "\n".join(cleaned_lines)
                        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        except Exception as e:
            # print(f"DEBUG: Error parsing table: {e}")
            pass

    # 2. Fallback: Regex for Bullet Points (only if table failed)
    if df is None or df.empty:
        try:
            tickers = []
            values = []
            segments = content.split('\n\n')
            for seg in segments:
                ticker_match = re.search(r'\*+(\w+)', seg)
                value_match = re.search(r'Value:\s*\$([\d,]+\.?\d*)', seg)
                if ticker_match and value_match:
                    tickers.append(ticker_match.group(1))
                    clean_val = value_match.group(1).replace(',', '')
                    values.append(float(clean_val))
            if tickers and values:
                df = pd.DataFrame({'Ticker': tickers, 'ValueClean': values})
        except Exception:
            pass
            
    # Render Logic
    if df is not None and not df.empty:
        import plotly.express as px
        st.markdown("### Portfolio")
        fig = px.pie(df, values='ValueClean', names='Ticker', title='Portfolio Allocation by Value')
        st.plotly_chart(fig, key=unique_key)
    
    st.markdown(cleaned_content)

def render_market_content(content, unique_key):
    """
    Parses market data table and renders a Line Chart.
    """
    df = None
    cleaned_content = content
    
    # Validating if it's a market data table (Date, Ticker, Close) using Regex
    # Matches | Date ... | Ticker ... | Close ... |
    market_table_pattern = re.compile(r'\|\s*Date\s*\|\s*Ticker\s*\|\s*Close\s*\|', re.IGNORECASE)

    if market_table_pattern.search(content):
        print("DEBUG: Market Data Table detected.")
        try:
            lines = content.split('\n')
            table_lines = [line for line in lines if "|" in line and len(line.strip()) > 3]
            if len(table_lines) > 2:
                table_str = "\n".join(table_lines)
                df = pd.read_csv(StringIO(table_str), sep="|", skipinitialspace=True)
                df = df.dropna(axis=1, how='all')
                df.columns = [c.strip() for c in df.columns]
                
                # Cleanup
                for col in df.columns:
                     if df[col].dtype == 'object':
                         df[col] = df[col].astype(str).str.strip()
                
                if 'Close' in df.columns:
                    # Remove separator lines
                    df = df[~df['Close'].astype(str).str.contains('---', na=False)]
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    df = df.dropna(subset=['Close'])
                    
                    if not df.empty:
                        # Strip table from content
                        # Remove any line that looks like part of the table
                        cleaned_lines = [line for line in lines if not ("|" in line and len(line.strip()) > 3)]
                        cleaned_content = "\n".join(cleaned_lines)
                        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        except Exception as e:
            print(f"DEBUG: Error parsing market table: {e}")
            pass

    # Render Chart First
    if df is not None and not df.empty:
        import plotly.express as px
        st.markdown("### Market Trends (30 Days)")
        # Ensure Date is datetime for better axis
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        fig = px.line(df, x='Date', y='Close', color='Ticker', title='Market Overview', markers=True)
        st.plotly_chart(fig, key=unique_key)

    st.markdown(cleaned_content)

def handle_chat_view(current_view_name):
    st.header(current_view_name)
    
    key = f"messages_{current_view_name.replace(' ', '_')}"
    if key not in st.session_state:
        st.session_state[key] = []
        
    for i, message in enumerate(st.session_state[key]):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            if message.content:
                with st.chat_message("assistant"):
                    if current_view_name == "Portfolio Analysis":
                        # Use special renderer that strips table
                        render_portfolio_content(message.content, f"{key}_{i}")
                    elif current_view_name == "Market Analysis":
                        render_market_content(message.content, f"{key}_{i}")
                    else:
                        st.markdown(message.content)
    
    if prompt := st.chat_input(f"Ask in {current_view_name}...", key=f"input_{key}"):  
        if "GOOGLE_API_KEY" not in os.environ:
            st.warning("⚠️ Please enter your Google API Key in the sidebar before sending a message.")
            return
        
        user_msg = HumanMessage(content=prompt)
        st.session_state[key].append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.spinner(f"{current_view_name} is processing..."):
            try:
                # Hard block on UI level before LangGraph runs
                if st.session_state.get("llm_daily_count", 0) >= DAILY_LLM_RATE_LIMIT:
                    raise Exception(f"Daily LLM rate limit of {DAILY_LLM_RATE_LIMIT} requests exceeded for this session. Please try again tomorrow.")

                inputs = {"messages": st.session_state[key]}
                final_state = graph.invoke(inputs)
                sender = final_state.get("sender") 
                
                target_view_name = current_view_name 
                if sender and sender in reverse_agent_map:
                    mapped_view = reverse_agent_map[sender]
                    if mapped_view != current_view_name:
                         target_view_name = mapped_view

                if target_view_name != current_view_name:
                    target_key = f"messages_{target_view_name.replace(' ', '_')}"
                    if target_key not in st.session_state:
                        st.session_state[target_key] = []
                    st.session_state[target_key].append(user_msg)
                    response_msg = final_state["messages"][-1]
                    st.session_state[target_key].append(response_msg)
                    st.session_state.selected_agent = target_view_name
                    if key in st.session_state and st.session_state[key]:
                        st.session_state[key].pop()
                    st.rerun()
                    return

                st.session_state[key] = final_state["messages"]
                
                last_msg = st.session_state[key][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    with st.chat_message("assistant"):
                        if current_view_name == "Portfolio Analysis":
                            render_portfolio_content(last_msg.content, f"{key}_{len(st.session_state[key])-1}")
                        elif current_view_name == "Market Analysis":
                            render_market_content(last_msg.content, f"{key}_{len(st.session_state[key])-1}")
                        else:
                            st.markdown(last_msg.content)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if "selected_agent" in st.session_state:
    handle_chat_view(st.session_state.selected_agent)
