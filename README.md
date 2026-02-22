# FinancialAssist AI

FinancialAssist is a comprehensive AI-powered financial advisory application built with LangGraph, LangChain, and Streamlit. It orchestrates specialized AI agents to help users with financial education, portfolio analysis, market trends, goal planning, and tax concepts.

## 🏗️ Architecture Overview

The system uses a **Multi-Agent Orchestration Architecture** powered by `LangGraph` to route user intent to specialized expert agents. A central **Router Agent** assesses the user's prompt and forwards the conversation to one of six specialized LLM agents:

1. **Finance Q&A Agent**: Educates users on financial concepts using the internal knowledge base.
2. **Portfolio Analysis Agent**: Evaluates user portfolios, fetches current stock prices, and calculates allocations.
3. **Market Analysis Agent**: Retrieves current market trends and fetches historical data (S&P 500, Dow Jones, Nasdaq).
4. **Goal Planning Agent**: Helps users define SMART financial goals.
5. **News Synthesizer Agent**: Contextualizes financial news utilizing trusted definitions.
6. **Tax Education Agent**: Explains tax implications using internal educational documents.

Data flows from the user interface (`Streamlit`) -> `StateGraph` Router -> Specialized Agent -> Execution Tools -> User UI.

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([get one here](https://aistudio.google.com/app/apikey))

### 2. Installation
Clone the repository and install the required dependencies:
```bash
# Create a virtual environment
python -m venv myenv
myenv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize the Knowledge Base (RAG)
Before asking educational questions, populate the FAISS vector database:
```bash
# 1. Scrape trusted financial articles
python src/data/scrape_data.py

# 2. Generate embeddings and build FAISS DB
python src/rag/embedding.py
```

### 4. Run the Application
Start the Streamlit UI:
```bash
python -m streamlit run src/web_app/main.py
```

### 5. Run the Tests
To ensure the application is functioning properly, you can run the test suite using `pytest`. This repository includes comprehensive tests across the `src/` modules.

```bash
# Run all tests
python -m pytest tests/

# Run all tests with coverage report
python -m pytest --cov=src tests/
```

---

## 🔄 End-to-End Workflow

Follow these steps in order every time you set up or refresh the application:

```
1. Configure environment
         ↓
2. Install dependencies
         ↓
3. Scrape financial documents
         ↓
4. Build FAISS vector database (Gemini embeddings)
         ↓
5. Launch the Streamlit UI
         ↓
6. (Optional) Run test suite
```

### Step-by-step

**Step 1 — Configure environment**
Copy `.env` and fill in your keys:
```bash
GOOGLE_API_KEY=your_google_api_key   # Required – Gemini LLM + embeddings
GROQ_API_KEY=your_groq_api_key       # Optional – alternative LLM provider
```

**Step 2 — Install dependencies**
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**Step 3 — Scrape financial documents**
```bash
python src/data/scrape_data.py
# Output: financial_docs/ directory populated with .txt files
```

**Step 4 — Build the FAISS vector database**
```bash
python src/rag/embedding.py
# Output: financial_db/ directory with Gemini-generated embeddings
```
> ⚠️ Re-run this step whenever new documents are scraped or models are changed.

**Step 5 — Launch the app**
```bash
python -m streamlit run src/web_app/main.py
# Opens at http://localhost:8501
```

**Step 6 — Run the test suite** *(optional)*
```bash
python -m pytest tests/                    # All tests
python -m pytest --cov=src tests/          # With coverage report
```

---

## 🔌 API Documentation (Internal Tools)

Agents are provided with a shared suite of tools defined in `src/core/tools.py`:

- `get_stock_price(ticker: str)`: Fetches the latest daily closing price for a given ticker via Yahoo Finance.
- `get_company_info(ticker: str)`: Retrieves sector, industry, and a business summary for a given ticker.
- `get_market_data(period: str)`: Retrieves historical performance for major US indices (default: `3mo`).
- `analyze_portfolio(positions: str)`: Takes a JSON string map of tickers to quantities (e.g., `{"AAPL": 10}`) and returns a markdown table calculating current asset allocation and total value.
- `query_knowledge_base(query: str)`: Embeds the search query and retrieves the top 3 most relevant documents from the local FAISS vector database.

## 💡 Usage Examples

- **General Q&A**: "What is the difference between a traditional IRA and a Roth IRA?"
  - *Router -> Tax Education Agent -> `query_knowledge_base` -> Response with trusted URL citations.*
- **Portfolio Review**: "Can you analyze my portfolio? I have 50 shares of AAPL and 10 shares of TSLA."
  - *Router -> Portfolio Analysis Agent -> `analyze_portfolio` & `get_company_info` -> Emits detailed portfolio pie chart and breakdown.*
- **Market Update**: "How has the market been doing over the last 3 months?"
  - *Router -> Market Analysis Agent -> `get_market_data` -> Emits interactive line chart of S&P 500, Nasdaq, and Dow Jones.*

## 🆘 Troubleshooting Guide

**1. `RuntimeError: Event loop is closed` or `missing ScriptRunContext`**
This is caused by background threads created by `yfinance` conflicting with Streamlit and LangChain's asyncio loop.
*Fix*: Ensure you are not wrapping `_fetch_market_data` with `@st.cache_data` in multiple threads, and let LangChain handle the async tool execution naturally. 

**2. Vector DB Not Found**
Error: `Knowledge base not available.`
*Fix*: The `financial_db` directory is missing. Run `python src/rag/embedding.py` to compile the FAISS database first.

**3. Streamlit shows "Empty Portfolio provided"**
*Fix*: Ensure the query clearly states the ticker symbol and the exact number of shares (e.g. "10 shares of MSFT").

**4. NameError: name 'finance_qa_agent' is not defined in tests**
*Fix*: Agents are lazily initialized via getter functions (e.g., `get_finance_qa_agent()`) to prevent premature API key requirements. Ensure your tests call these getter functions instead of static variable names.
