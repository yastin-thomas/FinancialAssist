# Technical Design Document: FinancialAssist

## 1. System Architecture Decisions

### 1.1 LangGraph Orchestration
Instead of relying on a monolithic agent with a bloated system prompt, the system employs **LangGraph** to build a `StateGraph`. This allows for deterministic routing and isolated agent capabilities. 
- **Advantage**: If the `Tax Education` agent needs an update, it can safely be modified without regressing the behavior of the `Portfolio Analysis` agent.
- **State Management**: The graph maintains an `AgentState` containing the `messages` list and the `sender`. This ensures that tools know exactly which agent initiated a request, allowing the response edge (`route_back_to_agent`) to precisely return tool outputs to the original caller.

### 1.2 Lazy Initialization
Language Model instances (`ChatOpenAI`) and Vector Databases (`FAISS`) are wrapped in lazy-loading getters (e.g., `get_llm()`, `get_vector_db()`).
- **Advantage**: Decouples the initialization of heavy or credential-dependent objects from python module import time. This significantly improves unit test execution speeds and prevents `OPENAI_API_KEY` environment crashes during CI/CD steps.

### 1.3 UI Decoupling (Streamlit)
Streamlit is heavily utilized for rendering rich interactive charts (Plotly Pie and Line charts) and handling the chat UI.
- **Decision**: Streamlit UI code and decorators (like `@st.cache_data`) are strictly isolated to `src/web_app/main.py`. Tool definitions in `src/core/tools.py` remain entirely generic Python functions returning markdown text.
- **Advantage**: The tools and LangGraph core can theoretically be migrated to a FastAPI backend or a Discord bot without changing any core logic or tests. UI rendering relies on parsing the standardized Markdown tables emitted by the agents.

---

## 2. Agent Communication Protocols

### 2.1 The Router Node
1. The user inputs a message.
2. The `router_logic` conditional edge is invoked. It feeds the conversation history to a lightweight `router_runnable` (powered by GPT-4o).
3. The Router outputs a single string explicitly matching one of the predefined agent keys (e.g., `portfolio_analysis`).
4. We utilize an `AGENT_MAPPING` Exact-Match Dictionary to defensively map strings. Fuzzy substring fallback matching is included to handle unexpected LLM casing or prefixing, resolving directly to the target node.

### 2.2 Tool Execution Cycle
1. Once routed, the specialized Agent receives the `AgentState`.
2. If the LLM determines a tool is needed, it emits an `AIMessage` containing a `tool_calls` payload.
3. The Graph's `should_continue` edge intercepts this payload and redirects flow to the universal `ToolNode`.
4. The `ToolNode` executes the python function and emits a `ToolMessage`.
5. The `route_back_to_agent` checks the `state["sender"]` variable and conditionally paths back to the exact agent that originally asked for the data.
6. The Agent synthesizes the `ToolMessage` and generates a final `AIMessage` for the user. Flow returns to `END`.

---

## 3. RAG Implementation Details

The Retrieval-Augmented Generation (RAG) system is built entirely locally (excluding the LLM API) for cost efficiency and privacy.

### 3.1 Data Acquisition (`scrape_data.py`)
- We use the `newspaper3k` library to fetch and parse ~70+ curated Investopedia articles.
- Articles are categorized logically (`Taxes`, `Retirement`, `Markets`) by deducing keywords dynamically via `urlparse` and title checks.
- Data is cleansed of HTML formatting and serialized into flat `.txt` files containing strict Header metadata (`Title`, `Source`, `Category`).

### 3.2 Vectorization (`embedding.py`)
- Uses `langchain_community.document_loaders.DirectoryLoader` to bulk inject the scraped text.
- `RecursiveCharacterTextSplitter` aggressively chunks the documents (chunk size: 1000, overlap: 200) ensuring that definitions aren't severed mid-sentence.
- `GoogleGenerativeAIEmbeddings` convert chunks to vector space, which are pushed to a `FAISS` local database and serialized to disk (`/financial_db`).

### 3.3 Retrieval (`tools.py` -> `query_knowledge_base`)
- The `query_knowledge_base` tool wraps the FAISS similarity search.
- When an agent calls this tool, FAISS computes the cosine similarity against the user query and returns the top 3 chunks (k=3).
- **Enforced Citation System**: The tool formats the returned text explicitly appending the metadata `[Source: URL]` string. The internal system prompts of the Agents (e.g., `Finance Q&A`) are enforced via prompt-engineering to blindly echo this citation URL whenever they use RAG, creating a transparent, highly-trusted UX preventing hallucination.
