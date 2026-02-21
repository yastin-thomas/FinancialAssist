from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from src.core.tools import get_stock_price, get_company_info, query_knowledge_base, analyze_portfolio, get_market_data, calculate_investment_projection
import datetime
import os
import streamlit as st

# Configuration
DAILY_LLM_RATE_LIMIT = int(os.getenv("DAILY_LLM_RATE_LIMIT", 500))

class DailyRateLimitCallback(BaseCallbackHandler):
    # Fallbacks for non-streamlit environments (e.g., PyTest)
    fallback_count = 0
    fallback_date = datetime.date.today()

    def on_llm_start(self, serialized, prompts, **kwargs):
        today = datetime.date.today()
        
        # Check if we are inside a Streamlit context
        try:
            # Initialize session variables if they don't exist
            if "llm_current_date" not in st.session_state:
                st.session_state.llm_current_date = today
                st.session_state.llm_daily_count = 0
                
            # Reset counter if it's a new day for this session
            if today != st.session_state.llm_current_date:
                st.session_state.llm_current_date = today
                st.session_state.llm_daily_count = 0
                
            if st.session_state.llm_daily_count >= DAILY_LLM_RATE_LIMIT:
                raise Exception(f"Daily LLM rate limit of {DAILY_LLM_RATE_LIMIT} requests exceeded for this session. Please try again tomorrow.")
                
            st.session_state.llm_daily_count += 1
            print(f"--- Session LLM Invocation Count Today: {st.session_state.llm_daily_count}/{DAILY_LLM_RATE_LIMIT} ---")
            
        except Exception as e:
            # If st.session_state throws an exception (e.g., running outside Streamlit), 
            # or if we explicitly threw the rate limit exception above:
            if "rate limit" in str(e).lower():
                raise e # Re-raise the actual rate limit limit hit
                
            # Fallback logic for test suites
            if today != DailyRateLimitCallback.fallback_date:
                DailyRateLimitCallback.fallback_date = today
                DailyRateLimitCallback.fallback_count = 0
                
            if DailyRateLimitCallback.fallback_count >= DAILY_LLM_RATE_LIMIT:
                raise Exception(f"Daily LLM rate limit of {DAILY_LLM_RATE_LIMIT} requests exceeded. Please try again tomorrow.")
                
            DailyRateLimitCallback.fallback_count += 1
            print(f"--- Fallback LLM Invocation Count Today: {DailyRateLimitCallback.fallback_count}/{DAILY_LLM_RATE_LIMIT} ---")

# Initialize LLM Lazily
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            callbacks=[DailyRateLimitCallback()]
        )
    return _llm

# 1. Finance Q&A Agent
# Updated to use ONLY query_knowledge_base

def get_finance_qa_agent():
    return get_llm().bind_tools([query_knowledge_base])
    
finance_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful financial educator. Your goal is to explain financial concepts simply and clearly to beginners. "
               "ALWAYS use 'query_knowledge_base' to find accurate definitions and explanations from our trusted internal documents. "
               "Do not make up definitions. If the knowledge base returns no results, state that you cannot find the information in the trusted docs. "
               "CRITICAL: You MUST include the [Source: URL] from the tool output in your final answer."),
    ("placeholder", "{messages}")
])
class _FinanceQARunnable:
    def invoke(self, *args, **kwargs):
        return (finance_qa_prompt | get_finance_qa_agent()).invoke(*args, **kwargs)
finance_qa_runnable = _FinanceQARunnable()

# 2. Portfolio Analysis Agent
def get_portfolio_analysis_agent():
    return get_llm().bind_tools([get_stock_price, get_company_info, analyze_portfolio])
portfolio_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a portfolio analysis expert. You help users understand their investment portfolios. "
               "1. Use 'analyze_portfolio' to calculate total value and allocation. "
               "2. THEN, use 'get_company_info' for EACH significant holding to get background info. "
               "3. Finally, provide a comprehensive analysis including: "
               "   - Diversification check (are they all in the same sector?) "
               "   - Company summaries "
               "   - Actionable insights "
               "CRITICAL: The 'analyze_portfolio' and 'get_company_info' tools REQUIRE valid stock TICKER symbols (e.g., AAPL). "
               "If the user provides a company name (e.g., 'Apple', 'Tesla'), you MUST convert it to its exact ticker symbol (e.g., 'AAPL', 'TSLA') BEFORE calling the tool. "
               "DO NOT pass full company names to the tools under any circumstance. "
               "CRITICAL: The 'analyze_portfolio' tool returns a Markdown Table. You MUST output this table EXACTLY as provided. "
               "DO NOT convert it into a bulleted list. DO NOT summarize it. Output the raw Markdown Table."
               "DISCLAIMER: You are NOT a financial advisor or tax professional. Always include a disclaimer."
               "CRITICAL: You MUST include the [Source: URL] from the tool output in your final answer."),
    ("placeholder", "{messages}")
])
class _PortfolioAnalysisRunnable:
    def invoke(self, *args, **kwargs):
        return (portfolio_analysis_prompt | get_portfolio_analysis_agent()).invoke(*args, **kwargs)
portfolio_analysis_runnable = _PortfolioAnalysisRunnable()

# 3. Market Analysis Agent
def get_market_analysis_agent():
    return get_llm().bind_tools([get_stock_price, get_company_info, get_market_data])
market_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a market analyst. You track market trends and provide real-time insights. "
               "Use 'get_stock_price' to check current market status. "
               "Use 'get_market_data' when asked for a market overview or trends (S&P 500, Dow, Nasdaq). It defaults to 3 months (90 days). "
               "If you use 'get_market_data', you MUST output the returned Markdown Table EXACTLY as is. "
               "DO NOT summarize the table into text or bullets. Output the raw Markdown Table."
               "Provide context on why the market is moving."
               "CRITICAL: You MUST include the [Source: URL] from the tool output in your final answer."),
    ("placeholder", "{messages}")
])
class _MarketAnalysisRunnable:
    def invoke(self, *args, **kwargs):
        return (market_analysis_prompt | get_market_analysis_agent()).invoke(*args, **kwargs)
market_analysis_runnable = _MarketAnalysisRunnable()

# 4. Goal Planning Agent
def get_goal_planning_agent():
    return get_llm().bind_tools([calculate_investment_projection])
goal_planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """<persona>
You are The Financial Architect, a high-level financial planning assistant. Your tone is professional, encouraging, and analytical. You help users transform vague financial desires into concrete, mathematically sound roadmaps. You prioritize clarity and actionable steps over complex jargon.
</persona>

<interaction_protocol>
You will guide the user through a structured 3-phase process. Follow these phases strictly.

<phase name="1_Level_Assessment_and_Hook">
Do not provide advice immediately. Your first message must:
1. Briefly introduce yourself.
2. Ask the user for their experience level (e.g., Novice, Intermediate, or Pro).
3. Ask for one primary financial goal they are currently thinking about.
</phase>

<phase name="2_Iterative_Discovery">
Once a goal is identified, ask targeted questions to fill in the "Three Pillars":
- The Target: What is the specific dollar amount needed?
- The Timeline: When is the "due date" for this money?
- The Current State: What is already saved, and what is the monthly surplus available to contribute?
Also ask about Risk Tolerance (e.g., "If the market dipped 10% tomorrow, would you want to sell, stay the course, or buy more?").
Ask 2-3 questions per turn to gather this information.
</phase>

<phase name="3_The_Roadmap_Output">
Once data is gathered, generate a "Financial Blueprint" using a Markdown table.
The table MUST include these columns: Goal, Target Amount, Monthly Contribution Needed, Suggested Account Type (e.g., HYSA for short-term, Roth IRA/Brokerage for long-term).
Provide a "Feasibility Check": If the goal is mathematically impossible based on the numbers, suggest an adjusted timeline or contribution amount.
</phase>
</interaction_protocol>

<constraints>
- Disclaimer: You must include a brief disclaimer at the very end of the roadmap: "I am an AI, not a certified financial planner. This is a mathematical projection, not formal investment advice."
- No Walls of Text: Use bullet points and bold text to make responses highly scannable.
- One Step at a Time: Never ask more than three questions in a single response.
</constraints>

<instructions_for_tools>
- Use the 'calculate_investment_projection' tool to crunch numbers.
- NEVER attempt to calculate compounding interest on your own. Always defer to the tool to perform the "Feasibility Check".
- Assume a standard 0.07 annual return for long-term stock goals and 0.04 for HYSAs unless the user specifies otherwise.
</instructions_for_tools>"""),
    ("placeholder", "{messages}")
])
class _GoalPlanningRunnable:
    def invoke(self, *args, **kwargs):
        return (goal_planning_prompt | get_goal_planning_agent()).invoke(*args, **kwargs)
goal_planning_runnable = _GoalPlanningRunnable()

# 5. News Synthesizer Agent
# Replaced web search with knowledge base for context
def get_news_synthesizer_agent():
    return get_llm().bind_tools([query_knowledge_base]) 
news_synthesizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial news synthesizer. Your job is to summarize complex financial news into "
               "digestible insights. Use 'query_knowledge_base' to find background context and definitions for news events. "
               "Always cite your sources using the URL provided by the tool."),
    ("placeholder", "{messages}")
])
class _NewsSynthesizerRunnable:
    def invoke(self, *args, **kwargs):
        return (news_synthesizer_prompt | get_news_synthesizer_agent()).invoke(*args, **kwargs)
news_synthesizer_runnable = _NewsSynthesizerRunnable()

# 6. Tax Education Agent
# Removed web search
def get_tax_education_agent():
    return get_llm().bind_tools([query_knowledge_base])
tax_education_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tax education specialist. You explain tax concepts (e.g., capital gains, dividends, tax-advantaged accounts). "
               "DISCLAIMER: You are NOT a financial advisor or tax professional. Always include a disclaimer. "
               "ALWAYS use 'query_knowledge_base' to find accurate tax-related definitions via our internal docs. "
               "CRITICAL: You MUST include the [Source: URL] from the tool output in your final answer."),
    ("placeholder", "{messages}")
])
class _TaxEducationRunnable:
    def invoke(self, *args, **kwargs):
        return (tax_education_prompt | get_tax_education_agent()).invoke(*args, **kwargs)
tax_education_runnable = _TaxEducationRunnable()

# Router
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a router. Your job is to route the user's query to the most appropriate agent. "
               "Available agents: "
               "'finance_qa' (general concepts), "
               "'portfolio_analysis' (portfolio review), "
               "'market_analysis' (current market trends), "
               "'goal_planning' (setting goals), "
               "'news_synthesizer' (news summaries), "
               "'tax_education' (tax concepts). "
               "Return ONLY the name of the agent."),
    ("placeholder", "{messages}")
])
# Create router via property or getter so llm isn't called top-level
class RouterRunnable:
    def invoke(self, *args, **kwargs):
        runnable = router_prompt | get_llm()
        return runnable.invoke(*args, **kwargs)

router_runnable = RouterRunnable()
