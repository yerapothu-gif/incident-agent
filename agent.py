import os
from typing import Annotated, Literal, TypedDict
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg_pool import ConnectionPool

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# --- DATABASE POOL ---
pool = ConnectionPool(
    conninfo=os.getenv("DATABASE_URL"),
    max_size=10,
    kwargs={"autocommit": True, "connect_timeout": 15, "prepare_threshold": None},
)

rag_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    task_type="RETRIEVAL_QUERY",
    output_dimensionality=768,
)

# --- UTILITIES ---

def extract_text(content) -> str:
    """Safely extracts a flat string whether content is a str or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content or "")

# --- TOOLS ---

@tool
def query_service_health(service_name: str) -> str:
    """Checks health and uptime metrics of a backend microservice."""
    services = {
        "auth": "Auth Service: DEGRADED. High latency on token refresh endpoint.",
        "database": "Primary Postgres: HEALTHY. Read replicas at 22% CPU.",
        "payments": "Stripe Gateway: HEALTHY. No errors reported.",
    }
    return services.get(service_name.lower(), f"Service '{service_name}' not found.")


@tool
def search_remediation_runbooks(query: str) -> str:
    """Vector search over internal system runbooks in Supabase pgvector."""
    query_vector = rag_embeddings.embed_query(query)
    vec_str = f"[{','.join(str(x) for x in query_vector)}]"

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT metadata->>'service' as service, content, 1 - (embedding <=> %s::vector) as similarity
                FROM incident_docs
                ORDER BY embedding <=> %s::vector
                LIMIT 2;
                """,
                (vec_str, vec_str),
            )
            rows = cur.fetchall()

    if not rows:
        return "No relevant runbooks found."
    return "\n---\n".join([f"[{r[0]}]: {r[1]}" for r in rows])


@tool
def escalate_ticket(ticket_title: str, severity: str) -> str:
    """CRITICAL: Escalates an unresolved incident to the on-call engineering team."""
    return f"Ticket created: '{ticket_title}' [Severity: {severity.upper()}]. On-call team paged."


safe_tools = [query_service_health, search_remediation_runbooks]
sensitive_tools = [escalate_ticket]
all_tools = safe_tools + sensitive_tools

# --- STATE & ROUTING ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
).bind_tools(all_tools)


def call_model(state: AgentState):
    system_prompt = SystemMessage(
        content=(
            "You are an Autonomous Incident Triage Agent. "
            "1. Inspect system health first for degraded components. "
            "2. Search remediation runbooks using search_remediation_runbooks. "
            "3. If manual intervention is needed or service remains degraded, call escalate_ticket."
        )
    )
    response = llm.invoke([system_prompt] + state["messages"])
    # Normalize list-structured content to clean text to prevent UI/downstream type issues
    response.content = extract_text(response.content)
    return {"messages": [response]}


def route_tools(state: AgentState) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    if last_message.tool_calls[0]["name"] == "escalate_ticket":
        return "sensitive_tools"
    return "safe_tools"

# --- WORKFLOW COMPILATION ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("safe_tools", ToolNode(safe_tools))
workflow.add_node("sensitive_tools", ToolNode(sensitive_tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    route_tools,
    {
        "safe_tools": "safe_tools",
        "sensitive_tools": "sensitive_tools",
        END: END,
    },
)
workflow.add_edge("safe_tools", "agent")
workflow.add_edge("sensitive_tools", "agent")


def get_agent_app():
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["sensitive_tools"],
    )