import os
import json
from typing import List
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from google.auth import default
from vertexai import init

EMBEDDING_MODEL = "text-embedding-005"
LOCATION = "us-central1"
LLM_LOCATION = "global"
LLM = "gemini-2.0-flash-001"
PROJECT_ID = default()[1]

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LLM_LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

init(project=PROJECT_ID, location=LOCATION)

embedding = VertexAIEmbeddings(
    project=PROJECT_ID,
    location=LOCATION,
    model_name=EMBEDDING_MODEL,
)

def load_real_estate_data(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def listings_to_documents(listings: List[dict]) -> List[Document]:
    return [
        Document(
            page_content=json.dumps(entry, indent=2),
            metadata={"location": entry.get("location")}
        )
        for entry in listings
    ]

listings = load_real_estate_data("C:/Users/Asus/Desktop/real-estate-bot/real_estate_listings.json")
documents = listings_to_documents(listings)
faiss_index = FAISS.from_documents(documents, embedding)

@tool
def search_real_estate(query: str) -> str:
    """Searches real estate listings based on a user query."""
    results = faiss_index.similarity_search(query, k=5)
    if not results:
        return "No matching real estate listings found for your query."
    return "\n\n".join(doc.page_content for doc in results)

tools = [search_real_estate]

llm = ChatVertexAI(
    model=LLM,
    location=LLM_LOCATION,
    temperature=0.2,
    max_tokens=1024,
    streaming=True
).bind_tools(tools)

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = {
        "type": "system",
        "content": "You are an AI assistant for answering real estate-related queries. Your task is to respond based on the retrieved listings."
    }
    messages_with_system = [system_message] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

agent = workflow.compile()
