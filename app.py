import os
from typing import Annotated, Sequence, TypedDict
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


from langchain_core.messages import BaseMessage, HumanMessage

from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. PROVIDE YOUR GROQ API KEY HERE (Replace this with a fresh one if you reset it)
os.environ["GROQ_API_KEY"] = ""

# Ensure tokenizers operate quietly on Windows


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# PHASE A: KNOWLEDGE BASE PREPROCESSING
# ==========================================

def setup_vector_store():
    print("Initializing document preprocessing and vector store...")
    if not os.path.exists("data") or not os.listdir("data"):
        print("⚠️ Warning: 'data/' directory is missing or empty. Please insert documents.")
        return None

    # Load local documents

    loader = PyPDFDirectoryLoader("data/")
    
    docs = loader.load()
    
    # Text Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # 100% Free Embedding Engine (Runs locally, no API credits required)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # In-memory vector database instance
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_vector_store()

# ==========================================
# PHASE B: CUSTOM AGENT TOOLS
# ==========================================

# We use a Pydantic class to force Groq to receive a flawless JSON schema definition
class SearchInput(BaseModel):

    query: str = Field(description="The search query text to look up in the documentation.")

@tool(args_schema=SearchInput)
def query_knowledge_base(query: str) -> str:
    """Searches the local document directory for master's module data or specific domain knowledge."""
    if retriever is None:
        return "The document database is currently empty. Suggest the user upload files or answer using general knowledge."
    
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant documents found in the database."
        
    context_blocks = [doc.page_content for doc in retrieved_docs]
    return "\n\n---\n\n".join(context_blocks)

tools = [query_knowledge_base]

# ==========================================
# PHASE C: THE CUSTOM LANGGRAPH SYSTEM
# ==========================================

# 1. Define Graph State FIRST so nodes can use it
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# System instructions to prevent formatting slip-ups
system_prompt = (
    "You are an advanced Agentic RAG assistant. You have access to a local knowledge base tool.\n"
    "1. If the user asks a general knowledge question not related to your specific documents, "
    "or if the tool tells you the database is empty, answer directly using your internal knowledge.\n"
    "2. If you need to use a tool, use the standard tool calling mechanism. Do not type raw XML tags."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Initialize Groq Engine safely
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)




# Node Action: LLM processing step with explicit prompt binding
def call_agent_brain(state: AgentState):
    messages = state['messages']
    # Chain the prompt template before the LLM to provide strict formatting rules
    chain = prompt_template | llm_with_tools
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}

# Conditional Node Router
def determine_routing(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    
    # Check if the LLM successfully generated a valid tool call structural request
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    
    return END

# Build Graph from Scratch
workflow = StateGraph(AgentState)

workflow.add_node("agent_brain", call_agent_brain)
workflow.add_node("execute_tools", ToolNode(tools))

workflow.add_edge(START, "agent_brain")
workflow.add_conditional_edges(
    "agent_brain", 
    determine_routing,
    {
        "execute_tools": "execute_tools",
        END: END
    }
)
workflow.add_edge("execute_tools", "agent_brain")

memory_checkpoint = MemorySaver()
compiled_agent = workflow.compile(checkpointer=memory_checkpoint)


# ==========================================
# PHASE D: GENERATE SYSTEM BLUEPRINT
# ==========================================
try:
    with open("graph_architecture.png", "wb") as file:
        file.write(compiled_agent.get_graph().draw_mermaid_png())
    print(" Success: Graph blueprint image generated as 'graph_architecture.png'.")
except Exception as error:
    print(f"Could not generate blueprint visual file natively: {error}")

# ==========================================
# PHASE E: INTERACTIVE CLIENT INTERFACE
# ==========================================
import time


if __name__ == "__main__":
    session_config = {"configurable": {"thread_id": "groq_session_active"}}

    print("\n" + "="*50)
    print(" Groq-Powered Custom Agentic RAG Framework Initialized.")
    print(" Type 'exit' at any time to quit the prompt.")
    print("="*50 + "\n")

    count = 1

    while True:
        prompt = input(f"Ask your question to GROP (Q{count}): ")

        if prompt.strip().lower() == "exit":
            break

        if not prompt.strip():
            continue


        start_time = time.perf_counter()

        execution_stream = compiled_agent.stream(
            {"messages": [HumanMessage(content=prompt)]},
            session_config
        )

        for iteration in execution_stream:
            for step_key, response_payload in iteration.items():
                if "messages" in response_payload:
                    last_msg = response_payload["messages"][-1]

                    if last_msg.content:
                        print(
                            f"\n[Node: {step_key}] Assistant: {last_msg.content}\n"
                        )

        end_time = time.perf_counter()


        elapsed = end_time - start_time

        print(f"Response time: {elapsed:.2f} seconds\n")

        count += 1