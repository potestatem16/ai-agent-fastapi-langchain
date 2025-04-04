# app\core\retrieval_flow.py

import os
import re
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from langchain_core.messages import SystemMessage, HumanMessage

# Graph
from langgraph.graph import START, StateGraph
from langchain_core.runnables.graph import MermaidDrawMethod

# Database
from app.db.database import get_database

# Use OpenAI model (gpt-4o-mini) instead of Bedrock
from app.services.openai import get_openai_model

# Chroma Imports for vector search
import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

###############################################################################
# 1. Define State
###############################################################################
class State(TypedDict):
    question: str   # User question
    query: str      # SQL query
    result: str     # SQL execution result
    answer: str     # Final answer to user
    validation_details: str

###############################################################################
# 2. LLM Initialization (by default) using gpt-4o-mini
###############################################################################
llm = get_openai_model("gpt-4o-mini", temperature=0.0)

def set_llm_for_sql_flow(model_id: str, temperature: float = 0.0, max_tokens: int = None):
    global llm
    llm = get_openai_model(model_id, temperature=temperature, max_tokens=max_tokens)

###############################################################################
# 3. Chroma Relevant Info Retrieval
###############################################################################
def get_relevant_info(query: str, k: int = 3) -> str:
    """
    Performs a similarity search in the Chroma vector database to retrieve
    relevant table information.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    chroma_client = chromadb.PersistentClient(
        path="chroma_db/",  # Path to the Chroma database
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
    vector_store = Chroma(
        client=chroma_client,
        collection_name='travelplanner_sections',  # The desired collection name
        embedding_function=embeddings
    )
    results = vector_store.similarity_search(query, k=k)
    texts = []
    for doc in results:
        if hasattr(doc, "metadata") and doc.metadata.get("section"):
            texts.append(f"{doc.metadata.get('section')}: {doc.page_content}")
        else:
            texts.append(doc.page_content)
    return "\n".join(texts)

###############################################################################
# 4. Query Generation
###############################################################################
from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
assert len(query_prompt_template.messages) == 1

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    # Get relevant table info from the Chroma database
    relevant_table_info = get_relevant_info(state["question"])
    
    prompt = query_prompt_template.invoke(
        {
            "dialect": get_database().dialect,
            "top_k": 10,
            "table_info": relevant_table_info,  # Using Chroma search result instead of db.get_table_info()
            "input": state["question"],
        }
    )

    # Convert the first SystemMessage to HumanMessage, if needed
    old_messages = prompt.messages
    new_messages = []
    for i, m in enumerate(old_messages):
        if i == 0 and isinstance(m, SystemMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(m)

    new_prompt = ChatPromptTemplate.from_messages(new_messages)
    prompt_value = new_prompt.format_prompt()

    # The LLM is set up to return structured output
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt_value)
    return {"query": result["query"]}

###############################################################################
# 5. Query Validation
###############################################################################
query_check_system_prompt = """Double-check the user's {dialect} query for correctness and potential improvements.
If you provide a corrected SQL, wrap it in triple backticks, e.g.:
```sql
SELECT ...
```
"""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system_prompt), ("human", "{query}")]
).partial(dialect=get_database().dialect)

validation_chain = query_check_prompt | llm | StrOutputParser()

def check_query(state: State) -> dict:
    validated_query = validation_chain.invoke({"query": state["query"]})

    # Look for triple backticks ```sql ... ```
    match = re.search(r"```(?:sql)?\s*(.+?)```", validated_query, re.DOTALL)
    if match:
        query_only = match.group(1).strip()
    else:
        query_only = validated_query.strip()

    # Return both the pure query (for execution) and full explanation.
    return {
        "query": query_only,
        "validation_details": validated_query
    }

###############################################################################
# 6. Query Execution (commented out for now)
###############################################################################
def execute_query(state: State) -> dict:
    execute_query_tool = QuerySQLDatabaseTool(db=get_database())
    return {"result": execute_query_tool.invoke(state["query"])}

# def dummy_execute_query(state: State) -> dict:
#     # Add a dummy result to avoid missing the "result" key.
#     return {"result": "Query execution is currently disabled."}

###############################################################################
# 7. Final Answer Generation
###############################################################################
def generate_answer(state: State) -> dict:
    """
    Generates the final answer based on the user question, the SQL query, and the result.
    """
    prompt = (
        "Given the following user question:\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}\n'
        "Please provide a concise, helpful answer to the user."
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

###############################################################################
# 8. Build the Graph
###############################################################################
# The sequence is built without execute_query for now; you can uncomment it later.
graph_builder = StateGraph(State).add_sequence(
    [write_query, check_query,  
     execute_query 
    # dummy_execute_query #  <-- commented out for now
     , 
     generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# # (Optional) Generate a Mermaid diagram
# output_file_path = "visualization_graph.png"
# graph.get_graph().draw_mermaid_png(
#   draw_method=MermaidDrawMethod.API,
#   output_file_path=output_file_path
# )

###############################################################################
# 9. Main Execution Function
###############################################################################
def run_sql_flow(
    user_question: str,
    model_id: str = None,
    temperature: float = 0.0,
    max_tokens: int = None
    ) -> tuple[list[dict], dict]:
    """
    Runs the SQL flow pipeline for a given user question.

    :param user_question: The question from the user.
    :param model_id: Optional inference profile to switch the LLM used.
    :return: A tuple (list_of_intermediate_states, final_state).
    """
    
    if model_id:
        set_llm_for_sql_flow(model_id, temperature, max_tokens)

    steps = list(graph.stream({"question": user_question}, stream_mode="updates"))
    final = steps[-1]
    return steps, final



# Note:
# To execute the complete flow (including query execution), simply uncomment the execute_query function
# and add it back into the graph sequence in section 8.