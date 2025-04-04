import os
import re
import logging
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

# Setup logging to a file
logging.basicConfig(
    filename="retrieval_flow.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

###############################################################################
# 1. Define State
###############################################################################
class State(TypedDict):
    question: str          # User question
    query_type: str        # 'sql' or 'general'
    query: str             # SQL query
    result: str            # SQL execution result
    answer: str            # Final answer to user
    validation_details: str

###############################################################################
# 2. LLM Initialization (by default) using gpt-4o-mini
###############################################################################
llm = get_openai_model("o3-mini", temperature=0.0)

def set_llm_for_sql_flow(model_id: str, temperature: float = 0.0, max_tokens: int = None):
    logging.info("==== set_llm_for_sql_flow: Start ====")
    try:
        global llm
        llm = get_openai_model(model_id, temperature=temperature, max_tokens=max_tokens)
        logging.info("==== set_llm_for_sql_flow: Success ====")
    except Exception as e:
        logging.error("==== set_llm_for_sql_flow: Error: %s ====", str(e), exc_info=True)
        raise e

###############################################################################
# 3. Embedding and Retrieval
###############################################################################
def get_relevant_info(query: str, k: int = 3) -> str:
    logging.info("==== get_relevant_info: Start ====")
    try:
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
        output = "\n".join(texts)
        logging.info("==== get_relevant_info: Success ====")
        return output
    except Exception as e:
        logging.error("==== get_relevant_info: Error: %s ====", str(e), exc_info=True)
        raise e

def generate_embedding(query: str) -> list[float]:
    logging.info("==== generate_embedding: Start ====")
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        embedding_vector = embeddings.embed_query(query)
        logging.info("==== generate_embedding: Success ====")
        return embedding_vector
    except Exception as e:
        logging.error("==== generate_embedding: Error: %s ====", str(e), exc_info=True)
        raise e

def semantic_search(query: str, k: int = 3) -> str:
    logging.info("==== semantic_search: Start ====")
    try:
        # Compute the embedding for the query
        query_vector = generate_embedding(query)
        # Format the embedding as a vector literal for SQL similarity search
        vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"
        # SQL query to retrieve top-k similar content from vector_embeddings
        sql_query = f"SELECT content FROM vector_embeddings ORDER BY embedding <-> '{vector_literal}' LIMIT {k};"
        # Execute the query against the database
        execute_query_tool = QuerySQLDatabaseTool(db=get_database())
        search_results = execute_query_tool.invoke(sql_query)
        # Combine the search results into a single string
        if isinstance(search_results, str):
            result_text = search_results
        else:
            result_text = "\n".join(str(r) for r in search_results)
        logging.info("==== semantic_search: Success ====")
        return result_text
    except Exception as e:
        logging.error("==== semantic_search: Error: %s ====", str(e), exc_info=True)
        raise e

###############################################################################
# 4. Question Type Analysis (Modified)
###############################################################################
analysis_system_prompt = """Analyze the user question and classify it into one of two categories:
- 'sql' if the question requires a SQL query to retrieve structured data from the travel_plans table (for example, counting records, computing averages, or recommendations that require filtering by location).
- 'general' if the question is about general information of the database (e.g., "What is the database about?", "Who were the authors of the database?", "What kind of data does the database have?").
Answer with a single word: 'sql' or 'general'."""
analysis_prompt = ChatPromptTemplate.from_messages([("system", analysis_system_prompt), ("human", "{question}")])
analysis_chain = analysis_prompt | llm | StrOutputParser()

def analyze_question(state: State) -> dict:
    logging.info("==== analyze_question: Start ====")
    try:
        result = analysis_chain.invoke({"question": state["question"]})
        query_type = "sql" if "sql" in result.lower() else "general"
        logging.info("==== analyze_question: Success - Query Type: %s ====", query_type)
        return {"query_type": query_type}
    except Exception as e:
        logging.error("==== analyze_question: Error: %s ====", str(e), exc_info=True)
        raise e

###############################################################################
# 5. Query Generation
###############################################################################
query_system_prompt = """You are an expert travel planner assistant with access to two main data sources:
1. A relational database table 'travel_plans' for structured travel data (trip origins, destinations, travel dates, number of days, etc.).
2. A vector-based knowledge base 'vector_embeddings' for experiential content (activities, restaurants, attractions, cultural experiences, etc.).

When answering the user's question:
- Use the 'travel_plans' table for SQL queries when the question requires structured data retrieval (e.g., counts, averages, or filtering by location).
- For recommendations and similar requests, use the 'vector_embeddings' table for semantic search.
- IMPORTANT: If the question involves a location filter (e.g., “restaurants in Chicago”), first query the 'travel_plans' table to extract relevant travel plan IDs mentioning the location. Then use those IDs to run a semantic search on the 'vector_embeddings' table.
- If the question involves both types of information, a hybrid approach may be used.

Relevant Information:
{table_info}

User Question: {input}

Now, draft an appropriate SQL query (or queries) to answer the user's question, using the sources as instructed. Only provide the SQL query, without any explanation."""
query_prompt = ChatPromptTemplate.from_messages([("system", query_system_prompt), ("human", "{input}")]).partial(dialect=get_database().dialect)

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State) -> dict:
    logging.info("==== write_query: Start ====")
    try:
        # Get relevant table info from the Chroma database
        relevant_table_info = get_relevant_info(state["question"])
        prompt_value = query_prompt.format_prompt(table_info=relevant_table_info, input=state["question"])
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt_value)
        logging.info("==== write_query: Success - Query Generated ====")
        return {"query": result["query"]}
    except Exception as e:
        logging.error("==== write_query: Error: %s ====", str(e), exc_info=True)
        raise e

###############################################################################
# 6. (Suppressed) Query Validation is omitted for SQL questions as per requirements.
###############################################################################
# The analyze_query_result function is defined below but will not be used in the main flow.
def analyze_query_result(state: State) -> dict:
    logging.info("==== analyze_query_result: Start ====")
    try:
        if state.get("query_status") == "success":
            prompt = (
                f"La siguiente consulta SQL se ejecutó exitosamente:\n"
                f"Consulta: {state['query']}\n"
                f"Resultado: {state['result']}\n"
                "Por favor, analiza por qué esta consulta funcionó bien y qué información provee."
            )
            response = llm.invoke(prompt)
            logging.info("==== analyze_query_result: Success - Query Analysis ====")
            return {"analysis": response.content, "fixed_query": None}
        else:
            error_msg = state["result"]
            prompt = (
                f"La siguiente consulta SQL falló con este error:\n"
                f"Consulta: {state['query']}\n"
                f"Error: {error_msg}\n"
                "Genera una versión corregida de esta consulta que solucione el error. "
                "Proporciona únicamente la consulta corregida, sin explicación."
            )
            response = llm.invoke(prompt)
            logging.info("==== analyze_query_result: Success - Generated Fixed Query ====")
            return {"analysis": f"La consulta original falló con el error: {error_msg}", "fixed_query": response.content}
    except Exception as e:
        error_msg = str(e)
        logging.error("==== analyze_query_result: Error: %s ====", error_msg)
        return {"analysis": f"Error durante el análisis: {error_msg}", "fixed_query": None}

###############################################################################
# 7. Query Execution
###############################################################################
def execute_query(state: State) -> dict:
    logging.info("==== execute_query: Start ====")
    try:
        query = state["query"]
        
        # If the query requires vector binding
        if "::vector" in query:
            embedding = generate_embedding(state["question"])
            vector_str = '[' + ','.join(map(str, embedding)) + ']'
            
            USER = os.getenv("DB_USER")
            PASSWORD = os.getenv("DB_PASSWORD")
            HOST = os.getenv("DB_HOST")
            PORT = os.getenv("DB_PORT")
            DBNAME = os.getenv("DB_NAME")
            
            try:
                connection = psycopg2.connect(
                    user=USER,
                    password=PASSWORD,
                    host=HOST,
                    port=PORT,
                    dbname=DBNAME
                )
                print("Connection successful!")
            except Exception as e:
                print(f"Error: {e}")
                return {"result": f"Error opening connection: {str(e)}", "query_status": "error"}
            
            cursor = connection.cursor()
            
            if "$1::vector" in query:
                modified_query = query.replace("$1::vector", "%s::vector")
                cursor.execute(modified_query, (vector_str,))
            elif "%s::vector" in query:
                if query.count("%s") > 1:
                    cursor.execute(query, (vector_str, vector_str))
                else:
                    cursor.execute(query, (vector_str,))
            else:
                final_query = query.replace("%s::vector", f"'{vector_str}'::vector") \
                                  .replace("$1::vector", f"'{vector_str}'::vector")
                cursor.execute(final_query)
            
            rows = cursor.fetchall()
            formatted_result = "\n".join([str(row) for row in rows])
            
            cursor.close()
            connection.close()
            result = formatted_result
        else:
            USER = os.getenv("DB_USER")
            PASSWORD = os.getenv("DB_PASSWORD")
            HOST = os.getenv("DB_HOST")
            PORT = os.getenv("DB_PORT")
            DBNAME = os.getenv("DB_NAME")
            
            try:
                connection = psycopg2.connect(
                    user=USER,
                    password=PASSWORD,
                    host=HOST,
                    port=PORT,
                    dbname=DBNAME
                )
                print("Connection successful!")
            except Exception as e:
                print(f"Error: {e}")
                return {"result": f"Error opening connection: {str(e)}", "query_status": "error"}
            
            cursor = connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            formatted_result = "\n".join([str(row) for row in rows])
            
            cursor.close()
            connection.close()
            result = formatted_result
        
        logging.info("==== execute_query: Success - SQL Query Executed ====")
        return {"result": result, "query_status": "success"}
    except Exception as e:
        error_msg = str(e)
        logging.error("==== execute_query: Error: %s ====", error_msg)
        return {"result": f"Error: {error_msg}", "query_status": "error"}

def execute_vector_query(state: State) -> dict:
    logging.info("==== execute_vector_query: Start ====")
    try:
        search_results = semantic_search(state["question"])
        logging.info("==== execute_vector_query: Success - Vector Query Executed ====")
        return {"result": search_results}
    except Exception as e:
        logging.error("==== execute_vector_query: Error: %s ====", str(e), exc_info=True)
        raise e

###############################################################################
# 8. Final Answer Generation
###############################################################################
def generate_answer(state: State) -> dict:
    logging.info("==== generate_answer: Start ====")
    try:
        if state.get("query_type") == "general":
            prompt = (
                f"Given the following user question:\n"
                f'Question: {state["question"]}\n'
                f'Retrieved Information: {state["result"]}\n'
                "Please provide a concise, helpful answer to the user."
            )
        else:  # sql branch
            prompt = (
                f"Given the following user question:\n"
                f'Question: {state["question"]}\n'
                f'SQL Query: {state["query"]}\n'
                f'SQL Result: {state["result"]}\n'
                "Please provide a concise, helpful answer to the user."
            )
        response = llm.invoke(prompt)
        logging.info("==== generate_answer: Success - Answer Generated ====")
        return {"answer": response.content}
    except Exception as e:
        logging.error("==== generate_answer: Error: %s ====", str(e), exc_info=True)
        raise e

###############################################################################
# 9. Build the Graph
###############################################################################

from langchain_core.runnables import RunnableLambda
import psycopg2
import os
import re
from dotenv import load_dotenv

load_dotenv()
   
logging.info("==== Building the Graph: Start ====")
try:
    # (Re)define execute_query to use psycopg2 directly with proper % escaping
    def execute_query(state: State) -> dict:
        logging.info("==== execute_query: Start ====")
        try:
            query = state["query"]
            
            # Check for vector search query
            has_vector = "::vector" in query
            
            # If using vector embedding
            if has_vector:
                embedding = generate_embedding(state["question"])
                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                
                # Connect to database
                USER = os.getenv("DB_USER")
                PASSWORD = os.getenv("DB_PASSWORD")
                HOST = os.getenv("DB_HOST")
                PORT = os.getenv("DB_PORT")
                DBNAME = os.getenv("DB_NAME")
                
                try:
                    connection = psycopg2.connect(
                        user=USER,
                        password=PASSWORD,
                        host=HOST,
                        port=PORT,
                        dbname=DBNAME
                    )
                    print("Connection successful!")
                except Exception as e:
                    print(f"Error: {e}")
                    return {"result": f"Error opening connection: {str(e)}", "query_status": "error"}
                
                cursor = connection.cursor()
                
                # IMPORTANT: Escape % in LIKE/ILIKE patterns by doubling them before parameter substitution
                if "LIKE '%" in query or "ILIKE '%" in query:
                    # Replace single % with %% in LIKE patterns but not in %s::vector
                    pattern = r"(LIKE|ILIKE)\s+'%([^']*?)%'"
                    query = re.sub(pattern, r"\1 '%%\2%%'", query)
                
                # Handle different parameter styles
                if "$1::vector" in query:
                    modified_query = query.replace("$1::vector", "%s::vector")
                    cursor.execute(modified_query, (vector_str,))
                elif "%s::vector" in query:
                    # Count how many vector parameters we need
                    param_count = query.count("%s::vector")
                    params = tuple([vector_str] * param_count)
                    cursor.execute(query, params)
                else:
                    final_query = query.replace("%s::vector", f"'{vector_str}'::vector") \
                                       .replace("$1::vector", f"'{vector_str}'::vector")
                    cursor.execute(final_query)
                    
                rows = cursor.fetchall()
                formatted_result = "\n".join([str(row) for row in rows])
                cursor.close()
                connection.close()
                result = formatted_result
            else:
                # For non-vector queries, still handle % escaping
                USER = os.getenv("DB_USER")
                PASSWORD = os.getenv("DB_PASSWORD")
                HOST = os.getenv("DB_HOST")
                PORT = os.getenv("DB_PORT")
                DBNAME = os.getenv("DB_NAME")
                
                try:
                    connection = psycopg2.connect(
                        user=USER,
                        password=PASSWORD,
                        host=HOST,
                        port=PORT,
                        dbname=DBNAME
                    )
                    print("Connection successful!")
                except Exception as e:
                    print(f"Error: {e}")
                    return {"result": f"Error opening connection: {str(e)}", "query_status": "error"}
                
                cursor = connection.cursor()
                
                # Escape % in LIKE patterns for regular queries too
                if "LIKE '%" in query or "ILIKE '%" in query:
                    pattern = r"(LIKE|ILIKE)\s+'%([^']*?)%'"
                    query = re.sub(pattern, r"\1 '%%\2%%'", query)
                
                cursor.execute(query)
                rows = cursor.fetchall()
                formatted_result = "\n".join([str(row) for row in rows])
                cursor.close()
                connection.close()
                result = formatted_result
                
            logging.info("==== execute_query: Success - SQL Query Executed ====")
            return {"result": result, "query_status": "success"}
        except Exception as e:
            error_msg = str(e)
            logging.error("==== execute_query: Error: %s ====", error_msg)
            return {"result": f"Error: {error_msg}", "query_status": "error"}
    
    # Resto del código de construcción del grafo...
    # Wrapper to correct placeholders if needed
    def write_query_wrapper(state: State) -> dict:
        result = write_query(state)
        if "query" in result and "$1::vector" in result["query"]:
            result["query"] = result["query"].replace("$1::vector", "%s::vector")
        return result
    
    # Conditional routing: if question is 'sql', go to write_query; else go directly to execute_vector_query.
    def route_execution(state: State) -> str:
        if state["query_type"] == "sql":
            return "write_query"
        else:
            return "execute_vector_query"
    
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("analyze_question", analyze_question)
    graph_builder.add_node("write_query", write_query_wrapper)  # Generates SQL query
    graph_builder.add_node("execute_query", execute_query)
    graph_builder.add_node("execute_vector_query", execute_vector_query)
    graph_builder.add_node("generate_answer", generate_answer)
    
    # Define graph edges based on the simplified flow
    graph_builder.add_edge(START, "analyze_question")
    
    # Create the routing function
    named_route_execution = RunnableLambda(route_execution, name="route_execution")
    
    # Add conditional edges directly from "analyze_question" 
    graph_builder.add_conditional_edges("analyze_question", path=named_route_execution)
    
    # For SQL questions: write query then execute it, then generate answer.
    graph_builder.add_edge("write_query", "execute_query")
    graph_builder.add_edge("execute_query", "generate_answer")
    
    # For general questions: directly execute vector query then generate answer.
    graph_builder.add_edge("execute_vector_query", "generate_answer")
    
    # Compile the graph and store it
    compiled_graph = graph_builder.compile()
    
    # Generate a Mermaid diagram of the graph.
    output_file_path = "visualization_graph.png"
    compiled_graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
        output_file_path=output_file_path
    )
    
    logging.info("==== Building the Graph: Success ====")
except Exception as e:
    logging.error("==== Building the Graph: Error: %s ====", str(e), exc_info=True)
    raise e



###############################################################################
# 10. Main Execution Function
###############################################################################
def run_sql_flow(
    user_question: str,
    model_id: str = None,
    temperature: float = 0.0,
    max_tokens: int = None
) -> tuple[list[dict], dict]:
    logging.info("==== run_sql_flow: Start ====")
    try:
        if model_id:
            set_llm_for_sql_flow(model_id, temperature, max_tokens)
        
        # Use compiled_graph instead of graph_builder
        steps = list(compiled_graph.stream({"question": user_question}, stream_mode="updates"))
        
        final = steps[-1]
        logging.info("==== run_sql_flow: Success ====")
        return steps, final
    except Exception as e:
        logging.error("==== run_sql_flow: Error: %s ====", str(e), exc_info=True)
        raise e


# Note: To execute the complete flow (including SQL query execution), ensure that the execute_query function
# is properly connected in the graph sequence and that your database and vector store are accessible.
