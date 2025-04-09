# Intelligent AI Agent Travel Planner

## 🧭 What is the Project About?

We’re building an intelligent **travel assistant** that can understand natural language questions and retrieve answers from both **structured** and **unstructured** data sources. The system is designed with a dual-path architecture, enabling it to intelligently route questions based on their nature:

### 🛤️ Two Retrieval Paths

1. **SQL-Based Retrieval**  
   For questions that require direct access to structured information — such as trip dates, durations, destinations, or cost — the system generates and executes a SQL query over the `travel_plans` table in **Supabase** (PostgreSQL).

2. **Vector-Based Semantic Search**  
   For broader, more general questions — like "What kind of data does this table contain?" or "What are some popular activities?" — the assistant performs a semantic search over an internal **Chroma vector database**, which stores embedded descriptions and metadata.

The assistant uses **language models and embeddings** to classify the question and automatically decide which path to follow in order to generate the most accurate and helpful answer.


## 🚀 Tech Stack

The chosen tools for this retrieval system are not only **state-of-the-art** but also **battle-tested**, **scalable**, and aligned with modern best practices in API-driven AI applications. 

### 🗃️ Supabase (Firebase Alternative)

- **Why Supabase?**  
  Supabase offers a powerful open-source alternative to Firebase, built directly on **PostgreSQL**. It combines real-time capabilities, built-in authentication, and scalable storage in a developer-friendly environment.
- **Key Benefits**:
  - SQL-native and supports advanced querying.
  - Built-in REST and GraphQL APIs.
  - Scales effortlessly for production use.
  - Seamlessly integrates with vector-aware extensions for hybrid search use cases.

### 🧠 Chroma (Vector Store)

- **Why Chroma?**  
  Chroma is a next-generation vector database purpose-built for **semantic search and embeddings**, designed to work with modern LLM pipelines.
- **Key Benefits**:
  - Blazing-fast similarity search.
  - Native support for storing, indexing, and filtering high-dimensional vectors.
  - Integrates seamlessly with LangChain and OpenAI embeddings.
  - Excellent for managing unstructured knowledge, metadata, and table embeddings.

### ⚡ FastAPI + Uvicorn

- **Why FastAPI and Uvicorn?**  
  FastAPI is one of the most modern and efficient frameworks for building APIs with Python, paired with **Uvicorn** as a lightning-fast ASGI server.
- **Key Benefits**:
  - Asynchronous, non-blocking I/O for high performance.
  - Auto-generated documentation (Swagger / ReDoc).
  - Minimal boilerplate and fast development cycles.
  - Ideal for production-grade AI microservices and endpoints.

### 🔗 LangChain & LangGraph

- **Why LangChain and LangGraph?**  
  LangChain is the most widely adopted framework for **orchestrating LLM-powered workflows**, while LangGraph builds on top of it to enable **stateful, graph-based flow control**.
- **Key Benefits**:
  - Abstractions for memory, agents, and multi-step chains.
  - LangGraph brings conditional branching and DAG-style execution.
  - Highly modular and composable.
  - Perfect for designing reusable, explainable LLM systems.

### 🤖 GPT Models via OpenAI API

- **Why OpenAI API?**  
  Leveraging the GPT family of models via the OpenAI API ensures access to **cutting-edge natural language capabilities** backed by world-class infrastructure.
- **Key Benefits**:
  - State-of-the-art performance in language understanding and generation.
  - Supports function calling, tool use, and structured outputs.
  - Scalable, secure, and backed by SLAs.
  - Regularly updated with the latest research and optimizations.


## 📁 Folder Structure Explained (Layer by Layer)

### 📌 **Top-Level Files**
| File / Folder | Purpose |
|---------------|---------|
| `.env` | Stores environment variables (DB credentials, API keys) |
| `requirements.txt` / `pyproject.toml` | Define Python dependencies |
| `README.md` | Project documentation |
| `retrieval_flow.log` | Logging output from pipeline executions |
| `retrieval_flow.log` | Logs info |
| `archive/` | Stores intermediate results or fallback answers |
| `data/` | Raw and processed datasets (CSV, PKL, JSON) |
| `chroma_db/` | Persistent vector store (Chroma DB files) |
| `scripts/` | One-off or utility scripts for building or testing |

---

#### ⚙️ `app/config.py`

- Manages **configuration** logic — e.g., environment loading, constants


#### 🧠 `app/core/retrieval_flow_v3.py`

This is the heart of the system — the **LangGraph flow**.

It includes:
- State definition (`State`)
- Routing logic between SQL and vector search
- SQL execution, embedding generation
- Answer generation
- Mermaid graph export

#### 📡 `app/api/sql_flow.py`

- Exposes an **API route** (like `POST /ask`) that receives user questions
- Internally calls `run_sql_flow()` from `retrieval_flow_v3.py`
- Parses and returns the final answer as JSON

This file **bridges the LangGraph backend with the FastAPI frontend**.


#### 🗃️ `app/db/database.py`

- Contains functions for **connecting to PostgreSQL**
- Used in vector search and SQL execution
- Uses `psycopg2` or SQLAlchemy


#### 🧠 `app/services/`

This is where it's define wrappers for external services.

| File | Purpose |
|------|---------|
| `openai.py` | Connects to OpenAI (text + embedding models) |
| `supabase.py` | access to Supabase DB directly to load and update tables|
| `huggingface.py` | For retrieving data and models |


## 📡 How It All Comes Together

Here’s how **a full API call flows** from end to end:

1. **User sends POST request** to the API (`/ask`) with a travel question.
2. **FastAPI** in `main.py` routes the request to the handler in `sql_flow.py`.
3. `sql_flow.py`:
   - Invokes `run_sql_flow()` from `retrieval_flow_v3.py`.
   - This function executes the **LangGraph logic**:
     - Classifies the question.
     - Routes it through SQL or vector search.
     - Executes the query.
     - Uses LLM to generate a clean answer.
4. The final answer is returned via the FastAPI route to the user.
5. Logs are saved in `retrieval_flow.log`.

## 🔍 Why This Structure Works Well

| Layer | Benefit |
|-------|---------|
| `core/` | Contains **all business logic** in one place (LangGraph, embeddings, SQL, vector search) |
| `api/` | Keeps **routes clean** and separated from logic |
| `services/` | Handles **external integrations** (OpenAI, Supabase) in a modular way |
| `db/` | Database logic is centralized — easier to update connection settings |
| `scripts/` | Easy experimentation or data prep — not mixed with production code |
| `chroma_db/` | Vector store is embedded, no external dependencies |


## 🧠 In-depth Diagram Walkthrough for Retrieval Flow

### 🟢 `__start__` → `analyze_question`

- **Purpose**: The system begins by analyzing the **user's question** to determine whether it's better handled via **SQL-based retrieval** or **semantic vector search**.
- **Code Match**: Function `analyze_question(state: State) -> dict`
  - Uses an OpenAI model with a prompt (`analysis_prompt`) that classifies the question as `"sql"` or `"general"`.
  - Updates `query_type` in the state.


### 🔀 Conditional Routing from `analyze_question`

- **Routing Function**: `route_execution(state: State)`
  - If `query_type == "sql"` → routes to `write_query`.
  - Else → routes to `execute_vector_query`.
- **LangGraph**: `graph_builder.add_conditional_edges(...)`


### 🧾 `write_query`

- **Purpose**: Generates a SQL query based on the user question and contextual information from the table.
- **Code Match**: `write_query(state: State)`
  - Builds a prompt using both the question and **relevant context** retrieved from the vector store (via `get_relevant_info`).
  - Uses a structured output (`QueryOutput`) to ensure the result is valid SQL.
  - 💡 **Note**: Even though this is a SQL-based flow, it can also perform **semantic lookups in Supabase** via vector embeddings stored in the table, enabling semantically-aware SQL construction.


### ▶️ `execute_query`

- **Purpose**: Executes the generated SQL query against the Supabase (PostgreSQL) database.
- **Code Match**: 
  - Manages queries with vector parameters (`::vector`).
  - If needed, computes query embeddings via `generate_embedding(query)`.
  - Connects to the Supabase DB to retrieve matching rows.


### 🔄 `execute_vector_query`

- **Purpose**: Performs a **semantic vector search** over an **internal Chroma vector store**.
- **Use Case**: Best suited for **general questions** about the table itself, such as:
  - What kind of data does the table store?
  - What do the columns mean?
  - What's the general topic of the dataset?
- **Code Match**: `execute_vector_query(state: State)`
  - Uses `semantic_search()` to generate the embedding and fetch the most semantically similar entries.


### ✍️ `generate_answer`

- **Purpose**: Merges the result (from either SQL or vector search) with the original question and produces a natural language response.
- **Code Match**: `generate_answer(state: State)`
  - If `query_type == "sql"`, includes both the SQL and its results.
  - If `general`, uses the vector-based content directly to generate the answer.


### 🏁 `__end__` (implicit in LangGraph)

- The final natural language answer is returned at the end of the graph execution.


## 🧱 Summary of Architecture

| Stage                | Component                  | Description                                                                 |
|---------------------|----------------------------|-----------------------------------------------------------------------------|
| Start               | `run_sql_flow`             | Accepts a user question and starts the pipeline.                           |
| Classification      | `analyze_question`         | Determines if the question is SQL-based or semantic.                       |
| SQL Route           | `write_query → execute_query → generate_answer` | For factual queries that retrieve data from Supabase.      |
| Vector Route        | `execute_vector_query → generate_answer`         | For general questions about the table's purpose or structure.              |
| Embeddings          | `generate_embedding`, `get_relevant_info` | Used for both SQL and vector-based semantic lookups.                      |
| Logging             | Everywhere (`logging`)     | Logs every step into `retrieval_flow.log`.                                 |
| Output              | Final natural language answer | Delivered as output from the LangChain graph execution.                    |



## 🔧 Implementation & Future Improvements

This retrieval system is designed to be **modular, extensible, and ready for production deployment**. It can be easily integrated into real-world applications, APIs, or internal tools. 

### ⚙️ How It Can Be Deployed

- **API as a Service**  
  The entire system runs behind a FastAPI + Uvicorn stack, making it simple to deploy as a cloud-native API. It can be containerized with **Docker** and orchestrated via **Kubernetes** or **serverless platforms** as **AWS Lambda** or **Azure Functions**.

- **Database Integration**  
  The assistant can connect to any **PostgreSQL-compatible** database or other **SQL** or **NoSQL** database, enabling seamless integration with existing business data or operational pipelines.

- **Embeddings Pipeline**  
  A background process or scheduled job can continuously embed new data into the Chroma vector store to ensure up-to-date semantic search capabilities.

- **CI/CD Ready**  
  The modular structure allows for rapid iterations and safe deployment using **CI/CD pipelines** (e.g., GitHub Actions, GitLab CI, etc.).

### 🚀 Future Improvements & Integrations

- **Advanced Query Memory**  
  Implement a memory layer (e.g., LangChain Memory or Redis) to maintain conversation context across multiple questions for a more interactive assistant experience.

- **User Personalization**  
  Extend the system to personalize answers based on user preferences, past trips, or profile data.

- **Multimodal Retrieval**  
  Integrate image, audio, or location-based embeddings to support richer question answering (e.g., "Show me pictures of my past trips").

- **Authentication & Permissions**  
  Add secure authentication (e.g., OAuth, Supabase Auth) and role-based access control for user-specific data access.

- **Multi-language Support**  
  Add multilingual understanding and response generation to serve global users in their native language.

- **Realtime Vector Sync**  
  Use event-driven workflows (e.g., Supabase Webhooks, n8n, Prefect) to update vector embeddings in real time when the underlying data changes.

- **Analytics & Monitoring**  
  Integrate with monitoring tools (e.g., Prometheus, Grafana) to track usage, query types, and system health in production.

