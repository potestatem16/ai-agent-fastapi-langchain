import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Initialize the persistent client
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Connect to the existing vector store
vector_store = Chroma(
    client=persistent_client,
    collection_name="travelplanner_sections",
    embedding_function=embeddings
)

query_text = "search methods"
query_text = "travel plans table structure"

# Perform the similarity search
results = vector_store.similarity_search(query_text, k=1)

# Print the results
print(f"\nSearch results for query: '{query_text}'\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Section: {doc.metadata.get('section', 'N/A')}")
    print("Content:")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    print("-" * 80)
    


