import json
import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Initialize the persistent client for ChromaDB
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Ensure any existing collection is deleted first
try:
    persistent_client.delete_collection("travelplanner_sections")
    print("Deleted existing collection")
except:
    print("No existing collection to delete")


# Load the JSON file with the sections
path = r'data/TravelPlanner_sections.json'
with open(path, 'r', encoding='utf-8') as f:
    sections = json.load(f)

# Create a list of Document objects from the sections
documents = [
    Document(page_content=content, metadata={'section': key})
    for key, content in sections.items()
]

print(f"Creating vector store with {len(documents)} documents")

# Create the Chroma vector store using LangChain's interface
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    client=persistent_client,
    collection_name="travelplanner_sections"
)

