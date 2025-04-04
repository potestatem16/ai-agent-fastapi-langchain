import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt

# load environment variables from .env file
load_dotenv()

USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = os.getenv("DB_NAME")

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


# Initialize tokenizer for text-embedding-3-small
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embedding(text: str, max_tokens: int = 8190) -> list:
    """Generate embeddings with automatic token truncation"""
    tokens = tokenizer.encode(text)[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    
    response = client.embeddings.create(
        input=[truncated_text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


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


# Generate embedding for search query
search_query = "attractions in chicago"
query_embedding = get_embedding(search_query)

cursor = connection.cursor()




try:
    # Convert embedding to proper format for vector casting
    vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    # # Perform vector similarity search using cosine distance
    # search_query_sql = """
    # SELECT id, content, 1 - (embedding <#> %s::vector) as similarity
    # FROM vector_embeddings
    # ORDER BY embedding <#> %s::vector
    # LIMIT 5;
    # """
    
    # search_query_sql = """
    # SELECT id, content, 1 - (embedding <#> %s::vector) AS similarity 
    # FROM vector_embeddings
    # ORDER BY embedding <#> %s::vector
    # LIMIT 5;
    # """
    
    # search_query_sql = """
    # SELECT id, content, 1 - (embedding <#> %s::vector) AS similarity
    # FROM vector_embeddings
    # WHERE id=1
    # ORDER BY embedding <#> %s::vector
    # LIMIT 5;
    # """
    
    # search_query_sql = """WITH relevant_ids AS (
    #     SELECT id
    #     FROM travel_plans
    #     WHERE org = 'Chicago' OR dest = 'Chicago'
    #     )
    #     SELECT ve.id, ve.content, 1 - (ve.embedding <#> %s::vector) AS similarity
    #     FROM vector_embeddings ve
    #     WHERE ve.id IN (SELECT id FROM relevant_ids)
    #     ORDER BY ve.embedding <#> %s::vector
    #     LIMIT 5;
    # """
    
    search_query_sql = """
    WITH relevant_ids AS (
        SELECT id
        FROM travel_plans
        WHERE dest ILIKE '%%Chicago%%'
        OR org ILIKE '%%Chicago%%' OR query ILIKE '%%Chicago%%'
        OR reference_information ILIKE '%%Chicago%%'
        )
    SELECT ve.id, ve.content, 1 - (ve.embedding <#> %s::vector) AS similarity
    FROM vector_embeddings ve
    WHERE ve.id IN (SELECT id FROM relevant_ids)
    ORDER BY ve.embedding <#> %s::vector
    LIMIT 5;
    """
    
    
    
    cursor.execute(search_query_sql, (vector_str, vector_str))
    results = cursor.fetchall()
    
    # Save results to a text file
    output_file = "vector_search_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Search results for '{search_query}':\n\n")
        for id, content, similarity in results:
            f.write(f"--- Result ID: {id} (Similarity: {similarity:.4f}) ---\n\n")
            f.write(f"Content: {content}\n\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Search completed successfully. Found {len(results)} results.")
    print(f"Results saved to '{output_file}'")
    
except Exception as e:
    print(f"Error during search execution: {str(e)}")
    print(f"Full error details: {repr(e)}")
    
finally:
    cursor.close()
    connection.close()
    print("Database connection closed.")


