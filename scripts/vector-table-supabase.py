import os
import pandas as pd
from supabase import create_client, Client
import dotenv
import pickle
import tiktoken
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
tqdm.pandas()  # Enable progress_apply
# Load environment variables
dotenv.load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Load pickle file
df = pickle.load(open(r"data/TravelPlanner.pkl", "rb"))
# Add ID column right after loading the DataFrame
df = df.reset_index(drop=True)
df['id'] = df.index + 1  # Creates sequential IDs starting from 1
# Combine 'query' and 'reference_information' into 'content'
df['content'] = df['query'] + '\n' + df['reference_information']

# create openai client with api key from env variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize tokenizer for text-embedding-3-small
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_truncated_embedding(text: str, max_tokens: int = 8190) -> list:
    """Generate embeddings with automatic token truncation"""
    tokens = tokenizer.encode(text)[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    
    response = client.embeddings.create(
        input=[truncated_text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Generate embeddings with truncation
print("Generating embeddings...")
df['embedding'] = df['content'].progress_apply(get_truncated_embedding)


# Prepare data for upload
vector_data = df[['id', 'content', 'embedding']].copy()

# Batch upload function
def upload_vectors(data: pd.DataFrame, batch_size: int = 100) -> None:
    """Upload vectors to Supabase with JSON array formatting"""
    records = data.to_dict(orient='records')
    
    # Convert embeddings to JSON-serializable arrays
    for record in records:
        record['embedding'] = list(record['embedding'])  # Ensure proper array format
    
    with tqdm(total=len(records), desc="Uploading vectors") as pbar:
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                supabase.table('vector_embeddings').insert(batch).execute()
                pbar.update(len(batch))
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                raise
            
# Execute the upload
upload_vectors(vector_data)

