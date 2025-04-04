import os
import json
import pandas as pd
from supabase import create_client, Client
import dotenv
import pickle
import ast

# Load environment variables from .env file
dotenv.load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# load pickle file
import pickle
df = pickle.load(open(r"data/TravelPlanner.pkl", "rb"))

from collections import Counter

# Inicializamos un contador para las keys
key_counter = Counter()

# Iteramos por cada fila del DataFrame
for index, row in df.iterrows():
    ref_info = row['reference_information_parsed']
    # Verificamos que ref_info sea una lista (de diccionarios)
    if isinstance(ref_info, list):
        for item in ref_info:
            if isinstance(item, dict):
                key_counter.update(item.keys())

# Imprimir las keys más comunes
print("Keys más comunes en la columna JSON:")
for key, count in key_counter.most_common():
    print(f"{key}: {count}")



def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in Supabase"""
    try:
        response = supabase.table(table_name).select("*").limit(1).execute()
        return True
    except Exception as e:
        print(f"Error checking if table exists: {e}")
        return False

check_table_exists('travel_plans')

# def create_travel_plans_table():
#     """Create the travel_plans table with appropriate schema"""
#     # This requires RLS policies that allow the service role to create tables
#     sql = """
#     CREATE TABLE IF NOT EXISTS travel_plans (
#         id SERIAL PRIMARY KEY,
#         org TEXT,
#         dest TEXT,
#         days INTEGER,
#         date TEXT,
#         query TEXT NOT NULL,
#         level TEXT,
#         reference_information TEXT,
#         reference_information_parsed JSONB
#     );

#     -- Create indexes for common query fields
#     CREATE INDEX IF NOT EXISTS idx_org_dest ON travel_plans (org, dest);
#     CREATE INDEX IF NOT EXISTS idx_days ON travel_plans (days);
    
#     -- Create indexes on JSON fields
#     CREATE INDEX IF NOT EXISTS idx_total_days ON travel_plans ((reference_information_parsed->>'total_days'));
#     CREATE INDEX IF NOT EXISTS idx_airports ON travel_plans ((reference_information_parsed->>'departure_airport'), (reference_information_parsed->>'arrival_airport'));
#     """
    
#     try:
#         # Note: This requires appropriate permissions and may not work with all Supabase setups
#         # You might need to execute this SQL manually in the Supabase SQL Editor
#         response = supabase.rpc('exec_sql', {'query': sql}).execute()
#         print("Table 'travel_plans' created or verified successfully!")
#         return True
#     except Exception as e:
#         print(f"Error creating table: {e}")
#         print("You may need to create this table manually using the Supabase SQL Editor.")
#         return False
    
# create_travel_plans_table()


def prepare_data_for_upload(df):
    """Prepare data for upload, converting reference_information to JSONB format"""
    upload_df = df.copy()
    
    # Check if reference_information_parsed already exists with valid data
    if 'reference_information_parsed' in upload_df.columns and not upload_df['reference_information_parsed'].isna().all():
        print("Using existing reference_information_parsed column")
        return upload_df
    
    # If not already processed, process reference_information
    if 'reference_information' in upload_df.columns:
        sample = upload_df['reference_information'].iloc[0]
        
        if isinstance(sample, str):
            try:
                # Try json.loads first for strict JSON
                upload_df['reference_information_parsed'] = upload_df['reference_information'].apply(json.loads)
                print("Successfully parsed reference_information as JSON")
            except Exception as e:
                print(f"Error with json.loads: {e}")
                try:
                    # Fall back to ast.literal_eval for Python literals
                    upload_df['reference_information_parsed'] = upload_df['reference_information'].apply(ast.literal_eval)
                    print("Successfully parsed reference_information using ast.literal_eval")
                except Exception as e:
                    print(f"Error with ast.literal_eval: {e}")
                    upload_df['reference_information_parsed'] = upload_df['reference_information']
        elif isinstance(sample, (dict, list)):
            # Already a Python object, no parsing needed
            upload_df['reference_information_parsed'] = upload_df['reference_information']
            print("reference_information is already in dictionary/list format")
        else:
            print(f"Unexpected data type for reference_information: {type(sample)}")
            try:
                # Convert to string and try parsing
                upload_df['reference_information_parsed'] = upload_df['reference_information'].astype(str).apply(
                    lambda x: ast.literal_eval(x) if x.strip() else None
                )
                print("Successfully converted and parsed reference_information")
            except Exception as e:
                print(f"Error converting and parsing: {e}")
                upload_df['reference_information_parsed'] = upload_df['reference_information']
    
    return upload_df

prepare_data_for_upload(df)

def upload_data_to_supabase(df, table_name, batch_size=100):
    """Upload data to Supabase in batches"""
    records = df.to_dict(orient='records')
    
    total_records = len(records)
    total_batches = (total_records + batch_size - 1) // batch_size
    
    for i in range(0, total_records, batch_size):
        batch = records[i:i+batch_size]
        batch_number = i // batch_size + 1
        
        try:
            response = supabase.table(table_name).insert(batch).execute()
            print(f"Uploaded batch {batch_number}/{total_batches} ({len(batch)} records)")
        except Exception as e:
            print(f"Error uploading batch {batch_number}: {e}")
            return False
    
    print(f"Successfully uploaded {total_records} records to {table_name}!")
    return True

upload_data_to_supabase(df, 'travel_plans')

def query_by_org_dest(org, dest):
    """Query trips by origin and destination cities"""
    try:
        response = supabase.table('travel_plans').select('*').eq('org', org).eq('dest', dest).execute()
        results = response.data
        print(f"Found {len(results)} trips from {org} to {dest}")
        return results
    except Exception as e:
        print(f"Error querying by org/dest: {e}")
        return []
    

query_by_org_dest('Sarasota', 'Chicago')



def query_by_description(search_term):
    """Search trips by description with semantic search capabilities using REST API.
    The function fetches all records from 'travel_plans', filters them based on the search term,
    saves the filtered results to 'results_fallback.txt', and prints a summary status.
    """
    try:
        # Fetch all records using the REST API.
        all_results = supabase.table('travel_plans').select('*').execute().data
        
        # Filter the records based on the 'Description' field in reference_information_parsed.
        filtered_results = []
        for record in all_results:
            ref_info = record.get('reference_information_parsed')
            if isinstance(ref_info, list):
                for item in ref_info:
                    if isinstance(item, dict) and 'Description' in item and search_term.lower() in item['Description'].lower():
                        filtered_results.append({
                            'record': record,
                            'matched_description': item['Description'],
                            'content': item.get('Content', 'No content available')
                        })
                        break
        
        # Save the filtered results to a file.
        with open("results_fallback.txt", "w") as f:
            json.dump(filtered_results, f, indent=2)
        
        # Print a brief status message.
        print(f"Process complete: Found {len(filtered_results)} trips containing '{search_term}' using REST API fallback.")
        # return filtered_results # return the general status of the process instead of the filtered results 
        
    except Exception as e2:
        print(f"Error with fallback method: {e2}")
        return []




# Example usage:
query_by_description('Attractions in Chicago')


