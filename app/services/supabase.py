"""
Supabase service module.

This module contains functions to interact with Supabase,
including uploading data and querying the database.
"""

import json
from app.config import supabase

def upload_data_to_supabase(df, table_name, batch_size=100):
    """
    Upload data to Supabase in batches.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to upload.
        table_name (str): Name of the Supabase table to upload data to.
        batch_size (int): Number of records per batch.

    Returns:
        True if all batches were uploaded successfully, False otherwise.
    """
    records = df.to_dict(orient='records')
    total_records = len(records)
    total_batches = (total_records + batch_size - 1) // batch_size

    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        batch_number = i // batch_size + 1

        try:
            response = supabase.table(table_name).insert(batch).execute()
            print(f"Uploaded batch {batch_number}/{total_batches} ({len(batch)} records)")
        except Exception as e:
            print(f"Error uploading batch {batch_number}: {e}")
            return False

    print(f"Successfully uploaded {total_records} records to {table_name}!")
    return True

def query_by_org_dest(org, dest):
    """
    Query trips by origin and destination cities.

    Args:
        org (str): Origin city.
        dest (str): Destination city.

    Returns:
        A list of records matching the origin and destination.
    """
    try:
        response = supabase.table('travel_plans').select('*').eq('org', org).eq('dest', dest).execute()
        results = response.data
        print(f"Found {len(results)} trips from {org} to {dest}")
        return results
    except Exception as e:
        print(f"Error querying by origin/destination: {e}")
        return []

def query_by_description(search_term):
    """
    Search trips by description using semantic search capabilities via REST API.

    The function fetches all records from 'travel_plans', filters them based on the search term
    found within the 'Description' field inside 'reference_information_parsed', saves the filtered
    results to 'results_fallback.txt', and prints a summary status.

    Args:
        search_term (str): The term to search for within the trip descriptions.

    Returns:
        A list of records matching the search criteria.
    """
    try:
        # Fetch all records from Supabase
        all_results = supabase.table('travel_plans').select('*').execute().data

        # Filter records based on the 'Description' field in 'reference_information_parsed'
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

        # Save filtered results to a file
        with open("results_fallback.txt", "w") as f:
            json.dump(filtered_results, f, indent=2)

        print(f"Process complete: Found {len(filtered_results)} trips containing '{search_term}' using REST API fallback.")
        return filtered_results
    except Exception as e:
        print(f"Error with query_by_description: {e}")
        return []
