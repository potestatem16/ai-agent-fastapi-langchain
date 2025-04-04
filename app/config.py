"""
Configuration module for the application.

This module loads environment variables and sets up connections
to external services, such as Supabase.
"""

import os
import dotenv
from supabase import create_client, Client

# Load environment variables from the .env file
dotenv.load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Create a Supabase client instance
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
