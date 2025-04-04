from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

load_dotenv()

USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = os.getenv("DB_NAME")

# URL-encode the credentials
encoded_user = quote_plus(USER)
encoded_password = quote_plus(PASSWORD)

def get_database():
    postgres_uri = f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{HOST}:{PORT}/{DBNAME}"
    return SQLDatabase.from_uri(postgres_uri)

# # test postresql connection
# def test_postgresql_connection():
#     try:
#         db = get_database()
#         print("Connection successful!")
#         return db
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
# test_postgresql_connection()