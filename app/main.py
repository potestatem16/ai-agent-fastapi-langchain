# app/main.py

from fastapi import FastAPI
from app.api import sql_flow  # Import the module where the endpoint is located

def create_app() -> FastAPI:
    """
    Create and configure the main FastAPI instance.
    """
    app = FastAPI(title="FastAPI, LangChain, and SQL Flow API")
    
    # Mount the router for the SQL flow under the /sql prefix
    app.include_router(sql_flow.router, prefix="/sql", tags=["SQL Flow"])
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
