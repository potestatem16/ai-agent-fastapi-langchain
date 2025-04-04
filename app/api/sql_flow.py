# app/api/sql_flow.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import the main function of the SQL flow from the core
from app.core.retrieval_flow_v3 import run_sql_flow

router = APIRouter()

class SQLQueryRequest(BaseModel):
    question: str = Field(..., example="Restaurants in Chicago")
    model_id: str | None = Field(None, example="o3-mini")
    temperature: float | None = Field(None, example=0.0)
    max_tokens: int | None = Field(None, example=7000)

@router.post("/sql_query_execution")
def execute_sql_query(payload: SQLQueryRequest):
    """
    Endpoint that receives a user's question and executes the flow:
      1. SQL query generation.
      2. Query validation.
      3. Execution (dummy, for now).
      4. Final response generation.
      
    Optionally, a `model_id` can be passed to change the default model.
    """
    try:
        steps, final = run_sql_flow(
            user_question=payload.question,
            model_id=payload.model_id,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens
        )
        return {
            "step_by_step": steps,
            "final_state": final
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
