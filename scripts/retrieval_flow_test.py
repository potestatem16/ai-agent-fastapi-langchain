from scripts.retrieval_flow_v1 import get_relevant_info, write_query, check_query, dummy_execute_query, generate_answer, run_sql_flow

# Create a dummy state dictionary for testing individual functions
state = {
    "question": "Restaurants in Chicago",
    "query": "",
    "result": "",
    "answer": "",
    "validation_details": ""
}

# Test get_relevant_info:
info = get_relevant_info(state["question"])
print("Relevant Info:", info)


# Test write_query (this will call get_relevant_info inside it):
query_result = write_query(state)
print("Generated Query:", query_result)


steps, final = run_sql_flow("Restaurants in Chicago")
for i, step in enumerate(steps, start=1):
    print(f"Step {i}:", step)
print("Final state:", final)

