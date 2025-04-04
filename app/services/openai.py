import os
from langchain_openai import ChatOpenAI

def get_openai_model(model_name: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = None, **kwargs):
    """
    Returns an instance of the OpenAI model with the specified configuration.
    For models that do not support temperature (e.g., function-calling or tools-only models),
    the parameter is automatically excluded.
    """
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Ensure the API key is loaded from the environment

    # List of models that do NOT support temperature
    models_without_temperature = {"o3-mini"}

    model_kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        **kwargs
    }

    if model_name not in models_without_temperature:
        model_kwargs["temperature"] = temperature

    return ChatOpenAI(**model_kwargs)
