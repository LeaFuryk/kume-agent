from collections.abc import Callable


def ask_recommendation(query: str, llm_call: Callable[[str], str], context: str = "") -> str:
    prompt = f"You are a nutrition expert.\n\n{context}\n\nThe user asks: {query}\n\nProvide a helpful, personalized nutrition recommendation."
    return llm_call(prompt)
