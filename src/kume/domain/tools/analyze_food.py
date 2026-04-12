from collections.abc import Callable


def analyze_food(description: str, llm_call: Callable[[str], str], context: str = "") -> str:
    prompt = f"You are a nutrition expert.\n\n{context}\n\nAnalyze this food: {description}\n\nProvide nutritional assessment and whether it aligns with common health goals."
    return llm_call(prompt)
