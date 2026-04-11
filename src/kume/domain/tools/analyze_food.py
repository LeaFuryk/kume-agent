from collections.abc import Callable


def analyze_food(description: str, llm_call: Callable[[str], str]) -> str:
    prompt = f"You are a nutrition expert. Analyze this food: {description}\n\nProvide nutritional assessment and whether it aligns with common health goals."
    return llm_call(prompt)
