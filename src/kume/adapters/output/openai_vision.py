import base64
from typing import Any

from openai import AsyncOpenAI

from kume.ports.output.vision import VisionPort


class OpenAIVisionAdapter(VisionPort):
    """Vision adapter using OpenAI's chat completions API with image support."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=3)
        self._model = model

    async def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        )
        if response.choices:
            return response.choices[0].message.content or ""
        return ""

    async def analyze_image_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: dict[str, Any],
    ) -> str:
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "nutrition_analysis",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )
        if not response.choices:
            return "{}"
        message = response.choices[0].message
        if hasattr(message, "refusal") and message.refusal:
            raise ValueError(f"Vision model refused: {message.refusal}")
        return message.content or "{}"
