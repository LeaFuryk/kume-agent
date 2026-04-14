import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kume.adapters.output.openai_vision import OpenAIVisionAdapter
from kume.ports.output.vision import VisionPort


class TestOpenAIVisionAdapter:
    def test_implements_vision_port(self) -> None:
        assert issubclass(OpenAIVisionAdapter, VisionPort)

    @pytest.mark.asyncio
    async def test_analyze_image_calls_api(self) -> None:
        mock_message = MagicMock()
        mock_message.content = "This is a plate of rice and beans."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            result = await adapter.analyze_image(
                system_prompt="You are a nutrition analyst.",
                user_prompt="Analyze this meal.",
                image_bytes=b"fake-image",
                mime_type="image/jpeg",
            )

        assert result == "This is a plate of rice and beans."
        mock_client.chat.completions.create.assert_awaited_once()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]

        # System message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a nutrition analyst."

        # User message with text + image_url blocks
        assert messages[1]["role"] == "user"
        content_blocks = messages[1]["content"]
        assert len(content_blocks) == 2
        assert content_blocks[0]["type"] == "text"
        assert content_blocks[0]["text"] == "Analyze this meal."
        assert content_blocks[1]["type"] == "image_url"
        assert "image_url" in content_blocks[1]

    @pytest.mark.asyncio
    async def test_base64_encoding(self) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nfake-png-data"
        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")

        mock_message = MagicMock()
        mock_message.content = "analysis"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            await adapter.analyze_image(
                system_prompt="system",
                user_prompt="user",
                image_bytes=image_bytes,
                mime_type="image/png",
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        data_uri = call_kwargs["messages"][1]["content"][1]["image_url"]["url"]
        assert expected_b64 in data_uri

    @pytest.mark.asyncio
    async def test_mime_type_in_data_uri(self) -> None:
        mock_message = MagicMock()
        mock_message.content = "result"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            await adapter.analyze_image(
                system_prompt="system",
                user_prompt="user",
                image_bytes=b"img",
                mime_type="image/webp",
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        data_uri = call_kwargs["messages"][1]["content"][1]["image_url"]["url"]
        assert data_uri.startswith("data:image/webp;base64,")

    @pytest.mark.asyncio
    async def test_returns_response_content(self) -> None:
        mock_message = MagicMock()
        mock_message.content = "Estimated 450 kcal, high in protein."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            result = await adapter.analyze_image(
                system_prompt="system",
                user_prompt="analyze",
                image_bytes=b"data",
                mime_type="image/jpeg",
            )

        assert result == "Estimated 450 kcal, high in protein."

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_content_is_none(self) -> None:
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            result = await adapter.analyze_image(
                system_prompt="system",
                user_prompt="analyze",
                image_bytes=b"data",
                mime_type="image/jpeg",
            )

        assert result == ""

    @pytest.mark.asyncio
    async def test_custom_model(self) -> None:
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key", model="gpt-4o-mini")
            await adapter.analyze_image(
                system_prompt="system",
                user_prompt="user",
                image_bytes=b"img",
                mime_type="image/jpeg",
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"


class TestAnalyzeImageJson:
    """Tests for the structured JSON output method."""

    SAMPLE_SCHEMA = {
        "type": "object",
        "properties": {
            "calories": {"type": "number"},
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["calories", "items"],
        "additionalProperties": False,
    }

    @pytest.mark.asyncio
    async def test_analyze_image_json_passes_response_format(self) -> None:
        mock_message = MagicMock()
        mock_message.content = '{"calories": 450, "items": ["rice", "beans"]}'
        mock_message.refusal = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            await adapter.analyze_image_json(
                system_prompt="You are a nutrition analyst.",
                user_prompt="Analyze this meal.",
                image_bytes=b"fake-image",
                mime_type="image/jpeg",
                json_schema=self.SAMPLE_SCHEMA,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "nutrition_analysis"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] is self.SAMPLE_SCHEMA

    @pytest.mark.asyncio
    async def test_analyze_image_json_returns_content(self) -> None:
        expected = '{"calories": 450, "items": ["rice", "beans"]}'
        mock_message = MagicMock()
        mock_message.content = expected
        mock_message.refusal = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            result = await adapter.analyze_image_json(
                system_prompt="system",
                user_prompt="analyze",
                image_bytes=b"data",
                mime_type="image/jpeg",
                json_schema=self.SAMPLE_SCHEMA,
            )

        assert result == expected

    @pytest.mark.asyncio
    async def test_analyze_image_json_refusal_raises(self) -> None:
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.refusal = "I cannot analyze this image."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            with pytest.raises(ValueError, match="Vision model refused"):
                await adapter.analyze_image_json(
                    system_prompt="system",
                    user_prompt="analyze",
                    image_bytes=b"data",
                    mime_type="image/jpeg",
                    json_schema=self.SAMPLE_SCHEMA,
                )

    @pytest.mark.asyncio
    async def test_analyze_image_json_empty_choices_returns_empty_json(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = []

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            result = await adapter.analyze_image_json(
                system_prompt="system",
                user_prompt="analyze",
                image_bytes=b"data",
                mime_type="image/jpeg",
                json_schema=self.SAMPLE_SCHEMA,
            )

        assert result == "{}"

    @pytest.mark.asyncio
    async def test_analyze_image_still_works(self) -> None:
        """Backward compatibility: the old analyze_image method still works."""
        mock_message = MagicMock()
        mock_message.content = "Estimated 450 kcal, high in protein."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("kume.adapters.output.openai_vision.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIVisionAdapter(api_key="test-key")
            result = await adapter.analyze_image(
                system_prompt="system",
                user_prompt="analyze",
                image_bytes=b"data",
                mime_type="image/jpeg",
            )

        assert result == "Estimated 450 kcal, high in protein."
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" not in call_kwargs
