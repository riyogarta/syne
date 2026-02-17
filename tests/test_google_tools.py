"""Tests for Google Gemini tool calling implementation."""

import json
import pytest
from syne.llm.google import _convert_tools_to_gemini, GoogleProvider
from syne.llm.provider import ChatMessage


class TestConvertToolsToGemini:
    """Tests for _convert_tools_to_gemini helper function."""

    def test_empty_tools(self):
        """Empty list returns empty list."""
        assert _convert_tools_to_gemini([]) == []
        assert _convert_tools_to_gemini(None) == []

    def test_single_tool_conversion(self):
        """Single OpenAI tool converts to Gemini format."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = _convert_tools_to_gemini(openai_tools)

        assert len(result) == 1
        assert "functionDeclarations" in result[0]
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["description"] == "Get the current weather"
        assert decls[0]["parameters"]["properties"]["location"]["type"] == "string"

    def test_multiple_tools_conversion(self):
        """Multiple OpenAI tools convert to single Gemini tools array."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "First tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "description": "Second tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = _convert_tools_to_gemini(openai_tools)

        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 2
        assert decls[0]["name"] == "tool_a"
        assert decls[1]["name"] == "tool_b"

    def test_tool_without_parameters(self):
        """Tool without parameters still converts."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_params",
                    "description": "A tool without parameters",
                },
            }
        ]

        result = _convert_tools_to_gemini(openai_tools)

        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert decls[0]["name"] == "no_params"
        assert "parameters" not in decls[0]

    def test_invalid_tool_format_skipped(self):
        """Invalid tool formats are skipped."""
        openai_tools = [
            {"type": "invalid"},  # Wrong type
            {"function": {"name": "missing_type"}},  # Missing type
            {
                "type": "function",
                "function": {"name": "valid", "description": "Valid tool"},
            },
        ]

        result = _convert_tools_to_gemini(openai_tools)

        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "valid"


class TestFormatMessages:
    """Tests for GoogleProvider._format_messages with tool support."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mock credentials."""
        # Use API key mode to avoid credential setup
        return GoogleProvider(api_key="test-key")

    def test_basic_messages(self, provider):
        """Basic user/assistant messages format correctly."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]

        system_text, contents = provider._format_messages(messages)

        assert system_text == "You are helpful."
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello"
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Hi there!"

    def test_multiple_system_messages_concatenated(self, provider):
        """Multiple system messages are concatenated."""
        messages = [
            ChatMessage(role="system", content="Part 1"),
            ChatMessage(role="system", content="Part 2"),
            ChatMessage(role="user", content="Hello"),
        ]

        system_text, contents = provider._format_messages(messages)

        assert system_text == "Part 1\n\nPart 2"

    def test_assistant_with_tool_calls(self, provider):
        """Assistant message with tool_calls formats as functionCall."""
        tool_calls = [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo"}',
                }
            }
        ]
        messages = [
            ChatMessage(
                role="assistant",
                content="Let me check the weather.",
                metadata={"tool_calls": tool_calls},
            )
        ]

        _, contents = provider._format_messages(messages)

        assert len(contents) == 1
        parts = contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "Let me check the weather."
        assert "functionCall" in parts[1]
        assert parts[1]["functionCall"]["name"] == "get_weather"
        assert parts[1]["functionCall"]["args"] == {"location": "Tokyo"}

    def test_tool_message_formats_as_function_response(self, provider):
        """Tool result message formats as functionResponse."""
        messages = [
            ChatMessage(
                role="tool",
                content="Sunny, 25°C",
                metadata={"tool_name": "get_weather"},
            )
        ]

        _, contents = provider._format_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert "functionResponse" in parts[0]
        assert parts[0]["functionResponse"]["name"] == "get_weather"
        assert parts[0]["functionResponse"]["response"] == {"result": "Sunny, 25°C"}

    def test_tool_message_json_content(self, provider):
        """Tool result with JSON content parses correctly."""
        messages = [
            ChatMessage(
                role="tool",
                content='{"temp": 25, "condition": "sunny"}',
                metadata={"tool_name": "get_weather"},
            )
        ]

        _, contents = provider._format_messages(messages)

        response_data = contents[0]["parts"][0]["functionResponse"]["response"]
        assert response_data == {"temp": 25, "condition": "sunny"}

    def test_full_tool_call_flow(self, provider):
        """Full tool call flow: user -> assistant (tool call) -> tool result."""
        messages = [
            ChatMessage(role="user", content="What's the weather in Tokyo?"),
            ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            }
                        }
                    ]
                },
            ),
            ChatMessage(
                role="tool",
                content="Sunny, 25°C",
                metadata={"tool_name": "get_weather"},
            ),
        ]

        _, contents = provider._format_messages(messages)

        assert len(contents) == 3
        # User message
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "What's the weather in Tokyo?"
        # Assistant with tool call
        assert contents[1]["role"] == "model"
        assert "functionCall" in contents[1]["parts"][0]
        # Tool result
        assert contents[2]["role"] == "user"
        assert "functionResponse" in contents[2]["parts"][0]


class TestParseCCAResponse:
    """Tests for parsing CCA response with tool calls."""

    @pytest.fixture
    def provider(self):
        return GoogleProvider(api_key="test-key")

    def test_text_response(self, provider):
        """Normal text response parses correctly."""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello, how can I help?"}]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        response = provider._parse_cca_response(data, "gemini-2.5-pro")

        assert response.content == "Hello, how can I help?"
        assert response.tool_calls is None
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_function_call_response(self, provider):
        """Function call response parses correctly."""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Tokyo"},
                                }
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        response = provider._parse_cca_response(data, "gemini-2.5-pro")

        assert response.content == ""
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        # Arguments should be JSON string (OpenAI format)
        assert json.loads(response.tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}

    def test_mixed_text_and_function_call(self, provider):
        """Response with both text and function call."""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me check that for you."},
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "test"},
                                }
                            },
                        ]
                    }
                }
            ],
            "usageMetadata": {},
        }

        response = provider._parse_cca_response(data, "gemini-2.5-pro")

        assert response.content == "Let me check that for you."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "search"

    def test_cca_wrapped_response(self, provider):
        """CCA-wrapped response (with 'response' envelope) parses correctly."""
        data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Wrapped response"}]
                        }
                    }
                ],
                "usageMetadata": {},
            }
        }

        response = provider._parse_cca_response(data, "gemini-2.5-pro")

        assert response.content == "Wrapped response"
