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

    def test_single_tool_conversion_parametersJsonSchema(self):
        """Single OpenAI tool converts using parametersJsonSchema (not parameters)."""
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
        # #1: Must use parametersJsonSchema, NOT parameters
        assert "parametersJsonSchema" in decls[0]
        assert "parameters" not in decls[0]
        assert decls[0]["parametersJsonSchema"]["properties"]["location"]["type"] == "string"

    def test_legacy_parameters_for_claude(self):
        """Claude models use legacy parameters field."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "Test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = _convert_tools_to_gemini(openai_tools, use_parameters=True)

        decls = result[0]["functionDeclarations"]
        assert "parameters" in decls[0]
        assert "parametersJsonSchema" not in decls[0]

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
        assert "parametersJsonSchema" not in decls[0]

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
        """Tool result message formats as functionResponse with {output: ...}."""
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
        # #9: Must use {output: value} format, NOT raw JSON
        assert parts[0]["functionResponse"]["response"] == {"output": "Sunny, 25°C"}

    def test_tool_message_error_response(self, provider):
        """Tool error uses {error: value} format."""
        messages = [
            ChatMessage(
                role="tool",
                content="Connection timeout",
                metadata={"tool_name": "get_weather", "is_error": True},
            )
        ]

        _, contents = provider._format_messages(messages)

        response = contents[0]["parts"][0]["functionResponse"]["response"]
        assert response == {"error": "Connection timeout"}

    def test_tool_message_json_content(self, provider):
        """Tool result with JSON content uses output wrapper."""
        messages = [
            ChatMessage(
                role="tool",
                content='{"temp": 25, "condition": "sunny"}',
                metadata={"tool_name": "get_weather"},
            )
        ]

        _, contents = provider._format_messages(messages)

        response_data = contents[0]["parts"][0]["functionResponse"]["response"]
        # #9: JSON string wrapped in {output: ...}
        assert response_data == {"output": '{"temp": 25, "condition": "sunny"}'}

    def test_empty_text_blocks_skipped(self, provider):
        """Empty text blocks in assistant messages are skipped."""
        messages = [
            ChatMessage(role="assistant", content="", metadata=None),
            ChatMessage(role="assistant", content="  ", metadata=None),
            ChatMessage(role="user", content="hello", metadata=None),
        ]

        _, contents = provider._format_messages(messages)

        # Empty assistant messages should be skipped
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

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


class TestStreamingParsing:
    """Tests for CCA streaming response parsing logic."""

    def test_text_response(self):
        """Normal text response parses correctly from streaming chunk."""
        chunk = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello, how can I help?"}]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        }

        response_data = chunk.get("response", chunk)
        text_parts = []
        tool_calls = []
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part and part["text"].strip():
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    tool_calls.append(part["functionCall"])

        assert "".join(text_parts) == "Hello, how can I help?"
        assert tool_calls == []

    def test_function_call_response(self):
        """Function call response parses correctly from streaming chunk."""
        chunk = {
            "response": {
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
            }
        }

        response_data = chunk.get("response", chunk)
        tool_calls = []
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "functionCall" in part:
                    tool_calls.append(part["functionCall"])

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"location": "Tokyo"}

    def test_thinking_response(self):
        """Thinking parts are separated from text parts."""
        chunk = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Let me think...", "thought": True},
                                {"text": "Here is the answer."},
                            ]
                        }
                    }
                ],
            }
        }

        response_data = chunk.get("response", chunk)
        text_parts = []
        thinking_parts = []
        for candidate in response_data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if part.get("thought") and "text" in part:
                    thinking_parts.append(part["text"])
                elif "text" in part and part["text"].strip():
                    text_parts.append(part["text"])

        assert "".join(text_parts) == "Here is the answer."
        assert "".join(thinking_parts) == "Let me think..."
