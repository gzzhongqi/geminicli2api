"""Tests for function calling / tool use support."""

import json
import pytest
from src.routes.transformers.openai import (
    openai_request_to_gemini,
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
    _transform_openai_tools_to_gemini,
    _transform_tool_choice_to_gemini,
    _extract_content_and_reasoning,
)
from src.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    FunctionDefinition,
    ToolCall,
)


class TestTransformOpenAIToolsToGemini:
    """Tests for _transform_openai_tools_to_gemini."""

    def test_basic_function_tool(self):
        """Test transforming a basic function tool."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        result = _transform_openai_tools_to_gemini(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a location"
        assert result[0]["parameters"]["type"] == "object"
        assert "location" in result[0]["parameters"]["properties"]

    def test_removes_schema_and_additional_properties(self):
        """Test that $schema and additionalProperties are removed."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "parameters": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"arg": {"type": "string"}},
                    },
                },
            }
        ]
        result = _transform_openai_tools_to_gemini(tools)
        assert "$schema" not in result[0]["parameters"]
        assert "additionalProperties" not in result[0]["parameters"]
        assert result[0]["parameters"]["type"] == "object"

    def test_ignores_non_function_tools(self):
        """Test that non-function tools are ignored."""
        tools = [
            {"type": "code_interpreter"},
            {"type": "function", "function": {"name": "valid_func"}},
        ]
        result = _transform_openai_tools_to_gemini(tools)
        assert len(result) == 1
        assert result[0]["name"] == "valid_func"

    def test_tool_object_with_attributes(self):
        """Test transforming Tool Pydantic objects."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                ),
            )
        ]
        result = _transform_openai_tools_to_gemini(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"


class TestTransformToolChoiceToGemini:
    """Tests for _transform_tool_choice_to_gemini."""

    def test_auto_mode(self):
        """Test auto tool_choice."""
        result = _transform_tool_choice_to_gemini("auto")
        assert result == {"mode": "AUTO"}

    def test_none_mode(self):
        """Test none tool_choice."""
        result = _transform_tool_choice_to_gemini("none")
        assert result == {"mode": "NONE"}

    def test_required_mode(self):
        """Test required tool_choice."""
        result = _transform_tool_choice_to_gemini("required")
        assert result == {"mode": "ANY"}

    def test_specific_function(self):
        """Test specific function tool_choice."""
        result = _transform_tool_choice_to_gemini(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        assert result == {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}

    def test_none_value(self):
        """Test None tool_choice returns None."""
        result = _transform_tool_choice_to_gemini(None)
        assert result is None


class TestExtractContentAndReasoning:
    """Tests for _extract_content_and_reasoning with function calls."""

    def test_text_only(self):
        """Test extracting text-only content."""
        parts = [{"text": "Hello, world!"}]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == "Hello, world!"
        assert reasoning == ""
        assert tool_calls == []

    def test_function_call(self):
        """Test extracting function call from parts."""
        parts = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Tokyo"},
                }
            }
        ]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "location": "Tokyo"
        }
        assert tool_calls[0]["id"].startswith("call_")

    def test_text_and_function_call(self):
        """Test extracting both text and function call."""
        parts = [
            {"text": "Let me check the weather for you."},
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Paris"},
                }
            },
        ]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == "Let me check the weather for you."
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

    def test_multiple_function_calls(self):
        """Test extracting multiple function calls."""
        parts = [
            {"functionCall": {"name": "func1", "args": {"a": 1}}},
            {"functionCall": {"name": "func2", "args": {"b": 2}}},
        ]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "func1"
        assert tool_calls[1]["function"]["name"] == "func2"

    def test_thought_with_function_call(self):
        """Test extracting thought (reasoning) with function call."""
        parts = [
            {"text": "Let me think about this...", "thought": True},
            {"functionCall": {"name": "calculate", "args": {"x": 5}}},
        ]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == ""
        assert reasoning == "Let me think about this..."
        assert len(tool_calls) == 1


class TestOpenAIRequestToGemini:
    """Tests for openai_request_to_gemini with function calling."""

    def test_request_with_tools(self):
        """Test transforming request with tools."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="What's the weather?")],
            tools=[
                Tool(
                    type="function",
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={
                            "type": "object",
                            "properties": {"loc": {"type": "string"}},
                        },
                    ),
                )
            ],
        )
        result = openai_request_to_gemini(request)
        assert "tools" in result
        assert any("functionDeclarations" in t for t in result["tools"])
        func_decls = next(
            t["functionDeclarations"]
            for t in result["tools"]
            if "functionDeclarations" in t
        )
        assert func_decls[0]["name"] == "get_weather"

    def test_request_with_tool_choice(self):
        """Test transforming request with tool_choice."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Call the function")],
            tools=[
                Tool(
                    type="function",
                    function=FunctionDefinition(name="my_func"),
                )
            ],
            tool_choice="required",
        )
        result = openai_request_to_gemini(request)
        assert "toolConfig" in result
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"

    def test_assistant_message_with_tool_calls(self):
        """Test transforming assistant message with tool_calls."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[
                ChatMessage(role="user", content="Get weather"),
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            type="function",
                            function={
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        )
                    ],
                ),
                ChatMessage(
                    role="tool",
                    content='{"temperature": 72}',
                    tool_call_id="call_123",
                    name="get_weather",
                ),
            ],
        )
        result = openai_request_to_gemini(request)

        # Check assistant message with functionCall
        assert result["contents"][1]["role"] == "model"
        assert "functionCall" in result["contents"][1]["parts"][0]
        assert (
            result["contents"][1]["parts"][0]["functionCall"]["name"] == "get_weather"
        )

        # Check tool response with functionResponse
        assert result["contents"][2]["role"] == "user"
        assert "functionResponse" in result["contents"][2]["parts"][0]
        assert (
            result["contents"][2]["parts"][0]["functionResponse"]["name"]
            == "get_weather"
        )


class TestGeminiResponseToOpenAI:
    """Tests for gemini_response_to_openai with function calls."""

    def test_response_with_function_call(self):
        """Test transforming Gemini response with function call."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "London"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = gemini_response_to_openai(gemini_response, "gemini-2.5-pro")

        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] is None
        assert "tool_calls" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1

        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"location": "London"}
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_response_text_only(self):
        """Test that text-only response works correctly."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "The weather is sunny."}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = gemini_response_to_openai(gemini_response, "gemini-2.5-pro")
        assert result["choices"][0]["message"]["content"] == "The weather is sunny."
        assert "tool_calls" not in result["choices"][0]["message"]
        assert result["choices"][0]["finish_reason"] == "stop"


class TestGeminiStreamChunkToOpenAI:
    """Tests for gemini_stream_chunk_to_openai with function calls."""

    def test_stream_chunk_with_function_call(self):
        """Test transforming streaming chunk with function call."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "python"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = gemini_stream_chunk_to_openai(
            gemini_chunk, "gemini-2.5-pro", "resp_123"
        )

        assert "tool_calls" in result["choices"][0]["delta"]
        assert len(result["choices"][0]["delta"]["tool_calls"]) == 1
        assert (
            result["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "search"
        )
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_stream_chunk_text_only(self):
        """Test that text-only chunk works correctly."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    },
                }
            ]
        }
        result = gemini_stream_chunk_to_openai(
            gemini_chunk, "gemini-2.5-pro", "resp_123"
        )
        assert result["choices"][0]["delta"]["content"] == "Hello"
        assert "tool_calls" not in result["choices"][0]["delta"]
