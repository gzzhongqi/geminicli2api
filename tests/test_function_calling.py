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
from src.routes.transformers.responses import (
    responses_request_to_gemini,
    gemini_response_to_responses,
    gemini_stream_chunk_to_responses_events,
)
from src.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    FunctionDefinition,
    ToolCall,
)
from src.schemas.responses import ResponsesRequest


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
        result, has_web_search = _transform_openai_tools_to_gemini(tools)
        assert len(result) == 1
        assert has_web_search is False
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
        result, _ = _transform_openai_tools_to_gemini(tools)
        assert "$schema" not in result[0]["parameters"]
        assert "additionalProperties" not in result[0]["parameters"]
        assert result[0]["parameters"]["type"] == "object"

    def test_ignores_non_function_tools(self):
        """Test that non-function tools are ignored."""
        tools = [
            {"type": "code_interpreter"},
            {"type": "function", "function": {"name": "valid_func"}},
        ]
        result, _ = _transform_openai_tools_to_gemini(tools)
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
        result, _ = _transform_openai_tools_to_gemini(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_web_search_tool_detection(self):
        """Test that web_search tool is detected."""
        tools = [
            {"type": "web_search"},
            {"type": "function", "function": {"name": "my_func"}},
        ]
        result, has_web_search = _transform_openai_tools_to_gemini(tools)
        assert has_web_search is True
        assert len(result) == 1
        assert result[0]["name"] == "my_func"

    def test_web_search_preview_tool_detection(self):
        """Test that web_search_preview tool is detected."""
        tools = [{"type": "web_search_preview"}]
        result, has_web_search = _transform_openai_tools_to_gemini(tools)
        assert has_web_search is True
        assert len(result) == 0


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


class TestResponsesAPITransformers:
    """Tests for Responses API transformers."""

    def test_simple_string_input(self):
        """Test transforming simple string input."""
        request = ResponsesRequest(
            model="gemini-2.5-flash",
            input="Hello, world!",
        )
        result = responses_request_to_gemini(request)
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello, world!"

    def test_input_with_messages(self):
        """Test transforming array input with messages."""
        request = ResponsesRequest(
            model="gemini-2.5-flash",
            input=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And 3+3?"},
            ],
        )
        result = responses_request_to_gemini(request)
        assert len(result["contents"]) == 3
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"
        assert result["contents"][2]["role"] == "user"

    def test_instructions_as_system(self):
        """Test that instructions become systemInstruction."""
        request = ResponsesRequest(
            model="gemini-2.5-flash",
            input="Hi",
            instructions="You are a helpful assistant.",
        )
        result = responses_request_to_gemini(request)
        assert "systemInstruction" in result
        assert (
            result["systemInstruction"]["parts"][0]["text"]
            == "You are a helpful assistant."
        )

    def test_tools_transformation(self):
        """Test transforming tools to Gemini format."""
        request = ResponsesRequest(
            model="gemini-2.5-flash",
            input="What's the weather?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
        )
        result = responses_request_to_gemini(request)
        assert "tools" in result
        func_decls = None
        for tool in result["tools"]:
            if "functionDeclarations" in tool:
                func_decls = tool["functionDeclarations"]
                break
        assert func_decls is not None
        assert func_decls[0]["name"] == "get_weather"

    def test_web_search_tool(self):
        """Test web_search tool becomes googleSearch."""
        request = ResponsesRequest(
            model="gemini-2.5-flash",
            input="Search for latest news",
            tools=[{"type": "web_search"}],
        )
        result = responses_request_to_gemini(request)
        assert "tools" in result
        assert any("googleSearch" in t for t in result["tools"])

    def test_function_call_output(self):
        """Test transforming function_call_output input."""
        request = ResponsesRequest(
            model="gemini-2.5-flash",
            input=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "output": '{"temperature": 22, "condition": "sunny"}',
                },
            ],
        )
        result = responses_request_to_gemini(request)
        # Function output should be converted to functionResponse
        assert len(result["contents"]) == 2
        assert "functionResponse" in result["contents"][1]["parts"][0]

    def test_gemini_response_to_responses_text(self):
        """Test transforming Gemini text response to Responses format."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello! How can I help you?"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18,
            },
        }
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-flash")

        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["output_text"] == "Hello! How can I help you?"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["role"] == "assistant"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 8

    def test_gemini_response_to_responses_function_call(self):
        """Test transforming Gemini function call to Responses format."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Tokyo"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-flash")

        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "function_call"
        assert result["output"][0]["name"] == "get_weather"
        assert json.loads(result["output"][0]["arguments"]) == {"location": "Tokyo"}
        assert result["output"][0]["call_id"].startswith("call_")

    def test_gemini_response_with_reasoning(self):
        """Test transforming Gemini response with reasoning/thought."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Let me think about this...", "thought": True},
                            {"text": "The answer is 42."},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-flash")

        # Should have reasoning item and message item
        types = [item["type"] for item in result["output"]]
        assert "reasoning" in types
        assert "message" in types
        assert result["output_text"] == "The answer is 42."

    def test_stream_events_text(self):
        """Test streaming events for text content."""
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
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )
        assert len(events) >= 1
        text_events = [e for e in events if e["type"] == "response.output_text.delta"]
        assert len(text_events) == 1
        assert text_events[0]["delta"] == "Hello"

    def test_stream_events_function_call(self):
        """Test streaming events for function call."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris"},
                                }
                            }
                        ],
                    },
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )

        # Should have item.added and arguments.done events
        event_types = [e["type"] for e in events]
        assert "response.output_item.added" in event_types
        assert "response.function_call_arguments.done" in event_types

    def test_stream_events_with_finish_reason(self):
        """Test streaming events when finish reason is present."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Final message"}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )
        event_types = [e["type"] for e in events]
        # Should have text delta event
        assert "response.output_text.delta" in event_types

    def test_stream_events_with_reasoning(self):
        """Test streaming events for reasoning/thought content."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Thinking about this...", "thought": True}],
                    },
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )
        # Reasoning chunks should generate events
        assert len(events) >= 1

    def test_stream_events_empty_parts(self):
        """Test streaming events handle empty parts gracefully."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [],
                    },
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )
        # Should not crash, may return empty or minimal events
        assert isinstance(events, list)

    def test_stream_events_multiple_text_parts(self):
        """Test streaming events with multiple text parts in one chunk."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "First part. "},
                            {"text": "Second part."},
                        ],
                    },
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )
        text_deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        # Should have text delta events
        assert len(text_deltas) >= 1

    def test_stream_events_response_id_included(self):
        """Test streaming events include response_id."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Test"}],
                    },
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_abc123", 0
        )
        for event in events:
            if "response_id" in event:
                assert event["response_id"] == "resp_abc123"

    def test_stream_events_output_index_tracking(self):
        """Test streaming events track output_index correctly."""
        gemini_chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Content"}],
                    },
                }
            ]
        }
        # First call with output_index=0
        events_0 = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 0
        )
        # Second call with output_index=1
        events_1 = gemini_stream_chunk_to_responses_events(
            gemini_chunk, "gemini-2.5-flash", "resp_123", 1
        )

        # Check output_index is tracked in events
        for event in events_0:
            if "output_index" in event:
                assert event["output_index"] == 0
        for event in events_1:
            if "output_index" in event:
                assert event["output_index"] == 1
