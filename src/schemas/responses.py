"""
Pydantic schemas for OpenAI Responses API format.

The Responses API is a newer, simpler API that uses 'Items' for input/output,
supporting message, function_call, and function_call_output types.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ResponsesInputMessage(BaseModel):
    """Input message item for Responses API."""

    role: str = Field(
        ...,
        description="The role of the message author",
        examples=["user", "assistant", "system"],
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None,
        description="The content of the message",
    )


class ResponsesFunctionCall(BaseModel):
    """Function call item in Responses API output."""

    type: str = Field(default="function_call", description="Item type")
    id: Optional[str] = Field(default=None, description="Unique ID for this item")
    call_id: str = Field(..., description="Unique identifier for this function call")
    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON string of function arguments")
    status: Optional[str] = Field(default="completed", description="Status of the call")


class ResponsesFunctionCallOutput(BaseModel):
    """Function call output item for providing results back to the model."""

    type: str = Field(default="function_call_output", description="Item type")
    call_id: str = Field(
        ..., description="The call_id of the function call this output responds to"
    )
    output: str = Field(..., description="JSON-serialized function execution result")


class ResponsesToolFunction(BaseModel):
    """Function tool definition for Responses API."""

    type: str = Field(default="function", description="Tool type")
    name: str = Field(..., description="Function name")
    description: Optional[str] = Field(
        default=None, description="Description of what the function does"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for function parameters"
    )
    strict: Optional[bool] = Field(
        default=None, description="Enable strict schema validation"
    )


class ResponsesRequest(BaseModel):
    """
    Request body for OpenAI Responses API.

    The Responses API uses 'input' instead of 'messages' and supports
    a simplified tool definition format.
    """

    model: str = Field(
        ...,
        description="Model ID to use",
        examples=["gemini-2.5-pro", "gemini-2.5-flash"],
    )
    input: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Input text or array of input items (messages, function outputs)",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="System instructions (replaces system message)",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tools available to the model",
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Controls which tool is called",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0,
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling probability",
        ge=0.0,
        le=1.0,
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
        ge=1,
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    store: Optional[bool] = Field(
        default=True,
        description="Whether to store the response (ignored, always false for proxy)",
    )
    previous_response_id: Optional[str] = Field(
        default=None,
        description="Previous response ID for chaining (ignored for proxy)",
    )
    reasoning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reasoning configuration for thinking models",
    )

    model_config = ConfigDict(extra="allow")


class ResponsesOutputMessage(BaseModel):
    """Message output item in Responses API."""

    id: Optional[str] = Field(default=None, description="Unique ID for this item")
    type: str = Field(default="message", description="Item type")
    role: str = Field(default="assistant", description="Message role")
    status: str = Field(default="completed", description="Message status")
    content: List[Dict[str, Any]] = Field(
        default_factory=list, description="Message content parts"
    )


class ResponsesOutputItem(BaseModel):
    """Generic output item - can be message, function_call, or reasoning."""

    id: Optional[str] = Field(default=None, description="Unique ID for this item")
    type: str = Field(..., description="Item type (message, function_call, reasoning)")
    # Fields for message type
    role: Optional[str] = Field(default=None, description="Message role")
    status: Optional[str] = Field(default=None, description="Status")
    content: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Message content"
    )
    # Fields for function_call type
    call_id: Optional[str] = Field(default=None, description="Function call ID")
    name: Optional[str] = Field(default=None, description="Function name")
    arguments: Optional[str] = Field(default=None, description="Function arguments")
    # Fields for reasoning type
    summary: Optional[List[Any]] = Field(default=None, description="Reasoning summary")


class ResponsesUsage(BaseModel):
    """Token usage information."""

    input_tokens: int = Field(default=0, description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")


class ResponsesResponse(BaseModel):
    """Response from Responses API endpoint."""

    id: str = Field(..., description="Unique identifier for this response")
    object: str = Field(default="response", description="Object type")
    created_at: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used")
    output: List[Dict[str, Any]] = Field(
        default_factory=list, description="Output items"
    )
    output_text: Optional[str] = Field(
        default=None, description="Convenience property for text output"
    )
    usage: Optional[ResponsesUsage] = Field(default=None, description="Token usage")
    status: str = Field(default="completed", description="Response status")


class ResponsesStreamEvent(BaseModel):
    """Streaming event for Responses API."""

    type: str = Field(..., description="Event type")
    response_id: Optional[str] = Field(default=None, description="Response ID")
    output_index: Optional[int] = Field(default=None, description="Output item index")
    item: Optional[Dict[str, Any]] = Field(default=None, description="Item data")
    delta: Optional[str] = Field(default=None, description="Delta content")
    content_index: Optional[int] = Field(default=None, description="Content index")
