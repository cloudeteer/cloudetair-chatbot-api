"""
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any


class Message(BaseModel):
    """Represents a chat message with role and content."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions following OpenAI API format."""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


class Choice(BaseModel):
    """A single choice in a chat completion response."""
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""
    id: str
    object: str
    choices: List[Choice]


class ModelData(BaseModel):
    """Model information for the models endpoint."""
    id: str
    object: str


class ModelsResponse(BaseModel):
    """Response model for available models."""
    data: List[ModelData]


class Delta(BaseModel):
    """Delta changes in streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """A single choice in a streaming chat completion response."""
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Chunk in a streaming chat completion response."""
    id: str
    object: str
    choices: List[StreamChoice]
