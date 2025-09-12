"""
Base LLM Provider abstract class.

This module defines the common interface that all LLM service providers must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from app.models.chat import ChatCompletionRequest


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM service providers.

    This class defines the common interface that all concrete LLM providers must implement.
    It ensures consistency across different LLM services and makes the system extensible.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available and properly configured.

        Returns:
            bool: True if the service is available, False otherwise
        """
        raise NotImplementedError("Subclasses must implement is_available")

    @abstractmethod
    async def generate_response(self, req: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate a chat completion response.

        This method should create a non-streaming chat completion response
        following the OpenAI API format.

        Args:
            req: Chat completion request containing model, messages, and parameters

        Returns:
            Chat completion response in OpenAI API format

        Raises:
            HTTPException: If the service is not configured or API call fails
        """
        raise NotImplementedError("Subclasses must implement generate_response")

    @abstractmethod
    async def generate_streaming_response(
        self, req: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming chat completion response.

        This method should create a streaming chat completion response
        following the OpenAI streaming API format with Server-Sent Events.

        Args:
            req: Chat completion request containing model, messages, and parameters

        Yields:
            Server-sent events in OpenAI streaming format

        Raises:
            HTTPException: If the service is not configured or API call fails
        """
        raise NotImplementedError(
            "Subclasses must implement generate_streaming_response"
        )
        # This will never execute, but Python needs it for type checking
        yield  # pragma: no cover
