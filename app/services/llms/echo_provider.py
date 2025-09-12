"""
Echo service provider implementation.

This module contains the Echo service provider that inherits from BaseLLMProvider.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.models.chat import ChatCompletionRequest

from .base_llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class EchoProvider(BaseLLMProvider):
    """Simple echo service provider that repeats user input for testing."""

    def is_available(self) -> bool:
        """Check if Echo service is available (always true)."""
        return True

    async def generate_response(self, req: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate an echo chat completion response.

        Args:
            req: Chat completion request

        Returns:
            Echo response in OpenAI format
        """
        logger.info("Creating echo chat completion")

        # Extract user messages
        user_messages = [m.content for m in req.messages if m.role == "user"]

        # Handle case with no user messages
        if not user_messages:
            user_message = "No user message provided"
        else:
            # Get the last user message
            user_message = user_messages[-1]

        response = {
            "id": "chatcmpl-echo",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"You said: {user_message}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        logger.info("Echo chat completion successful")
        return response

    async def generate_streaming_response(
        self, req: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming echo chat completion response.

        Args:
            req: Chat completion request

        Yields:
            Server-sent events in OpenAI streaming format
        """
        logger.info("Creating streaming echo chat completion")

        # Extract user messages
        user_messages = [m.content for m in req.messages if m.role == "user"]

        # Handle case with no user messages
        if not user_messages:
            user_message = "No user message provided"
        else:
            # Get the last user message
            user_message = user_messages[-1]

        response_text = f"You said: {user_message}"

        # Stream the response word by word
        words = response_text.split()

        # Send initial chunk with role
        chunk = {
            "id": "chatcmpl-echo-stream",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Send content chunks
        for i, word in enumerate(words):
            content = word if i == 0 else f" {word}"
            chunk = {
                "id": "chatcmpl-echo-stream",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # Simulate streaming delay
            await asyncio.sleep(0.1)

        # Send final chunk with finish_reason
        chunk = {
            "id": "chatcmpl-echo-stream",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Send done signal
        yield "data: [DONE]\n\n"

        logger.info("Echo streaming completion successful")

# Global provider instance
echo_provider = EchoProvider()
