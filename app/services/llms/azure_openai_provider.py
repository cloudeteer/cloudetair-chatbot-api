"""
Azure OpenAI service provider implementation.

This module contains the Azure OpenAI service provider that inherits from BaseLLMProvider.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException
from openai import AsyncAzureOpenAI

from app.models.chat import ChatCompletionRequest

from .base_llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI service provider implementation."""

    def __init__(self):
        """Initialize Azure OpenAI service with environment configuration."""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.embedding_api_version = os.getenv(
            "AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15"
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

        self.client = None
        if self.endpoint and self.api_key:
            # Main client for chat completions
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
            logger.info("Azure OpenAI provider initialized successfully")
        else:
            logger.warning(
                "Azure OpenAI provider not configured - missing endpoint or API key"
            )

    def is_available(self) -> bool:
        """Check if Azure OpenAI service is available."""
        return self.client is not None

    async def generate_response(self, req: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate a chat completion response using Azure OpenAI.

        Args:
            req: Chat completion request

        Returns:
            Chat completion response in OpenAI format

        Raises:
            HTTPException: If service is not configured or API call fails
        """
        if not self.client:
            raise HTTPException(
                status_code=500,
                detail="Azure OpenAI is not configured. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.",
            )

        try:
            logger.info(
                f"Creating chat completion with Azure OpenAI using model: {self.deployment_name}, API version: {self.api_version}, endpoint: {self.endpoint}."
            )

            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in req.messages
            ]

            # Call Azure OpenAI
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=openai_messages,  # type: ignore
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

            # Convert response to our format
            result = {
                "id": response.id,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ],
            }

            logger.info("Azure OpenAI chat completion successful")
            return result

        except Exception as e:
            logger.error(f"Azure OpenAI error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Azure OpenAI error: {str(e)}"
            ) from e

    async def generate_streaming_response(
        self, req: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming chat completion response using Azure OpenAI.

        Args:
            req: Chat completion request

        Yields:
            Server-sent events in OpenAI streaming format

        Raises:
            HTTPException: If service is not configured or API call fails
        """
        if not self.client:
            raise HTTPException(
                status_code=500,
                detail="Azure OpenAI is not configured. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.",
            )

        try:
            logger.info(
                f"Creating streaming chat completion with Azure OpenAI using model: {self.deployment_name}"
            )

            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in req.messages
            ]

            # Call Azure OpenAI with streaming
            stream = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=openai_messages,  # type: ignore
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
            )

            # Forward the stream
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Convert to our format
                    sse_chunk = {
                        "id": chunk.id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": choice.index,
                                "delta": {},
                                "finish_reason": choice.finish_reason,
                            }
                        ],
                    }

                    # Add delta content if available
                    if hasattr(choice, "delta") and choice.delta:
                        if hasattr(choice.delta, "role") and choice.delta.role:
                            sse_chunk["choices"][0]["delta"]["role"] = choice.delta.role
                        if hasattr(choice.delta, "content") and choice.delta.content:
                            sse_chunk["choices"][0]["delta"][
                                "content"
                            ] = choice.delta.content

                    yield f"data: {json.dumps(sse_chunk)}\n\n"

            # Send done signal
            yield "data: [DONE]\n\n"
            logger.info("Azure OpenAI streaming completion successful")

        except Exception as e:
            logger.error(f"Azure OpenAI streaming error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Azure OpenAI streaming error: {str(e)}"
            ) from e


# Global provider instance
azure_openai_provider = AzureOpenAIProvider()
