"""
Azure OpenAI embedding service provider implementation.

This module contains the Azure OpenAI embedding service provider that inherits from BaseEmbeddingProvider.
"""

import logging
import os
from typing import List, Optional, Literal

from openai import AsyncAzureOpenAI
from fastapi import HTTPException

from .base_embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding service provider implementation."""

    def __init__(self):
        """Initialize Azure OpenAI embedding service with environment configuration."""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15")
        
        # Prioritize new environment variable name, fallback to legacy, then safe default
        self.embedding_deployment = (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or
            "text-embedding-3-small"  # Safe default for embedding
        )
        
        # Embedding dimensions for different models
        self.embedding_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        self.client = None
        if self.endpoint and self.api_key:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
            logger.info(f"Azure embedding provider initialized successfully with deployment: {self.embedding_deployment}")
        else:
            logger.warning(
                "Azure embedding provider not configured - missing endpoint or API key"
            )

    def is_available(self) -> bool:
        """Check if Azure OpenAI embedding service is available."""
        return self.client is not None

    async def create_embedding(
        self, 
        text: str, 
        memory_action: Optional[Literal["add", "search", "update"]] = None
    ) -> List[float]:
        """
        Create embedding for the given text using Azure OpenAI.

        Args:
            text: Text to embed
            memory_action: Optional action context for embedding optimization
                         ("add" for storing new memories, "search" for queries, 
                          "update" for modifying existing memories)

        Returns:
            Embedding vector as list of floats

        Raises:
            HTTPException: If service is not configured or API call fails
        """
        if not self.client:
            raise HTTPException(
                status_code=500,
                detail="Azure OpenAI embedding service is not configured. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.",
            )

        try:
            # Clean text for embedding
            cleaned_text = text.replace("\n", " ").strip()
            
            logger.debug(f"Creating embedding for text: {cleaned_text[:100]}...")
            
            # Call Azure OpenAI embedding API
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,
                input=cleaned_text
            )

            embedding = response.data[0].embedding
            
            logger.debug(f"Successfully created embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Azure OpenAI embedding error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Azure OpenAI embedding error: {str(e)}"
            ) from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Number of dimensions in the embedding vector
        """
        return self.embedding_dimensions.get(self.embedding_deployment, 1536)


# Global provider instance
azure_embedding_provider = AzureEmbeddingProvider()
