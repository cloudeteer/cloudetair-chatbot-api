"""
Embedding services module.

This module contains embedding service providers that implement the BaseEmbeddingProvider interface.
"""

from .base_embedding_provider import BaseEmbeddingProvider
from .azure_openai import AzureEmbeddingProvider, azure_embedding_provider

__all__ = [
    "BaseEmbeddingProvider",
    "AzureEmbeddingProvider", 
    "azure_embedding_provider",
]
