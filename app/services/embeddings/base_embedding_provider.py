"""
Base Embedding Provider abstract class.

This module defines the common interface that all embedding service providers must implement,
following the same pattern as the LLM providers in the system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Literal


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for all embedding service providers.

    This class defines the common interface that all concrete embedding providers must implement.
    It ensures consistency across different embedding services and makes the system extensible.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the embedding service is available and properly configured.

        Returns:
            bool: True if the service is available, False otherwise
        """
        raise NotImplementedError("Subclasses must implement is_available")

    @abstractmethod
    async def create_embedding(
        self, 
        text: str, 
        memory_action: Optional[Literal["add", "search", "update"]] = None
    ) -> List[float]:
        """
        Create embedding for the given text.

        This method should create an embedding vector for the provided text.
        The memory_action parameter can be used to optimize embedding generation
        for different use cases.

        Args:
            text: Text to embed
            memory_action: Optional action context for embedding optimization
                         ("add" for storing new memories, "search" for queries, 
                          "update" for modifying existing memories)

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If the service is not configured or API call fails
        """
        raise NotImplementedError("Subclasses must implement create_embedding")
