"""Base vector store provider interface."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from ...models.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    Provides a unified interface for storing, retrieving, and managing document chunks
    with their vector embeddings across different vector database implementations.
    
    All implementations should support:
    - Document storage with automatic embedding generation
    - Vector similarity search
    - Index management operations
    - Idempotent operations (safe to re-run)
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the vector store is available and properly configured.
        
        Returns:
            True if the vector store can accept operations, False otherwise
        """
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[DocumentChunk]) -> None:
        """
        Add or update document chunks in the vector store.
        
        This method should:
        - Generate embeddings for document chunks if not already present
        - Store documents with their vector representations
        - Handle duplicates gracefully (upsert behavior)
        - Create the index if it doesn't exist
        
        Args:
            documents: List of DocumentChunk objects to store
            
        Raises:
            Exception: If storage operation fails
        """
        pass
    
    @abstractmethod
    async def query_by_vector(self, query_vector: List[float], top_k: int = 10) -> List[DocumentChunk]:
        """
        Search for similar documents using a query vector.
        
        Args:
            query_vector: The embedding vector to search with
            top_k: Maximum number of results to return
            
        Returns:
            List of DocumentChunk objects ordered by similarity (most similar first)
            
        Raises:
            Exception: If search operation fails
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """
        Remove documents from the vector store by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Raises:
            Exception: If deletion operation fails
        """
        pass
    
    # Index management methods
    
    @abstractmethod
    async def create_index(self, index_name: Optional[str] = None, **kwargs) -> None:
        """
        Create a new index in the vector store.
        
        This operation should be idempotent - creating an existing index should not fail.
        
        Args:
            index_name: Name of the index to create. If None, uses the default configured index
            **kwargs: Additional index configuration parameters specific to the implementation
            
        Raises:
            Exception: If index creation fails
        """
        pass
    
    @abstractmethod
    async def list_indexes(self) -> List[str]:
        """
        List all available indexes in the vector store.
        
        Returns:
            List of index names
            
        Raises:
            Exception: If listing operation fails
        """
        pass
    
    @abstractmethod
    async def delete_index(self, index_name: Optional[str] = None) -> None:
        """
        Delete an index from the vector store.
        
        This operation should be safe - deleting a non-existent index should not fail.
        
        Args:
            index_name: Name of the index to delete. If None, uses the default configured index
            
        Raises:
            Exception: If deletion operation fails
        """
        pass
    
    # Helper methods that can be overridden by implementations
    
    def get_default_index_name(self) -> str:
        """
        Get the default index name for this vector store.
        
        Returns:
            Default index name
        """
        return "embeddings-index"
    
    async def search_by_text(self, query_text: str, top_k: int = 10, embedding_provider=None) -> List[DocumentChunk]:
        """
        Convenience method to search using text query.
        
        This method generates an embedding for the query text and then performs vector search.
        Implementations can override this for more efficient text-based search.
        
        Args:
            query_text: Text query to search for
            top_k: Maximum number of results to return
            embedding_provider: Optional embedding provider to use for query embedding
            
        Returns:
            List of DocumentChunk objects ordered by similarity
            
        Raises:
            Exception: If search operation fails
        """
        if not embedding_provider:
            raise ValueError("Embedding provider is required for text-based search")
        
        if not embedding_provider.is_available():
            raise ValueError("Embedding provider is not available")
        
        query_vector = await embedding_provider.create_embedding(query_text, "search")
        return await self.query_by_vector(query_vector, top_k)