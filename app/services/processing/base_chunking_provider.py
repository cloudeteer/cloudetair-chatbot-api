"""Base abstract class for document chunking."""

from abc import ABC, abstractmethod
from typing import List

from docling.datamodel.document import DoclingDocument
from ...models.chunk import DocumentChunk


class BaseChunkingProvider(ABC):
    """
    Abstract base class for document chunking providers.
    
    Provides a unified interface for splitting documents into
    manageable chunks for embedding and retrieval.
    """
    
    @abstractmethod
    def chunk(self, doc: DoclingDocument) -> List[DocumentChunk]:
        """
        Split a document into chunks.
        
        Args:
            doc: The structured document to chunk
            
        Returns:
            List[DocumentChunk]: List of document chunks with metadata
            
        Raises:
            ValueError: If document is invalid or cannot be chunked
        """
        pass