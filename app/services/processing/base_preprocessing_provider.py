"""Base abstract class for document preprocessing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from docling.datamodel.document import DoclingDocument


class BasePreprocessingProvider(ABC):
    """
    Abstract base class for document preprocessing providers.
    
    Provides a unified interface for extracting and converting documents
    from various sources (files, URLs, etc.) into structured formats.
    """
    
    @abstractmethod
    def extract(self, source: Union[str, Path]) -> DoclingDocument:
        """
        Extract and parse a document from the given source.
        
        Args:
            source: File path, URL, or other source identifier
            
        Returns:
            DoclingDocument: Structured document representation
            
        Raises:
            ValueError: If source is invalid or unsupported
            IOError: If source cannot be accessed or read
        """
        pass
    
    @abstractmethod
    def to_markdown(self, doc: DoclingDocument) -> str:
        """
        Convert a DoclingDocument to Markdown format.
        
        Args:
            doc: The structured document to convert
            
        Returns:
            str: Markdown representation of the document
        """
        pass
    
    @abstractmethod
    def to_text(self, doc: DoclingDocument) -> str:
        """
        Convert a DoclingDocument to plain text.
        
        This should derive plaintext from Markdown, preserving
        structure like headings and lists while removing formatting.
        
        Args:
            doc: The structured document to convert
            
        Returns:
            str: Plain text representation of the document
        """
        pass