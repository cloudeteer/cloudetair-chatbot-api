"""
Docling-based document preprocessing implementation.

This module provides document extraction and conversion capabilities using Docling.
"""

import logging
import re
from pathlib import Path
from typing import Union

from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter

from .base_preprocessing_provider import BasePreprocessingProvider
from .utils import clean_text

logger = logging.getLogger(__name__)


class DoclingPreprocessor(BasePreprocessingProvider):
    """
    Docling-based implementation for document preprocessing.
    
    Uses Docling's DocumentConverter to extract and convert documents
    from various sources including HTML, PDF, Word, and other formats.
    
    Environment variables:
        None required - uses Docling's default configuration
    
    Example:
        >>> preprocessor = DoclingPreprocessor()
        >>> doc = preprocessor.extract("path/to/document.pdf")
        >>> markdown = preprocessor.to_markdown(doc)
        >>> text = preprocessor.to_text(doc)
    """
    
    def __init__(self):
        """Initialize the Docling preprocessor."""
        self.converter = DocumentConverter()
        logger.info("DoclingPreprocessor initialized")
    
    def extract(self, source: Union[str, Path]) -> DoclingDocument:
        """
        Extract and parse a document from the given source using Docling.
        
        Args:
            source: File path, URL, or other source identifier
            
        Returns:
            DoclingDocument: Structured document representation
            
        Raises:
            ValueError: If source is invalid or unsupported
            IOError: If source cannot be accessed or read
        """
        try:
            # Convert Path to string if needed
            source_str = str(source) if isinstance(source, Path) else source
            
            logger.info(f"Extracting document from source: {source_str}")
            
            # Use Docling's DocumentConverter to parse the document
            result = self.converter.convert(source_str)
            
            if not result or not result.document:
                raise ValueError(f"Failed to extract document from source: {source_str}")
            
            logger.info(f"Successfully extracted document with {len(result.document.texts)} text elements")
            return result.document
            
        except Exception as e:
            logger.error(f"Error extracting document from {source}: {str(e)}")
            if isinstance(e, (ValueError, IOError)):
                raise
            # Convert other exceptions to IOError
            raise IOError(f"Failed to extract document from {source}: {str(e)}") from e
    
    def to_markdown(self, doc: DoclingDocument) -> str:
        """
        Convert a DoclingDocument to Markdown format.
        
        Args:
            doc: The structured document to convert
            
        Returns:
            str: Markdown representation of the document
        """
        try:
            logger.debug("Converting DoclingDocument to Markdown")
            
            # Use Docling's built-in export to Markdown
            markdown = doc.export_to_markdown()
            
            if not markdown.strip():
                logger.warning("Document converted to empty Markdown")
                return ""
            
            logger.debug(f"Successfully converted to Markdown ({len(markdown)} characters)")
            return markdown
            
        except Exception as e:
            logger.error(f"Error converting document to Markdown: {str(e)}")
            raise ValueError(f"Failed to convert document to Markdown: {str(e)}") from e
    
    def to_text(self, doc: DoclingDocument) -> str:
        """
        Convert a DoclingDocument to plain text.
        
        Derives plaintext from Markdown, preserving structure like
        headings and lists while removing formatting markup.
        
        Args:
            doc: The structured document to convert
            
        Returns:
            str: Plain text representation of the document
        """
        try:
            # First get the Markdown representation
            markdown = self.to_markdown(doc)
            
            if not markdown.strip():
                return ""
            
            # Convert Markdown to plain text
            plain_text = self._markdown_to_text(markdown)
            
            logger.debug(f"Successfully converted to plain text ({len(plain_text)} characters)")
            return plain_text
            
        except Exception as e:
            logger.error(f"Error converting document to plain text: {str(e)}")
            raise ValueError(f"Failed to convert document to plain text: {str(e)}") from e
    
    def _markdown_to_text(self, markdown: str) -> str:
        """
        Convert Markdown to plain text while preserving structure.
        
        Args:
            markdown: Markdown content
            
        Returns:
            str: Plain text with preserved structure
        """
        # Remove code blocks but preserve their content
        text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', ''), markdown)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Convert headers to plain text but keep the text
        text = re.sub(r'^#{1,6}\s*(.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Remove bold/italic formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Convert links to just the text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Convert list items to plain text but preserve structure
        text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text


# Global instance for easy import
docling_preprocessor = DoclingPreprocessor()