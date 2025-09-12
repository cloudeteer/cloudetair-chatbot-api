"""Docling-based document chunking implementation."""

import logging
from typing import List, Optional

from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker import HierarchicalChunker

from ...models.chunk import DocumentChunk
from .base_chunking_provider import BaseChunkingProvider

logger = logging.getLogger(__name__)


class DoclingHierarchicalChunker(BaseChunkingProvider):
    """
    Docling-based implementation for document chunking using HierarchicalChunker.
    
    Uses Docling's HierarchicalChunker with tokenizer-aware splitting and merging
    to create optimal chunks for embedding and retrieval.
    
    The chunker respects document structure and ensures no chunk exceeds
    the specified token limit while maintaining semantic coherence.
    
    Environment variables:
        None required - uses sensible defaults
    
    Example:
        >>> chunker = DoclingHierarchicalChunker(max_tokens=1000)
        >>> chunks = chunker.chunk(docling_document)
        >>> assert all(len(chunk.text) > 0 for chunk in chunks)
    """
    
    def __init__(
        self,
        max_tokens: int = 1000,
        tokenizer: str = "cl100k_base"  # OpenAI tiktoken tokenizer
    ):
        """
        Initialize the Docling hierarchical chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk (default: 1000)
            tokenizer: Tokenizer to use for token counting (default: cl100k_base for OpenAI)
        """
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        
        # Initialize the Docling HierarchicalChunker
        self.chunker = HierarchicalChunker(
            max_tokens=max_tokens,
            tokenizer=tokenizer
        )
        
        logger.info(f"DoclingHierarchicalChunker initialized with max_tokens={max_tokens}")
    
    def chunk(self, doc: DoclingDocument) -> List[DocumentChunk]:
        """
        Split a DoclingDocument into chunks using HierarchicalChunker.
        
        Args:
            doc: The structured document to chunk
            
        Returns:
            List[DocumentChunk]: List of document chunks with metadata
            
        Raises:
            ValueError: If document is invalid or cannot be chunked
        """
        if not doc:
            raise ValueError("Document cannot be None. Please provide a valid Document.")
        
        try:
            logger.info(f"Starting chunking process for document")
            
            # Use Docling's HierarchicalChunker to split the document
            chunks = list(self.chunker.chunk(doc))
            
            if not chunks:
                logger.warning("No chunks produced from document")
                return []
            
            # Convert Docling chunks to DocumentChunk objects
            document_chunks = []
            doc_id = self._get_document_id(doc)
            
            for i, chunk in enumerate(chunks):
                try:
                    document_chunk = self._convert_chunk(chunk, doc_id, i)
                    document_chunks.append(document_chunk)
                except Exception as e:
                    logger.warning(f"Failed to convert chunk {i}: {str(e)}")
                    continue
            
            logger.info(f"Successfully created {len(document_chunks)} chunks from document")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise ValueError(f"Failed to chunk document: {str(e)}") from e
    
    def _get_document_id(self, doc: DoclingDocument) -> str:
        """
        Extract or generate a document ID from the DoclingDocument.
        
        Args:
            doc: The DoclingDocument
            
        Returns:
            str: Document identifier
        """
        # Try to get document name or URI from metadata
        if hasattr(doc, 'name') and doc.name:
            return doc.name
        
        if hasattr(doc, 'origin') and doc.origin:
            if hasattr(doc.origin, 'filename') and doc.origin.filename:
                return doc.origin.filename
            if hasattr(doc.origin, 'binary_hash') and doc.origin.binary_hash:
                return f"doc_{doc.origin.binary_hash[:12]}"
        
        # Fallback to a generic ID
        return f"document_{id(doc)}"
    
    def _convert_chunk(self, chunk, doc_id: str, chunk_index: int) -> DocumentChunk:
        """
        Convert a Docling BaseChunk to a DocumentChunk.
        
        Args:
            chunk: Docling BaseChunk object
            doc_id: Document identifier
            chunk_index: Index of this chunk in the document
            
        Returns:
            DocumentChunk: Converted chunk with metadata
        """
        # Extract basic text content
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        # Generate chunk ID
        chunk_id = f"chunk-{chunk_index:03d}"
        
        # Extract title from chunk if available
        title = None
        if hasattr(chunk, 'meta') and chunk.meta:
            # Handle both dict-like and object-like metadata
            if hasattr(chunk.meta, 'get'):
                title = chunk.meta.get('title') or chunk.meta.get('heading')
            else:
                title = getattr(chunk.meta, 'title', None) or getattr(chunk.meta, 'heading', None)
        
        # Extract section path from chunk hierarchy
        section_path = None
        if hasattr(chunk, 'meta') and chunk.meta:
            # Try to get hierarchical path
            if hasattr(chunk.meta, 'get'):
                if 'section_path' in chunk.meta:
                    section_path = chunk.meta['section_path']
                elif 'headings' in chunk.meta and isinstance(chunk.meta['headings'], list):
                    section_path = chunk.meta['headings']
            else:
                section_path = getattr(chunk.meta, 'section_path', None)
                if not section_path:
                    headings = getattr(chunk.meta, 'headings', None)
                    if headings and isinstance(headings, list):
                        section_path = headings
        
        # Extract page number if available
        page_no = None
        if hasattr(chunk, 'meta') and chunk.meta:
            if hasattr(chunk.meta, 'get'):
                page_no = chunk.meta.get('page_no')
            else:
                page_no = getattr(chunk.meta, 'page_no', None)
        
        # Build metadata dictionary
        meta = {
            'chunk_index': chunk_index,
            'token_count': len(chunk_text.split()) if chunk_text else 0  # Rough estimate
        }
        
        # Add any additional metadata from the chunk
        if hasattr(chunk, 'meta') and chunk.meta:
            if hasattr(chunk.meta, 'get'):
                # Dictionary-like metadata
                meta.update({k: v for k, v in chunk.meta.items() 
                            if k not in ['title', 'heading', 'section_path', 'headings', 'page_no']})
            else:
                # Object-like metadata - convert to dict
                try:
                    meta_dict = vars(chunk.meta) if hasattr(chunk.meta, '__dict__') else {}
                    meta.update({k: v for k, v in meta_dict.items() 
                                if k not in ['title', 'heading', 'section_path', 'headings', 'page_no']})
                except Exception:
                    # If we can't extract metadata, just continue
                    pass
        
        return DocumentChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=chunk_text,
            title=title,
            section_path=section_path,
            page_no=page_no,
            meta=meta
        )


# Global instance for easy import (keep the original name for backward compatibility)
docling_hybrid_chunker = DoclingHierarchicalChunker()
docling_hierarchical_chunker = DoclingHierarchicalChunker()