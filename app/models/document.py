"""
Document model for data loaders.

This module defines the standardized Document structure that all data loaders must return.
"""

from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Standardized document structure for data loaders.
    
    All data loaders must return documents in this format to ensure
    consistent processing throughout the RAG pipeline.
    """
    
    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full text content of the document")
    last_modified: datetime = Field(..., description="When the document was last modified")
    link: str = Field(..., description="URL or link to the original document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the document")
        
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(id='{self.id}', title='{self.title[:50]}...')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the document."""
        return (
            f"Document(id='{self.id}', title='{self.title}', "
            f"content_length={len(self.content)}, last_modified='{self.last_modified.isoformat()}')"
        )