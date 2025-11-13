"""Document chunk model for text processing and chunking."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Represents a chunk of a document with metadata.
    
    Used for storing processed document chunks from various sources
    like Confluence pages, PDFs, etc. Contains the text content
    and contextual metadata for retrieval and embedding.
    """
    
    doc_id: str = Field(
        description="Unique identifier for the source document"
    )
    
    chunk_id: str = Field(
        description="Unique identifier for this chunk within the document"
    )
    
    text: str = Field(
        description="The actual text content of the chunk"
    )
    
    title: Optional[str] = Field(
        default=None,
        description="Title or heading associated with this chunk"
    )
    
    section_path: Optional[List[str]] = Field(
        default=None,
        description="Hierarchical path of sections/headings leading to this chunk"
    )
    
    page_no: Optional[int] = Field(
        default=None,
        description="Page number if the source document has pages"
    )
    
    meta: Dict = Field(
        default_factory=dict,
        description="Additional metadata about the chunk (e.g., source, timestamps, etc.)"
    )
    
    sourceurl: Optional[str] = Field(
        default=None,
        description="URL of the source document if applicable"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "doc_id": "confluence-page-123",
                "chunk_id": "chunk-001",
                "text": "This is the introduction section explaining the key concepts...",
                "title": "Introduction",
                "section_path": ["Getting Started", "Introduction"],
                "page_no": 1,
                "meta": {
                    "source_url": "https://company.atlassian.net/wiki/spaces/DOC/pages/123",
                    "last_modified": "2025-01-15T10:30:00Z"
                },
                "sourceurl": "https://company.atlassian.net/wiki/spaces/DOC/pages/123"
            }
        }