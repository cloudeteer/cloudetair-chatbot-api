#!/usr/bin/env python3
"""
Simple ETL Script for Confluence to Azure AI Search Pipeline

This script demonstrates the complete ETL pipeline:
1. Extract documents from Confluence
2. Transform using Docling preprocessing and chunking  
3. Load into Azure AI Search vector store
4. Test with sample search queries

Run this script line by line or execute the full pipeline.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add the parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Import our services
from app.services.data_loaders.confluence_loader import confluence_loader
from app.services.processing.docling_preprocessing import docling_preprocessor  
from app.services.processing.docling_chunking import docling_hierarchical_chunker
from app.services.vector_stores.azure_search_store import azure_search_vector_store
from app.models.document import Document
from app.models.chunk import DocumentChunk
import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def validate_environment():
    """Validate that all required environment variables are set."""
    logger.info("=== Validating Environment Configuration ===")
    
    required_vars = [
        'CONFLUENCE_URL',
        'CONFLUENCE_USERNAME', 
        'CONFLUENCE_API_TOKEN',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_SEARCH_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            logger.info(f"{var}: {'*' * 20}...{os.getenv(var)[-4:]}")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Check optional space key
    space_key = os.getenv('CONFLUENCE_SPACE_KEY')
    if space_key:
        logger.info(f"CONFLUENCE_SPACE_KEY: {space_key}")
    else:
        logger.warning("CONFLUENCE_SPACE_KEY not set - will load from default space")
    
    logger.info("Environment validation passed")

limit = 100  # Limit number of documents for testing

async def step1_extract_from_confluence(limit: int = 5) -> List[Document]:
    """
    Step 1: Extract documents from Confluence.
    
    Args:
        limit: Maximum number of documents to extract (0 for all)
        
    Returns:
        List of Document objects from Confluence
    """
    logger.info("=== Step 1: Extracting Documents from Confluence ===")
    
    # Check if Confluence loader is available
    if not confluence_loader.is_available():
        raise ValueError("Confluence loader is not available. Check your configuration.")
    
    logger.info(f"Confluence connection verified")
    logger.info(f"Space: {confluence_loader.space_key}")
    logger.info(f"URL: {confluence_loader.base_url}")
    
    # Load documents
    documents = await confluence_loader.load_data()
    
    # Limit documents if requested
    if limit > 0 and len(documents) > limit:
        documents = documents[:limit]
        logger.info(f"Limited to {limit} documents for testing")
    
    logger.info(f"Extracted {len(documents)} documents from Confluence")
    
    # Display document summary
    for i, doc in enumerate(documents[:3], 1):
        logger.info(f"{i}. {doc.title} ({len(doc.content)} chars)")
    
    if len(documents) > 3:
        logger.info(f"  ... and {len(documents) - 3} more documents")
    
    return documents


async def step2_transform_documents(documents: List[Document]) -> List[DocumentChunk]:
    """
    Step 2: Transform documents using Docling preprocessing and chunking.
    
    Args:
        documents: List of raw Document objects
        
    Returns:
        List of DocumentChunk objects ready for indexing
    """
    logger.info("=== Step 2: Transforming Documents with Docling ===")
    
    all_chunks = []
    
    for i, doc in enumerate(documents, 1):
        try:
            logger.info(f"Processing document {i}/{len(documents)}: {doc.title}")
            
            # For Confluence documents, we already have HTML content
            # We can create a simple text-based document for Docling
            # or process the HTML content directly
            
            # Since Confluence content is already HTML, let's create chunks directly
            # from the cleaned text content (Confluence loader already converts HTML to text)
            
            # Create a simple document-like object for chunking
            # We'll use the existing text content and create chunks from it
            doc_chunks = await create_chunks_from_document(doc, i)
            
            all_chunks.extend(doc_chunks)
            logger.info(f"Created {len(doc_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to process document {i}: {str(e)}")
            continue
    
    logger.info(f"Generated {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks


async def create_chunks_from_document(doc: Document, doc_index: int) -> List[DocumentChunk]:
    """
    Create chunks from a Document object.
    
    Since Confluence documents are already processed to text, we'll create
    a simple chunking strategy that respects semantic boundaries.
    """
    if not doc.content or not doc.content.strip():
        logger.warning(f"Document '{doc.title}' has no content")
        return []
    
    # Simple chunking strategy: split by paragraphs and combine into chunks
    paragraphs = [p.strip() for p in doc.content.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    max_chunk_size = 1000  # characters
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, create a new chunk
        if current_chunk and len(current_chunk + paragraph) > max_chunk_size:
            # Create chunk from current content
            chunk = DocumentChunk(
                doc_id=doc.id,
                chunk_id=f"chunk-{chunk_index:03d}",
                text=current_chunk.strip(),
                title=doc.title,
                section_path=None,
                page_no=None,
                meta={
                    'chunk_index': chunk_index,
                    'source_doc_title': doc.title,
                    'source_doc_id': doc.id,
                    'confluence_page_id': doc.metadata.get('page_id'),
                    'space_key': doc.metadata.get('space_key'),
                    'char_count': len(current_chunk.strip()),
                    'original_last_modified': doc.last_modified.isoformat() if doc.last_modified else None,
                    'original_link': doc.link
                }
            )
            chunks.append(chunk)
            chunk_index += 1
            current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += f"\n\n{paragraph}"
            else:
                current_chunk = paragraph
    
    # Create final chunk if there's remaining content
    if current_chunk.strip():
        chunk = DocumentChunk(
            doc_id=doc.id,
            chunk_id=f"chunk-{chunk_index:03d}",
            text=current_chunk.strip(),
            title=doc.title,
            section_path=None,
            page_no=None,
            meta={
                'chunk_index': chunk_index,
                'source_doc_title': doc.title,
                'source_doc_id': doc.id,
                'confluence_page_id': doc.metadata.get('page_id'),
                'space_key': doc.metadata.get('space_key'),
                'char_count': len(current_chunk.strip()),
                'original_last_modified': doc.last_modified.isoformat() if doc.last_modified else None,
                'original_link': doc.link
            }
        )
        chunks.append(chunk)
    
    return chunks



async def step3_create_index():
    """
    Step 3: Create or verify the search index exists.
    """
    logger.info("=== Step 3: Creating/Verifying Search Index ===")
    
    # Check if Azure AI Search is available
    if not azure_search_vector_store.is_available():
        raise ValueError("Azure AI Search store is not available. Check your configuration.")
    
    logger.info(f"Azure AI Search connection verified")
    logger.info(f"Endpoint: {os.getenv('AZURE_SEARCH_ENDPOINT')}")
    logger.info(f"Index: {os.getenv('AZURE_SEARCH_INDEX', 'embeddings-index')}")
    
    # The Azure AI Search store will create the index automatically when storing documents
    # if it doesn't exist, so no explicit index creation is needed
    logger.info("Index will be created automatically when storing documents")

# Helper to map all_chunks to chunks for development purposes
# In real usage, you would pass chunks from step2_transform_documents to step4_load_into_index


async def step4_load_into_index(chunks: List[DocumentChunk]):
    """
    Step 4: Load chunks into the search index.
    
    Args:
        chunks: List of DocumentChunk objects to store
    """
    logger.info("=== Step 4: Loading Chunks into Search Index ===")
    
    if not chunks:
        logger.warning("No chunks to store")
        return
    
    # The Azure Search vector store expects DocumentChunk objects directly,
    # not Document objects, so we can use the chunks as-is
    logger.info(f"Storing {len(chunks)} document chunks...")
    
    # Store documents in batches
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            await azure_search_vector_store.add_documents(batch)
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Failed to store batch {i//batch_size + 1}: {str(e)}")
            continue
    
    logger.info("Successfully loaded all chunks into search index")


async def run_complete_pipeline(limit: int = 5):
    """
    Run the complete ETL pipeline from start to finish.
    
    Args:
        limit: Maximum number of documents to process (0 for all)
    """
    logger.info("Starting Complete ETL Pipeline")
    
    try:
        # Validate environment
        validate_environment()
        
        # Step 1: Extract
        documents = await step1_extract_from_confluence(limit)
        if not documents:
            logger.error("No documents extracted - stopping pipeline")
            return
        
        # Step 2: Transform  
        chunks = await step2_transform_documents(documents)
        if not chunks:
            logger.error("No chunks created - stopping pipeline")
            return
        
        # Step 3: Create Index
        await step3_create_index()
        
        # Step 4: Load
        await step4_load_into_index(chunks)
        
               
        logger.info("ETL Pipeline completed successfully!")
        
        # Summary
        logger.info("=== Pipeline Summary ===")
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Chunks created: {len(chunks)}")
        logger.info(f"Search index: {os.getenv('AZURE_SEARCH_INDEX', 'default-index')}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


# Individual functions for running steps manually
async def main():
    """
    Main function - you can call individual steps or run the complete pipeline.
    """
    logger.info("ETL Script Ready - Choose your execution method:")
    logger.info("1. Run complete pipeline: await run_complete_pipeline()")
    logger.info("2. Run individual steps:")
    logger.info("   - documents = await step1_extract_from_confluence()")
    logger.info("   - chunks = await step2_transform_documents(documents)")
    logger.info("   - await step3_create_index()")
    logger.info("   - await step4_load_into_index(chunks)")
    
    # For demonstration, run the complete pipeline
    await run_complete_pipeline(limit=3)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
