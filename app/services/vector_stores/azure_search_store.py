"""Azure AI Search vector store implementation."""

import json
import logging
import os
from typing import Dict, List, Optional

import aiohttp

from .base_vector_store_provider import BaseVectorStore
from ...models.chunk import DocumentChunk
from ...services.embeddings.base_embedding_provider import BaseEmbeddingProvider
from ...services.embeddings.azure_openai import azure_embedding_provider

logger = logging.getLogger(__name__)


class AzureSearchVectorStore(BaseVectorStore):
    """
    Azure AI Search implementation of vector store.
    
    Uses Azure AI Search REST API for vector operations including:
    - Document indexing with vector embeddings
    - Vector similarity search
    - Index management
    
    Environment Variables:
        AZURE_SEARCH_ENDPOINT: The Azure Search service endpoint
        AZURE_SEARCH_KEY: The Azure Search admin API key
        AZURE_SEARCH_INDEX: Optional index name (defaults to "embeddings-index")
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        api_version: str = "2023-11-01"
    ):
        """
        Initialize Azure Search vector store.
        
        Args:
            endpoint: Azure Search service endpoint
            api_key: Azure Search admin API key
            index_name: Name of the search index
            embedding_provider: Embedding provider for generating vectors
            api_version: Azure Search API version (use 2023-11-01 or later for vector search)
        """
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_KEY")
        self.index_name = index_name or os.getenv("AZURE_SEARCH_INDEX", "embeddings-index")
        self.api_version = api_version
        
        # Use provided embedding provider or default to Azure
        self.embedding_provider = embedding_provider or azure_embedding_provider
        
        # Vector search configuration
        self.vector_search_kind = "vector"  # Default vector search kind for Azure AI Search
        
        # Validate configuration
        if not self.endpoint:
            logger.warning("Azure Search endpoint not configured")
        if not self.api_key:
            logger.warning("Azure Search API key not configured")
        
        # Remove trailing slash from endpoint
        if self.endpoint and self.endpoint.endswith("/"):
            self.endpoint = self.endpoint[:-1]
    
    def is_available(self) -> bool:
        """Check if Azure Search is available and properly configured."""
        return bool(self.endpoint and self.api_key)
    
    def get_default_index_name(self) -> str:
        """Get the default index name."""
        return self.index_name
    
    async def _make_request(
        self,
        method: str,
        url_path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make an authenticated request to Azure Search API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url_path: URL path relative to the endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response JSON data
            
        Raises:
            Exception: If request fails
        """
        if not self.is_available():
            raise Exception("Azure Search not properly configured")
        
        url = f"{self.endpoint}{url_path}"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Add API version to params
        if not params:
            params = {}
        params["api-version"] = self.api_version
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if data else None,
                    params=params
                ) as response:
                    response_text = await response.text()
                    
                    # Handle successful responses
                    if response.status < 400:
                        try:
                            return json.loads(response_text) if response_text else {}
                        except json.JSONDecodeError:
                            return {}
                    
                    # Handle errors
                    error_msg = f"Azure Search API error {response.status}: {response_text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except aiohttp.ClientError as e:
            error_msg = f"Azure Search API client error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    async def create_index(self, index_name: Optional[str] = None) -> None:
        """
        Create index if it doesn't exist.
        
        Args:
            index_name: Name of index to create, defaults to configured index
        """
        target_index = index_name or self.index_name
        
        try:
            # Check if index exists
            await self._make_request("GET", f"/indexes('{target_index}')")
            logger.info(f"Index '{target_index}' already exists")
            return
        except Exception:
            # Index doesn't exist, create it
            logger.info(f"Creating index '{target_index}'")
            
        # Get embedding dimension from provider
        embedding_dim = 1536  # Default for OpenAI models
        if hasattr(self.embedding_provider, 'get_embedding_dimension'):
            embedding_dim = self.embedding_provider.get_embedding_dimension()
        
        # Define index schema
        index_schema = {
            "name": target_index,
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": False,
                    "facetable": False
                },
                {
                    "name": "doc_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "facetable": True
                },
                {
                    "name": "chunk_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "facetable": False
                },
                {
                    "name": "text",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "facetable": False
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "facetable": True
                },
                {
                    "name": "section_path",
                    "type": "Collection(Edm.String)",
                    "searchable": True,
                    "filterable": True,
                    "facetable": False
                },
                {
                    "name": "page_no",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": True,
                    "facetable": True
                },
                {
                    "name": "embedding",
                    "type": "Collection(Edm.Single)",
                    "dimensions": embedding_dim,
                    "vectorSearchProfile": "default-vector-profile",
                    "searchable": True,
                    "filterable": False,
                    "facetable": False
                }
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "default-hnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "metric": "cosine",
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500
                        }
                    }
                ],
                "profiles": [
                    {
                        "name": "default-vector-profile",
                        "algorithm": "default-hnsw"
                    }
                ]
            }
        }
        
        await self._make_request("POST", "/indexes", index_schema)
        logger.info(f"Successfully created index '{target_index}'")
    
    async def add_documents(self, documents: List[DocumentChunk]) -> None:
        """Add or update document chunks in Azure Search."""
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to Azure Search index '{self.index_name}'")
        
        # Ensure index exists
        await self.create_index()
        
        # Prepare documents for indexing
        search_documents = []
        for doc in documents:
            # Generate embedding if embedding provider is available
            embedding = None
            if self.embedding_provider and self.embedding_provider.is_available():
                try:
                    embedding = await self.embedding_provider.create_embedding(doc.text, "add")
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for document {doc.chunk_id}: {str(e)}")
            
            # Create search document
            search_doc = {
                "@search.action": "upload",  # Upsert behavior
                "id": f"{doc.doc_id}_{doc.chunk_id}",
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "text": doc.text,
                "title": doc.title,
                "section_path": doc.section_path or [],
                "page_no": doc.page_no
            }
            
            # Add embedding if available
            if embedding:
                search_doc["embedding"] = embedding
            
            search_documents.append(search_doc)
        
        # Upload documents in batches (Azure Search has a limit)
        batch_size = 100
        for i in range(0, len(search_documents), batch_size):
            batch = search_documents[i:i + batch_size]
            
            upload_data = {"value": batch}
            await self._make_request("POST", f"/indexes('{self.index_name}')/docs/index", upload_data)
            
            logger.info(f"Uploaded batch {i // batch_size + 1} of {(len(search_documents) + batch_size - 1) // batch_size}")
        
        logger.info(f"Successfully added {len(documents)} documents to index '{self.index_name}'")
    
    async def query_by_vector(self, query_vector: List[float], top_k: int = 10) -> List[DocumentChunk]:
        """Search for similar documents using vector similarity."""
        logger.info(f"Searching for {top_k} similar documents in index '{self.index_name}'")
        
        search_request = {
            "count": True,
            "top": top_k,
            "vectorQueries": [
                {
                    "kind": self.vector_search_kind,  # Required: specify vector search kind
                    "vector": query_vector,
                    "k": top_k,
                    "fields": "embedding"
                }
            ],
            "select": "id,doc_id,chunk_id,text,title,section_path,page_no"
        }
        
        try:
            response = await self._make_request("POST", f"/indexes('{self.index_name}')/docs/search", search_request)
            
            # Convert search results back to DocumentChunk objects
            results = []
            for result in response.get("value", []):
                chunk = DocumentChunk(
                    doc_id=result.get("doc_id", ""),
                    chunk_id=result.get("chunk_id", ""),
                    text=result.get("text", ""),
                    title=result.get("title"),
                    section_path=result.get("section_path"),
                    page_no=result.get("page_no"),
                    meta={
                        "search_score": result.get("@search.score", 0.0),
                        "search_id": result.get("id", "")
                    }
                )
                results.append(chunk)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
    
    # Currently not needed
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        pass
    # Index management methods
    
    async def list_indexes(self) -> List[str]:
        """List all indexes in the Azure Search service."""
        logger.info("Listing all indexes in Azure Search service")
        
        try:
            response = await self._make_request("GET", "/indexes")
            indexes = [index["name"] for index in response.get("value", [])]
            
            logger.info(f"Found {len(indexes)} indexes")
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to list indexes: {str(e)}")
            raise
        
# Currently not needed
    
    async def delete_index(self, index_name: Optional[str] = None) -> None:
        pass

# Global instance for easy import
azure_search_vector_store = AzureSearchVectorStore()