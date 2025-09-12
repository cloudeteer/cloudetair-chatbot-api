"""
Base Data Loader abstract class.

This module defines the common interface that all data loader implementations must implement,
following the same pattern as the LLM and embedding providers in the system.
"""

from abc import ABC, abstractmethod
from typing import List
import logging

from app.models.document import Document

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    This class defines the interface that all concrete data loader implementations
    must follow. Data loaders are responsible for fetching documents from various
    sources (Confluence, Blob Storage, SharePoint, file systems, etc.) and returning them in
    a standardized Document format.
    
    The design follows the same pattern as other service providers in the system
    (LLM providers, embedding providers) to ensure consistency and modularity.
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the data loader is properly configured and available.
        
        This method should verify that all necessary configuration (API keys,
        endpoints, credentials) is present and that the service is accessible.
        
        Returns:
            bool: True if the loader is available and ready to use, False otherwise.
        """
        pass
    
    @abstractmethod
    async def load_data(self) -> List[Document]:
        """
        Load documents from the data source.
        
        This method should handle the actual data retrieval from the source,
        including authentication, pagination, error handling, and conversion
        to the standardized Document format.
        
        Returns:
            List[Document]: List of documents loaded from the source.
            
        Raises:
            Exception: Should raise appropriate exceptions for authentication
                      failures, network errors, or other issues.
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """
        Get a human-readable name for this data source.
        
        Returns:
            str: Name of the data source (e.g., "Confluence", "SharePoint")
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the data loader."""
        return f"{self.__class__.__name__}(source='{self.get_source_name()}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the data loader."""
        return f"{self.__class__.__name__}(source='{self.get_source_name()}', available={self.is_available()})"