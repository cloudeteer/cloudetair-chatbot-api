"""
Confluence Data Loader implementation.

This module provides a data loader for Atlassian Confluence that implements
the BaseDataLoader interface. It fetches pages from a Confluence space and
returns them as standardized Document objects.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from app.services.data_loaders.base_loader import BaseDataLoader
from app.models.document import Document

logger = logging.getLogger(__name__)


class ConfluenceLoader(BaseDataLoader):
    """
    Confluence data loader implementation.
    
    Fetches pages from an Atlassian Confluence space using the REST API
    and returns them as standardized Document objects.
    
    Environment Variables:
        CONFLUENCE_URL: Base URL of the Confluence instance (e.g., https://company.atlassian.net/wiki)
        CONFLUENCE_USERNAME: Username or email for authentication
        CONFLUENCE_API_TOKEN: API token for authentication
        CONFLUENCE_SPACE_KEY: Space key to load documents from (optional, can be set via constructor)
    """
    
    def __init__(self, space_key: Optional[str] = None, base_url: Optional[str] = None, 
                 username: Optional[str] = None, api_token: Optional[str] = None):
        """
        Initialize the Confluence loader.
        
        Args:
            space_key: Confluence space key to load from. If not provided, uses CONFLUENCE_SPACE_KEY env var.
            base_url: Confluence base URL. If not provided, uses CONFLUENCE_URL env var.
            username: Username for authentication. If not provided, uses CONFLUENCE_USERNAME env var.
            api_token: API token for authentication. If not provided, uses CONFLUENCE_API_TOKEN env var.
        """
        self.space_key = space_key or os.getenv('CONFLUENCE_SPACE_KEY', '')
        self.base_url = base_url or os.getenv('CONFLUENCE_URL', '')
        self.username = username or os.getenv('CONFLUENCE_USERNAME', '')
        self.api_token = api_token or os.getenv('CONFLUENCE_API_TOKEN', '')
        
        # Remove trailing slash from base URL
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
            
        # Set up session for reuse
        self.session = requests.Session()
        if self.username and self.api_token:
            self.session.auth = (self.username, self.api_token)
    
    def is_available(self) -> bool:
        """
        Check if the Confluence loader is properly configured and available.
        
        Verifies that required environment variables are set and that we can
        connect to the Confluence instance.
        
        Returns:
            bool: True if the loader is available, False otherwise.
        """
        if not all([self.base_url, self.username, self.api_token]):
            logger.warning("Confluence loader missing required configuration")
            return False
            
        try:
            # Test connection by fetching server info
            response = self.session.get(
                f"{self.base_url}/rest/api/space",
                timeout=10,
                params={'limit': 1}
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Confluence connection test failed: {str(e)}")
            return False
    
    def get_source_name(self) -> str:
        """Get the source name for this loader."""
        return "Confluence"
    
    async def load_data(self) -> List[Document]:
        """
        Load documents from the Confluence space.
        
        Fetches all pages from the specified Confluence space and converts
        them to Document objects.
        
        Returns:
            List[Document]: List of documents loaded from Confluence.
            
        Raises:
            ValueError: If required configuration is missing.
            requests.RequestException: If API requests fail.
        """
        if not self.is_available():
            raise ValueError("Confluence loader is not properly configured")
            
        if not self.space_key:
            raise ValueError("No space key provided. Set CONFLUENCE_SPACE_KEY or pass space_key parameter")
        
        logger.info(f"Loading documents from Confluence space: {self.space_key}")
        
        documents = []
        start = 0
        limit = 50  # Confluence API default limit
        
        try:
            while True:
                # Fetch pages from the space
                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
                    params={
                        'spaceKey': self.space_key,
                        'type': 'page',
                        'status': 'current',
                        'expand': 'body.storage,version,space,history.lastUpdated',
                        'start': start,
                        'limit': limit
                    },
                    timeout=30
                )
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get('results'):
                    break
                
                for page in data['results']:
                    try:
                        document = self._convert_page_to_document(page)
                        documents.append(document)
                        logger.debug(f"Converted page: {document.title}")
                    except Exception as e:
                        logger.warning(f"Failed to convert page {page.get('title', 'Unknown')}: {str(e)}")
                        continue
                
                # Check if there are more pages
                if len(data['results']) < limit:
                    break
                start += limit
                
        except requests.RequestException as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise requests.RequestException(f"Authentication failed: {str(e)}")
            elif "404" in str(e):
                raise requests.RequestException(f"Space '{self.space_key}' not found or not accessible: {str(e)}")
            else:
                raise requests.RequestException(f"Failed to fetch pages from Confluence: {str(e)}")
        
        logger.info(f"Successfully loaded {len(documents)} documents from Confluence space '{self.space_key}'")
        return documents
    
    def _convert_page_to_document(self, page: Dict[str, Any]) -> Document:
        """
        Convert a Confluence page JSON object to a Document.
        
        Args:
            page: Confluence page JSON object from the API.
            
        Returns:
            Document: Standardized document object.
        """
        # Extract basic information
        page_id = page['id']
        title = page['title']
        
        # Extract content from storage format and convert HTML to text
        content_html = page.get('body', {}).get('storage', {}).get('value', '')
        content = self._html_to_text(content_html)
        
        # Extract last modified date
        last_updated = page.get('history', {}).get('lastUpdated', {}).get('when')
        if last_updated:
            last_modified = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
        else:
            last_modified = datetime.now()
        
        # Build the page URL
        base_url_parsed = urlparse(self.base_url)
        page_url = f"{base_url_parsed.scheme}://{base_url_parsed.netloc}/wiki/spaces/{self.space_key}/pages/{page_id}"
        
        # Extract metadata
        metadata = {
            'space_key': self.space_key,
            'space_name': page.get('space', {}).get('name', ''),
            'page_id': page_id,
            'version': page.get('version', {}).get('number', 1),
            'content_type': 'page',
            'source': 'confluence'
        }
        
        return Document(
            id=f"confluence_{self.space_key}_{page_id}",
            title=title,
            content=content,
            last_modified=last_modified,
            link=page_url,
            metadata=metadata
        )
    
    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML content to plain text.
        
        Args:
            html_content: HTML content from Confluence storage format.
            
        Returns:
            str: Plain text content.
        """
        if not html_content:
            return ""
        
        try:
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.warning(f"Failed to parse HTML content: {str(e)}")
            return html_content  # Return raw HTML as fallback
    
    def __del__(self):
        """Clean up the session when the loader is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


# Global instance for easy importing
confluence_loader = ConfluenceLoader()