import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import httpx

from connectors.base_connector import BaseConnector
from models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class APIConnector(BaseConnector):
    """Connector for REST API data sources"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.timeout = self.settings.API_REQUEST_TIMEOUT
        self.max_retries = self.settings.API_MAX_RETRIES
        self.retry_delay = self.settings.API_RETRY_DELAY
        self._connected = False
    
    async def connect(self) -> bool:
        """Initialize HTTP client"""
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
            self._connected = True
            logger.info("Connected to API client")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect API client: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close HTTP client"""
        try:
            if self.client:
                await self.client.aclose()
            self._connected = False
            logger.info("Disconnected API client")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting API client: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch data from API endpoints"""
        if not self._connected:
            await self.connect()
        
        api_configs = kwargs.get('api_configs', [])
        if not api_configs:
            logger.warning("No API configurations provided")
            return []
        
        documents = []
        
        for config in api_configs:
            try:
                docs = await self._fetch_from_endpoint(config)
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error fetching from API {config.get('url', 'unknown')}: {e}")
        
        logger.info(f"Fetched {len(documents)} documents from API sources")
        return documents
    
    async def _fetch_from_endpoint(self, config: Dict[str, Any]) -> List[Document]:
        """Fetch data from a single API endpoint"""
        url = config.get('url')
        if not url:
            logger.error("API configuration missing URL")
            return []
        
        method = config.get('method', 'GET').upper()
        headers = config.get('headers', {})
        params = config.get('params', {})
        data = config.get('data', None)
        auth = config.get('auth', None)
        response_path = config.get('response_path', None)  # JSON path to extract data
        
        documents = []
        
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    # Prepare authentication
                    auth_obj = None
                    if auth:
                        if auth.get('type') == 'basic':
                            auth_obj = httpx.BasicAuth(auth['username'], auth['password'])
                        elif auth.get('type') == 'bearer':
                            headers['Authorization'] = f"Bearer {auth['token']}"
                        elif auth.get('type') == 'api_key':
                            if auth.get('location') == 'header':
                                headers[auth['key']] = auth['value']
                            elif auth.get('location') == 'query':
                                params[auth['key']] = auth['value']
                    
                    # Make the request
                    response = await self.client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        json=data if method in ['POST', 'PUT', 'PATCH'] else None,
                        auth=auth_obj
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    if response.headers.get('content-type', '').startswith('application/json'):
                        data = response.json()
                        documents = await self._parse_json_response(data, config, url)
                    else:
                        # Treat as text content
                        content = response.text
                        doc = await self._create_document_from_text(content, config, url)
                        if doc:
                            documents = [doc]
                    
                    break  # Success, exit retry loop
                    
                except httpx.HTTPStatusError as e:
                    if attempt < self.max_retries:
                        logger.warning(f"HTTP error {e.response.status_code} for {url}, retrying...")
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
                        break
                        
                except httpx.RequestError as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Request error for {url}, retrying...")
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"Request error for {url}: {e}")
                        break
            
            return documents
            
        except Exception as e:
            logger.error(f"Unexpected error fetching from {url}: {e}")
            return []
    
    async def _parse_json_response(self, data: Union[Dict, List], config: Dict[str, Any], url: str) -> List[Document]:
        """Parse JSON response and create documents"""
        documents = []
        response_path = config.get('response_path')
        
        try:
            # Extract data using JSON path if specified
            if response_path:
                data = self._extract_json_path(data, response_path)
            
            # Handle different data structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    doc = await self._create_document_from_json(item, config, url, i)
                    if doc:
                        documents.append(doc)
            elif isinstance(data, dict):
                doc = await self._create_document_from_json(data, config, url)
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing JSON response from {url}: {e}")
            return []
    
    def _extract_json_path(self, data: Union[Dict, List], path: str) -> Union[Dict, List]:
        """Extract data using simple JSON path (e.g., 'data.items', 'results')"""
        try:
            parts = path.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    logger.warning(f"JSON path '{path}' not found in response")
                    return data
            
            return current
            
        except Exception as e:
            logger.error(f"Error extracting JSON path '{path}': {e}")
            return data
    
    async def _create_document_from_json(self, item: Dict[str, Any], config: Dict[str, Any], url: str, index: int = 0) -> Optional[Document]:
        """Create document from JSON item"""
        try:
            # Extract content based on configuration
            content_fields = config.get('content_fields', ['content', 'text', 'body', 'description'])
            title_fields = config.get('title_fields', ['title', 'name', 'subject'])
            id_fields = config.get('id_fields', ['id', 'uuid', 'key'])
            
            # Find content
            content = ""
            for field in content_fields:
                if field in item and item[field]:
                    content = str(item[field])
                    break
            
            if not content:
                # Use entire item as content if no specific field found
                content = json.dumps(item, indent=2)
            
            # Find title
            title = ""
            for field in title_fields:
                if field in item and item[field]:
                    title = str(item[field])
                    break
            
            # Find ID
            doc_id = ""
            for field in id_fields:
                if field in item and item[field]:
                    doc_id = str(item[field])
                    break
            
            if not doc_id:
                doc_id = str(hash(f"{url}_{index}_{content[:100]}"))
            
            # Create metadata
            metadata = DocumentMetadata(
                source="api",
                url=url,
                title=title,
                api_endpoint=url,
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'api_response_index': index,
                    'original_data': item,
                    'content_field_used': next((f for f in content_fields if f in item), 'json_dump')
                }
            )
            
            # Create document
            document = Document(
                id=doc_id,
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from JSON item: {e}")
            return None
    
    async def _create_document_from_text(self, content: str, config: Dict[str, Any], url: str) -> Optional[Document]:
        """Create document from text content"""
        try:
            if not content or len(content.strip()) < 10:
                return None
            
            # Create metadata
            metadata = DocumentMetadata(
                source="api",
                url=url,
                api_endpoint=url,
                file_type='.txt',
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'content_type': 'text',
                    'content_length': len(content)
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"{url}_{content[:100]}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from text: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of API connector"""
        try:
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "message": "API client not connected"
                }
            
            # Test with a simple HTTP request
            test_url = "https://httpbin.org/json"
            response = await self.client.get(test_url, timeout=10)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "client_connected": True,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                    "test_request": "successful"
                }
            else:
                return {
                    "status": "degraded",
                    "message": f"Test request failed with status {response.status_code}",
                    "client_connected": True
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Return connector metadata"""
        return {
            "name": "APIConnector",
            "type": "rest_api",
            "status": "connected" if self._connected else "disconnected",
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        }