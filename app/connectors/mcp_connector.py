import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import httpx

from connectors.base_connector import BaseConnector
from models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class MCPConnector(BaseConnector):
    """Connector for Model Context Protocol (MCP) servers"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.server_url = self.settings.MCP_SERVER_URL
        self.api_key = self.settings.MCP_API_KEY
        self.timeout = 30
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        if not self.settings.MCP_ENABLED:
            logger.info("MCP connector is disabled")
            return False
        
        if not self.server_url:
            logger.error("MCP server URL not configured")
            return False
        
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.client = httpx.AsyncClient(
                base_url=self.server_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout)
            )
            
            # Test connection with ping/health endpoint (use /posts/1 for JSONPlaceholder)
            health_endpoint = '/posts/1' if 'jsonplaceholder' in self.server_url else '/health'
            response = await self.client.get(health_endpoint)
            if response.status_code == 200:
                self._connected = True
                logger.info(f"Connected to MCP server: {self.server_url}")
                return True
            else:
                logger.error(f"MCP server health check failed: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from MCP server"""
        try:
            if self.client:
                await self.client.aclose()
            self._connected = False
            logger.info("Disconnected from MCP server")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch data via MCP protocol"""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return []
        
        mcp_requests = kwargs.get('mcp_requests', [])
        if not mcp_requests:
            logger.warning("No MCP requests provided")
            return []
        
        documents = []
        
        for request_config in mcp_requests:
            try:
                docs = await self._execute_mcp_request(request_config)
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error executing MCP request: {e}")
        
        logger.info(f"Fetched {len(documents)} documents via MCP")
        return documents
    
    async def _execute_mcp_request(self, request_config: Dict[str, Any]) -> List[Document]:
        """Execute an MCP request and return documents"""
        method = request_config.get('method', 'GET')
        endpoint = request_config.get('endpoint', '/')
        params = request_config.get('params', {})
        data = request_config.get('data', {})
        
        documents = []
        
        try:
            if method.upper() == 'GET':
                response = await self.client.get(endpoint, params=params)
            elif method.upper() == 'POST':
                response = await self.client.post(endpoint, json=data, params=params)
            else:
                response = await self.client.request(method, endpoint, json=data, params=params)
            
            response.raise_for_status()
            
            # Parse MCP response
            if response.headers.get('content-type', '').startswith('application/json'):
                response_data = response.json()
                documents = await self._parse_mcp_response(response_data, request_config, endpoint)
            else:
                # Handle text responses
                content = response.text
                doc = await self._create_document_from_mcp_text(content, request_config, endpoint)
                if doc:
                    documents = [doc]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error executing MCP request to {endpoint}: {e}")
            return []
    
    async def _parse_mcp_response(self, data: Dict[str, Any], request_config: Dict[str, Any], endpoint: str) -> List[Document]:
        """Parse MCP response and create documents"""
        documents = []
        
        try:
            # Handle different MCP response formats
            if 'resources' in data:
                # Standard MCP resources format
                resources = data['resources']
                for resource in resources:
                    doc = await self._create_document_from_mcp_resource(resource, request_config, endpoint)
                    if doc:
                        documents.append(doc)
            
            elif 'tools' in data:
                # MCP tools format
                tools = data['tools']
                for tool in tools:
                    doc = await self._create_document_from_mcp_tool(tool, request_config, endpoint)
                    if doc:
                        documents.append(doc)
            
            elif 'prompts' in data:
                # MCP prompts format
                prompts = data['prompts']
                for prompt in prompts:
                    doc = await self._create_document_from_mcp_prompt(prompt, request_config, endpoint)
                    if doc:
                        documents.append(doc)
            
            else:
                # Generic data format
                doc = await self._create_document_from_mcp_data(data, request_config, endpoint)
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing MCP response from {endpoint}: {e}")
            return []
    
    async def _create_document_from_mcp_resource(self, resource: Dict[str, Any], request_config: Dict[str, Any], endpoint: str) -> Optional[Document]:
        """Create document from MCP resource"""
        try:
            resource_uri = resource.get('uri', '')
            resource_name = resource.get('name', '')
            resource_description = resource.get('description', '')
            resource_mimeType = resource.get('mimeType', 'text/plain')
            
            # Get resource content
            content = ""
            if 'text' in resource:
                content = resource['text']
            elif 'blob' in resource:
                # Handle binary data (base64 encoded)
                content = f"[Binary data: {resource_mimeType}]"
            else:
                content = json.dumps(resource, indent=2)
            
            # Create metadata
            metadata = DocumentMetadata(
                source="mcp",
                title=resource_name or resource_uri,
                url=resource_uri,
                mcp_endpoint=endpoint,
                file_type=resource_mimeType,
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'mcp_resource_type': 'resource',
                    'mcp_uri': resource_uri,
                    'mcp_description': resource_description,
                    'mcp_mimetype': resource_mimeType,
                    'request_config': request_config
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"mcp_resource_{resource_uri}_{endpoint}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from MCP resource: {e}")
            return None
    
    async def _create_document_from_mcp_tool(self, tool: Dict[str, Any], request_config: Dict[str, Any], endpoint: str) -> Optional[Document]:
        """Create document from MCP tool"""
        try:
            tool_name = tool.get('name', '')
            tool_description = tool.get('description', '')
            tool_schema = tool.get('inputSchema', {})
            
            # Create content from tool definition
            content = f"Tool: {tool_name}\n"
            if tool_description:
                content += f"Description: {tool_description}\n"
            
            content += f"\nSchema:\n{json.dumps(tool_schema, indent=2)}"
            
            # Create metadata
            metadata = DocumentMetadata(
                source="mcp",
                title=f"MCP Tool: {tool_name}",
                mcp_endpoint=endpoint,
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'mcp_resource_type': 'tool',
                    'mcp_tool_name': tool_name,
                    'mcp_tool_description': tool_description,
                    'mcp_tool_schema': tool_schema,
                    'request_config': request_config
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"mcp_tool_{tool_name}_{endpoint}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from MCP tool: {e}")
            return None
    
    async def _create_document_from_mcp_prompt(self, prompt: Dict[str, Any], request_config: Dict[str, Any], endpoint: str) -> Optional[Document]:
        """Create document from MCP prompt"""
        try:
            prompt_name = prompt.get('name', '')
            prompt_description = prompt.get('description', '')
            prompt_messages = prompt.get('messages', [])
            
            # Create content from prompt
            content = f"Prompt: {prompt_name}\n"
            if prompt_description:
                content += f"Description: {prompt_description}\n"
            
            content += "\nMessages:\n"
            for msg in prompt_messages:
                content += f"- {msg.get('role', 'user')}: {msg.get('content', {}).get('text', '')}\n"
            
            # Create metadata
            metadata = DocumentMetadata(
                source="mcp",
                title=f"MCP Prompt: {prompt_name}",
                mcp_endpoint=endpoint,
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'mcp_resource_type': 'prompt',
                    'mcp_prompt_name': prompt_name,
                    'mcp_prompt_description': prompt_description,
                    'mcp_messages_count': len(prompt_messages),
                    'request_config': request_config
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"mcp_prompt_{prompt_name}_{endpoint}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from MCP prompt: {e}")
            return None
    
    async def _create_document_from_mcp_data(self, data: Dict[str, Any], request_config: Dict[str, Any], endpoint: str) -> Optional[Document]:
        """Create document from generic MCP data"""
        try:
            content = json.dumps(data, indent=2)
            
            # Create metadata
            metadata = DocumentMetadata(
                source="mcp",
                title=f"MCP Data from {endpoint}",
                mcp_endpoint=endpoint,
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'mcp_resource_type': 'generic_data',
                    'mcp_data_keys': list(data.keys()),
                    'request_config': request_config
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"mcp_data_{endpoint}_{datetime.now().isoformat()}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from MCP data: {e}")
            return None
    
    async def _create_document_from_mcp_text(self, content: str, request_config: Dict[str, Any], endpoint: str) -> Optional[Document]:
        """Create document from MCP text response"""
        try:
            if not content or len(content.strip()) < 10:
                return None
            
            # Create metadata
            metadata = DocumentMetadata(
                source="mcp",
                title=f"MCP Text from {endpoint}",
                mcp_endpoint=endpoint,
                file_type='.txt',
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'mcp_resource_type': 'text',
                    'content_length': len(content),
                    'request_config': request_config
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"mcp_text_{endpoint}_{content[:100]}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from MCP text: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of MCP connector"""
        try:
            if not self.settings.MCP_ENABLED:
                return {
                    "status": "disabled",
                    "message": "MCP connector is disabled"
                }
            
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "message": "MCP server not connected"
                }
            
            # Test MCP server health (use /posts/1 for JSONPlaceholder) 
            health_endpoint = '/posts/1' if 'jsonplaceholder' in self.server_url else '/health'
            response = await self.client.get(health_endpoint)
            
            if response.status_code == 200:
                return {
                    "status": "healthy", 
                    "server_url": self.server_url,
                    "connected": True,
                    "timeout": self.timeout,
                    "api_key_configured": bool(self.api_key),
                    "test_request": "successful"
                }
            else:
                return {
                    "status": "degraded",
                    "message": f"Health check returned status {response.status_code}",
                    "server_url": self.server_url,
                    "connected": True
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Return connector metadata"""
        return {
            "name": "MCPConnector",
            "type": "mcp",
            "status": "connected" if self._connected else "disconnected",
            "enabled": self.settings.MCP_ENABLED,
            "server_url": self.server_url,
            "api_key_configured": bool(self.api_key),
            "timeout": self.timeout
        }