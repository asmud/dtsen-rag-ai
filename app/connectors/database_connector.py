import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import asyncpg

from connectors.base_connector import BaseConnector
from models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class DatabaseConnector(BaseConnector):
    """Connector for RDBMS data sources"""
    
    def __init__(self):
        super().__init__()
        self.connection_pool = None
        self.db_url = self.settings.DB_QUERY_URL
        self.timeout = self.settings.DB_QUERY_TIMEOUT
        self.max_rows = self.settings.DB_QUERY_MAX_ROWS
        self._connected = False
    
    async def connect(self) -> bool:
        """Create database connection pool"""
        if not self.settings.DB_QUERY_ENABLED:
            logger.info("Database query connector is disabled")
            return False
        
        if not self.db_url:
            logger.error("Database URL not configured")
            return False
        
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                timeout=self.timeout,
                command_timeout=self.timeout
            )
            self._connected = True
            logger.info("Connected to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close database connection pool"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
            self._connected = False
            logger.info("Disconnected from database")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch data from database queries"""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return []
        
        queries = kwargs.get('queries', [])
        if not queries:
            logger.warning("No database queries provided")
            return []
        
        documents = []
        
        for query_config in queries:
            try:
                docs = await self._execute_query(query_config)
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error executing query: {e}")
        
        logger.info(f"Fetched {len(documents)} documents from database")
        return documents
    
    async def _execute_query(self, query_config: Dict[str, Any]) -> List[Document]:
        """Execute a database query and return documents"""
        query = query_config.get('query')
        if not query:
            logger.error("Query configuration missing SQL query")
            return []
        
        params = query_config.get('params', [])
        doc_id_column = query_config.get('id_column', 'id')
        content_columns = query_config.get('content_columns', ['content', 'text', 'body'])
        title_column = query_config.get('title_column', 'title')
        metadata_columns = query_config.get('metadata_columns', [])
        
        documents = []
        
        try:
            async with self.connection_pool.acquire() as connection:
                # Execute query with limit
                limited_query = f"SELECT * FROM ({query}) AS subquery LIMIT {self.max_rows}"
                rows = await connection.fetch(limited_query, *params)
                
                for row in rows:
                    doc = await self._create_document_from_row(
                        dict(row), query_config, doc_id_column, 
                        content_columns, title_column, metadata_columns
                    )
                    if doc:
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error executing database query: {e}")
            return []
    
    async def _create_document_from_row(
        self, 
        row: Dict[str, Any], 
        query_config: Dict[str, Any],
        doc_id_column: str,
        content_columns: List[str],
        title_column: str,
        metadata_columns: List[str]
    ) -> Optional[Document]:
        """Create document from database row"""
        try:
            # Extract document ID
            doc_id = str(row.get(doc_id_column, ''))
            if not doc_id:
                doc_id = str(hash(json.dumps(row, default=str, sort_keys=True)))
            
            # Extract content
            content = ""
            content_source = ""
            for col in content_columns:
                if col in row and row[col]:
                    content = str(row[col])
                    content_source = col
                    break
            
            if not content:
                # Use all row data as content if no specific content column
                content = json.dumps(row, default=str, indent=2)
                content_source = "full_row"
            
            # Extract title
            title = ""
            if title_column in row and row[title_column]:
                title = str(row[title_column])
            
            # Extract metadata
            extra_metadata = {
                'query_name': query_config.get('name', 'unnamed_query'),
                'content_source_column': content_source,
                'row_data': row
            }
            
            # Add specified metadata columns
            for col in metadata_columns:
                if col in row:
                    extra_metadata[f"meta_{col}"] = row[col]
            
            # Create metadata
            metadata = DocumentMetadata(
                source="database",
                title=title,
                database_table=query_config.get('table', 'unknown'),
                database_query=query_config.get('query', ''),
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata=extra_metadata
            )
            
            # Create document
            document = Document(
                id=doc_id,
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error creating document from database row: {e}")
            return None
    
    async def execute_custom_query(self, query: str, params: List = None) -> List[Dict[str, Any]]:
        """Execute a custom query and return raw results"""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return []
        
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query, *(params or []))
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            return []
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a database table"""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return {}
        
        try:
            async with self.connection_pool.acquire() as connection:
                # Get column information
                columns_query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = $1
                ORDER BY ordinal_position
                """
                columns = await connection.fetch(columns_query, table_name)
                
                # Get row count
                count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
                count_result = await connection.fetchrow(count_query)
                
                return {
                    "table_name": table_name,
                    "columns": [dict(col) for col in columns],
                    "row_count": count_result['row_count'] if count_result else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of database connector"""
        try:
            if not self.settings.DB_QUERY_ENABLED:
                return {
                    "status": "disabled",
                    "message": "Database query connector is disabled"
                }
            
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "message": "Database not connected"
                }
            
            # Test database connection
            async with self.connection_pool.acquire() as connection:
                result = await connection.fetchrow("SELECT 1 as test")
                
                if result and result['test'] == 1:
                    return {
                        "status": "healthy",
                        "connected": True,
                        "pool_size": self.connection_pool.get_size(),
                        "timeout": self.timeout,
                        "max_rows": self.max_rows,
                        "test_query": "successful"
                    }
                else:
                    return {
                        "status": "degraded",
                        "message": "Test query failed",
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
            "name": "DatabaseConnector",
            "type": "database",
            "status": "connected" if self._connected else "disconnected",
            "enabled": self.settings.DB_QUERY_ENABLED,
            "timeout": self.timeout,
            "max_rows": self.max_rows,
            "pool_size": self.connection_pool.get_size() if self.connection_pool else 0
        }