import asyncio
import json
import logging
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from config import get_settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based cache manager for embeddings and API responses"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.ttl = self.settings.REDIS_CACHE_TTL
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        if not self.settings.ENABLE_CACHING:
            logger.info("Caching is disabled")
            return False
        
        try:
            self.redis_client = redis.from_url(
                self.settings.REDIS_URL,
                max_connections=self.settings.REDIS_MAX_CONNECTIONS,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Redis"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            self._connected = False
            logger.info("Disconnected from Redis cache")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
            return False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a consistent hash from all arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._connected:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self._connected:
            return False
        
        try:
            ttl = ttl or self.ttl
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self._connected:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._connected:
            return False
        
        try:
            return bool(await self.redis_client.exists(key))
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._generate_key("embedding", text=text, model=model)
        result = await self.get(key)
        
        if result and 'embedding' in result:
            logger.debug(f"Cache hit for embedding: {key}")
            return result['embedding']
        
        return None
    
    async def set_embedding(self, text: str, model: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        """Cache embedding"""
        key = self._generate_key("embedding", text=text, model=model)
        value = {
            'embedding': embedding,
            'text_length': len(text),
            'model': model,
            'created_at': datetime.now().isoformat()
        }
        
        success = await self.set(key, value, ttl)
        if success:
            logger.debug(f"Cached embedding: {key}")
        
        return success
    
    async def get_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached API response"""
        key = self._generate_key("api_response", endpoint=endpoint, **params)
        result = await self.get(key)
        
        if result:
            logger.debug(f"Cache hit for API response: {key}")
            return result.get('response')
        
        return None
    
    async def set_api_response(self, endpoint: str, params: Dict[str, Any], response: Any, ttl: Optional[int] = None) -> bool:
        """Cache API response"""
        key = self._generate_key("api_response", endpoint=endpoint, **params)
        value = {
            'response': response,
            'endpoint': endpoint,
            'params': params,
            'created_at': datetime.now().isoformat()
        }
        
        success = await self.set(key, value, ttl)
        if success:
            logger.debug(f"Cached API response: {key}")
        
        return success
    
    async def get_document_chunks(self, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached document chunks"""
        key = self._generate_key("document_chunks", document_id=document_id)
        result = await self.get(key)
        
        if result and 'chunks' in result:
            logger.debug(f"Cache hit for document chunks: {key}")
            return result['chunks'] 
        
        return None
    
    async def set_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """Cache document chunks"""
        key = self._generate_key("document_chunks", document_id=document_id)
        value = {
            'chunks': chunks,
            'document_id': document_id,
            'chunk_count': len(chunks),
            'created_at': datetime.now().isoformat()
        }
        
        success = await self.set(key, value, ttl)
        if success:
            logger.debug(f"Cached document chunks: {key}")
        
        return success
    
    async def get_query_results(self, query: str, filters: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached query results"""
        key = self._generate_key("query_results", query=query, filters=filters or {})
        result = await self.get(key)
        
        if result:
            logger.debug(f"Cache hit for query results: {key}")
            return result.get('results')
        
        return None
    
    async def set_query_results(self, query: str, results: Any, filters: Dict[str, Any] = None, ttl: Optional[int] = None) -> bool:
        """Cache query results"""
        key = self._generate_key("query_results", query=query, filters=filters or {})
        value = {
            'results': results,
            'query': query,
            'filters': filters,
            'created_at': datetime.now().isoformat()
        }
        
        # Use shorter TTL for query results (they may become stale quickly)
        ttl = ttl or min(self.ttl, 1800)  # Max 30 minutes
        
        success = await self.set(key, value, ttl)
        if success:
            logger.debug(f"Cached query results: {key}")
        
        return success
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern"""
        if not self._connected:
            return 0
        
        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all cache"""
        if not self._connected:
            return False
        
        try:
            await self.redis_client.flushdb()
            logger.info("Cleared all cache")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._connected:
            return {"status": "disconnected"}
        
        try:
            info = await self.redis_client.info()
            
            # Get key counts by prefix
            key_counts = {}
            prefixes = ["embedding", "api_response", "document_chunks", "query_results"]
            
            for prefix in prefixes:
                count = 0
                async for _ in self.redis_client.scan_iter(match=f"{prefix}:*"):
                    count += 1
                key_counts[prefix] = count
            
            return {
                "status": "connected",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "key_counts": key_counts,
                "ttl_seconds": self.ttl,
                "max_connections": self.settings.REDIS_MAX_CONNECTIONS
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "message": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        try:
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "message": "Not connected to Redis"
                }
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": datetime.now().isoformat()}
            
            # Test set
            await self.set(test_key, test_value, 60)
            
            # Test get
            retrieved = await self.get(test_key)
            
            # Test delete
            await self.delete(test_key)
            
            if retrieved and retrieved.get("test") is True:
                return {
                    "status": "healthy",
                    "connected": True,
                    "operations": "all_working",
                    "ttl": self.ttl
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Cache operations not working properly"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }