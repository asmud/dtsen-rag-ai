import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
import asyncpg

from config import get_settings

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checking for all system components"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def check_all_systems(self) -> Dict[str, Any]:
        """Check health of all system components"""
        checks = await asyncio.gather(
            self.check_database(),
            self.check_ollama(),
            self.check_redis(),
            self.check_vector_store(),
            self.check_connectors(),
            return_exceptions=True
        )
        
        database_health, ollama_health, redis_health, vector_health, connectors_health = checks
        
        # Calculate overall status
        all_results = [database_health, ollama_health, redis_health, vector_health, connectors_health]
        statuses = [result.get('status', 'unhealthy') if isinstance(result, dict) else 'error' for result in all_results]
        
        if all(status == 'healthy' for status in statuses):
            overall_status = 'healthy'
        elif any(status == 'healthy' for status in statuses):
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'database': database_health if isinstance(database_health, dict) else {'status': 'error', 'message': str(database_health)},
                'ollama': ollama_health if isinstance(ollama_health, dict) else {'status': 'error', 'message': str(ollama_health)},
                'redis': redis_health if isinstance(redis_health, dict) else {'status': 'error', 'message': str(redis_health)},
                'vector_store': vector_health if isinstance(vector_health, dict) else {'status': 'error', 'message': str(vector_health)},
                'connectors': connectors_health if isinstance(connectors_health, dict) else {'status': 'error', 'message': str(connectors_health)}
            }
        }
    
    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database health"""
        try:
            connection = await asyncpg.connect(self.settings.DATABASE_URL)
            
            # Combined database health check query
            result = await connection.fetchrow("""
                SELECT 
                    1 as test,
                    version() as version,
                    EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as has_vector,
                    pg_database_size(current_database()) as db_size,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections
            """)
            
            await connection.close()
            
            return {
                'status': 'healthy',
                'version': result['version'].split(' ')[0] if result['version'] else 'unknown',
                'pgvector_installed': result['has_vector'],
                'database_size_bytes': result['db_size'],
                'active_connections': result['active_connections'],
                'response_time_ms': 0  # Could add timing
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    async def check_ollama(self) -> Dict[str, Any]:
        """Check Ollama service health"""
        try:
            async with httpx.AsyncClient() as client:
                # Check if Ollama is running
                response = await client.get(
                    f"{self.settings.OLLAMA_API}/api/tags",
                    timeout=10
                )
                
                if response.status_code == 200:
                    models = response.json()
                    
                    # Check if required models are available
                    model_names = [model['name'] for model in models.get('models', [])]
                    primary_model_available = self.settings.LLM_MODEL in model_names
                    fallback_model_available = self.settings.LLM_FALLBACK_MODEL in model_names
                    
                    # Test model inference
                    test_response = await self._test_ollama_inference(client)
                    
                    return {
                        'status': 'healthy' if primary_model_available else 'degraded',
                        'available_models': model_names,
                        'primary_model_available': primary_model_available,
                        'fallback_model_available': fallback_model_available,
                        'inference_test': test_response,
                        'api_url': self.settings.OLLAMA_API
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': f"Ollama API returned status {response.status_code}",
                        'api_url': self.settings.OLLAMA_API
                    }
                    
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e),
                'api_url': self.settings.OLLAMA_API
            }
    
    async def _test_ollama_inference(self, client: httpx.AsyncClient) -> Dict[str, Any]:
        """Test Ollama model inference"""
        try:
            response = await client.post(
                f"{self.settings.OLLAMA_API}/api/generate",
                json={
                    'model': self.settings.LLM_MODEL,
                    'prompt': 'Hello',
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'status': 'success',
                    'model': self.settings.LLM_MODEL,
                    'response_length': len(result.get('response', '')),
                    'total_duration': result.get('total_duration', 0),
                    'load_duration': result.get('load_duration', 0)
                }
            else:
                return {
                    'status': 'failed',
                    'message': f"Inference failed with status {response.status_code}"
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'message': str(e)
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis cache health"""
        if not self.settings.ENABLE_CACHING:
            return {
                'status': 'disabled',
                'message': 'Caching is disabled in configuration'
            }
        
        try:
            from utils.caching import CacheManager
            
            cache_manager = CacheManager()
            await cache_manager.connect()
            
            health_result = await cache_manager.health_check()
            await cache_manager.disconnect()
            
            return health_result
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    async def check_vector_store(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            connection = await asyncpg.connect(self.settings.DATABASE_URL)
            
            # Check if collection exists
            collection_exists = await connection.fetchrow(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1) as exists",
                self.settings.COLLECTION_NAME
            )
            
            if collection_exists['exists']:
                # Get vector count
                vector_count = await connection.fetchrow(
                    f"SELECT COUNT(*) as count FROM {self.settings.COLLECTION_NAME}"
                )
                
                # Check vector dimensions (if any vectors exist)
                dimension_check = None
                if vector_count['count'] > 0:
                    dimension_check = await connection.fetchrow(
                        f"SELECT vector_dims(embedding) as dimension FROM {self.settings.COLLECTION_NAME} LIMIT 1"
                    )
                
                await connection.close()
                
                return {
                    'status': 'healthy',
                    'collection_exists': True,
                    'vector_count': vector_count['count'],
                    'vector_dimension': dimension_check['dimension'] if dimension_check else None,
                    'expected_dimension': self.settings.VECTOR_DIMENSION,
                    'collection_name': self.settings.COLLECTION_NAME
                }
            else:
                await connection.close()
                return {
                    'status': 'degraded',
                    'collection_exists': False,
                    'message': f"Vector collection '{self.settings.COLLECTION_NAME}' does not exist",
                    'collection_name': self.settings.COLLECTION_NAME
                }
                
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    async def check_connectors(self) -> Dict[str, Any]:
        """Check health of all data connectors"""
        try:
            from connectors import (
                DocumentConnector, 
                WebConnector, 
                APIConnector, 
                DatabaseConnector, 
                MCPConnector
            )
            
            connectors = [
                ('document', DocumentConnector()),
                ('web', WebConnector()),
                ('api', APIConnector()),
                ('database', DatabaseConnector()),
                ('mcp', MCPConnector())
            ]
            
            results = {}
            
            for name, connector in connectors:
                try:
                    await connector.connect()
                    health = await connector.health_check()
                    await connector.disconnect()
                    results[name] = health
                except Exception as e:
                    results[name] = {
                        'status': 'unhealthy',
                        'message': str(e)
                    }
            
            # Calculate overall connector status
            statuses = [result.get('status', 'unhealthy') for result in results.values()]
            
            if all(status in ['healthy', 'disabled'] for status in statuses):
                overall_status = 'healthy'
            elif any(status == 'healthy' for status in statuses):
                overall_status = 'degraded'
            else:
                overall_status = 'unhealthy'
            
            return {
                'status': overall_status,
                'connectors': results
            }
            
        except Exception as e:
            logger.error(f"Connector health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    async def check_embeddings(self) -> Dict[str, Any]:
        """Check embedding model health"""
        try:
            from utils.embeddings import EmbeddingManager
            
            embedding_manager = EmbeddingManager()
            await embedding_manager.load_model()
            
            # Test embedding generation
            test_text = "This is a test sentence for embedding generation."
            embedding = await embedding_manager.get_embedding(test_text)
            
            if embedding and len(embedding) == self.settings.VECTOR_DIMENSION:
                return {
                    'status': 'healthy',
                    'model': self.settings.EMBEDDING_MODEL,
                    'dimension': len(embedding),
                    'expected_dimension': self.settings.VECTOR_DIMENSION,
                    'test_embedding_generated': True
                }
            else:
                return {
                    'status': 'degraded',
                    'message': 'Embedding generation failed or dimension mismatch',
                    'model': self.settings.EMBEDDING_MODEL
                }
                
        except Exception as e:
            logger.error(f"Embedding health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e),
                'model': self.settings.EMBEDDING_MODEL
            }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            status = 'healthy'
            warnings = []
            
            if cpu_percent > 80:
                status = 'degraded'
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 85:
                status = 'degraded'
                warnings.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = 'degraded'
                warnings.append(f"High disk usage: {disk.percent}%")
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'warnings': warnings
            }
            
        except ImportError:
            return {
                'status': 'unavailable',
                'message': 'psutil not installed, cannot check system resources'
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    def get_component_status(self, health_data: Dict[str, Any], component: str) -> str:
        """Get status of a specific component"""
        if component in health_data.get('checks', {}):
            return health_data['checks'][component].get('status', 'unknown')
        return 'unknown'
    
    def is_system_healthy(self, health_data: Dict[str, Any]) -> bool:
        """Check if system is healthy overall"""
        return health_data.get('overall_status') == 'healthy'
    
    def get_critical_issues(self, health_data: Dict[str, Any]) -> List[str]:
        """Get list of critical issues"""
        issues = []
        
        for component, check in health_data.get('checks', {}).items():
            if check.get('status') == 'unhealthy':
                message = check.get('message', f"{component} is unhealthy")
                issues.append(f"{component}: {message}")
        
        return issues