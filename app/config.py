import os
from typing import Optional, List
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
import logging
import platform
import psutil

class Settings(BaseSettings):
    # Essential Configuration (must be set)
    POSTGRES_PASSWORD: str = "rag_pass"  # Only essential variable that needs to be set
    
    # Database Configuration - Smart defaults using Docker service names
    DATABASE_URL: Optional[str] = None  # Auto-generated from components
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rag_db"
    POSTGRES_USER: str = "rag_user"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Vector Store Configuration
    COLLECTION_NAME: str = "data_rag_kb"
    VECTOR_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension
    
    # LLM Configuration - Smart defaults
    OLLAMA_HOST: str = "ollama"
    OLLAMA_PORT: int = 11434
    OLLAMA_API: Optional[str] = None  # Auto-generated
    LLM_MODEL: str = "gemma2:2b"
    LLM_FALLBACK_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512
    LLM_TIMEOUT: int = 120
    
    # Embedding Configuration - Auto-detected optimal settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_FALLBACK_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "auto"  # Auto-detect best device
    EMBEDDING_BATCH_SIZE: Optional[int] = None  # Auto-detect based on hardware
    EMBEDDING_BATCH_SIZE_GPU: Optional[int] = None  # Auto-detect based on GPU
    
    # Hardware Configuration - Auto-detected
    GPU_ENABLED: Optional[bool] = None  # Auto-detect GPU availability
    GPU_DEVICE_ID: Optional[int] = None  # Auto-detect best GPU
    GPU_MEMORY_FRACTION: float = 0.8
    CPU_WORKERS: Optional[int] = None  # Auto-detect based on CPU cores
    MIXED_PRECISION: Optional[bool] = None  # Auto-detect based on hardware
    
    # Document Processing
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_EXTENSIONS: str = ".txt,.md,.pdf,.docx,.html"
    
    # Web Crawling (crawl4ai)
    CRAWL_RATE_LIMIT: float = 1.0  # requests per second
    CRAWL_TIMEOUT: int = 30
    CRAWL_MAX_PAGES: int = 100
    CRAWL_USER_AGENT: str = "RAG-Crawler/1.0"
    CRAWL_RESPECT_ROBOTS: bool = True
    CRAWL_MAX_DEPTH: int = 3
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = False
    API_LOG_LEVEL: str = "info"
    CORS_ORIGINS: str = "*"
    
    # Redis Configuration - Smart defaults using Docker service names
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None  # Auto-generated
    REDIS_CACHE_TTL: int = 3600
    REDIS_MAX_CONNECTIONS: int = 10
    
    # Task Queue Configuration - Auto-generated URLs
    CELERY_BROKER_URL: Optional[str] = None  # Auto-generated as redis://redis:6379/1
    CELERY_RESULT_BACKEND: Optional[str] = None  # Auto-generated as redis://redis:6379/2
    TASK_QUEUE_ENABLED: bool = True
    
    # Performance Settings - Auto-optimized based on hardware
    ENABLE_CACHING: bool = True
    BATCH_PROCESSING: bool = True
    ASYNC_PROCESSING: bool = True
    MAX_CONCURRENT_REQUESTS: Optional[int] = None  # Auto-detect based on CPU cores
    REQUEST_TIMEOUT: int = 180
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    
    # Health Check Configuration
    HEALTH_CHECK_INTERVAL: int = 30
    HEALTH_CHECK_TIMEOUT: int = 5
    
    # Data Source Configuration
    DATA_DIR: str = "/app/data"
    BACKUP_DIR: str = "/app/backups"
    TEMP_DIR: str = "/tmp/rag"
    
    # API Integration Settings
    API_REQUEST_TIMEOUT: int = 30
    API_MAX_RETRIES: int = 3
    API_RETRY_DELAY: int = 1
    
    # MCP (Model Context Protocol) Settings
    MCP_ENABLED: bool = True
    MCP_SERVER_URL: str = "https://jsonplaceholder.typicode.com"
    MCP_API_KEY: Optional[str] = None
    
    # Database Query Settings - Uses main database connection
    DB_QUERY_ENABLED: bool = True
    DB_QUERY_URL: Optional[str] = None  # Uses DATABASE_URL
    DB_QUERY_TIMEOUT: int = 30
    DB_QUERY_MAX_ROWS: int = 1000
    
    # Security Settings
    API_KEY: Optional[str] = None
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Swagger UI Configuration
    SWAGGER_UI_ENABLED: bool = True
    SWAGGER_UI_PATH: str = "/docs"
    REDOC_PATH: str = "/redoc"
    OPENAPI_JSON_PATH: str = "/openapi.json"
    SWAGGER_UI_TITLE: str = "DTSEN RAG AI API Documentation"
    SWAGGER_UI_DESCRIPTION: str = "Comprehensive API documentation for the DTSEN RAG AI System"
    API_CONTACT_NAME: str = "DTSEN RAG AI Team"
    API_CONTACT_EMAIL: str = "support@dtsen-rag-ai.com"
    API_LICENSE_NAME: str = "MIT"
    API_LICENSE_URL: str = "https://opensource.org/licenses/MIT"
    
    # System Prompt Configuration
    SYSTEM_PROMPT_ENABLED: bool = True
    SYSTEM_PROMPT_DEFAULT: str = """You are a helpful AI assistant specializing in Retrieval-Augmented Generation (RAG). 
Your role is to provide accurate, informative responses based on the provided context from indexed documents.

Guidelines:
- Always base your responses on the provided context when available
- If information is not in the context, clearly state this limitation
- Provide source citations when possible
- Be concise but comprehensive in your explanations
- Maintain a professional and helpful tone
- If asked about topics outside the provided context, politely redirect to document-based queries"""
    SYSTEM_PROMPT_MAX_LENGTH: int = 2000
    SYSTEM_PROMPT_OVERRIDE_ALLOWED: bool = True
    SYSTEM_PROMPT_TEMPLATE_ENABLED: bool = True
    
    # Hardware Detection Results (set by auto_detect_hardware)
    DEPLOYMENT_PROFILE: Optional[str] = None
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('SUPPORTED_EXTENSIONS', mode='before')
    @classmethod
    def validate_extensions(cls, v):
        if isinstance(v, str):
            # Keep as string, just validate format
            extensions = [ext.strip() for ext in v.split(',')]
            validated_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
            return ','.join(validated_extensions)
        elif isinstance(v, list):
            # Convert list to comma-separated string
            validated_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in v]
            return ','.join(validated_extensions)
        return v
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions as a list"""
        if isinstance(self.SUPPORTED_EXTENSIONS, str):
            return [ext.strip() for ext in self.SUPPORTED_EXTENSIONS.split(',')]
        return self.SUPPORTED_EXTENSIONS
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list"""
        if isinstance(self.CORS_ORIGINS, str):
            if self.CORS_ORIGINS == "*":
                return ["*"]
            return [origin.strip() for origin in self.CORS_ORIGINS.split(',')]
        return self.CORS_ORIGINS
    
    def get_database_url(self) -> str:
        """Generate database URL from components"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def get_redis_url(self) -> str:
        """Generate Redis URL from components"""
        if self.REDIS_URL:
            return self.REDIS_URL
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_ollama_api(self) -> str:
        """Generate Ollama API URL"""
        if self.OLLAMA_API:
            return self.OLLAMA_API
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"
    
    def get_celery_broker_url(self) -> str:
        """Generate Celery broker URL"""
        if self.CELERY_BROKER_URL:
            return self.CELERY_BROKER_URL
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/1"
    
    def get_celery_result_backend(self) -> str:
        """Generate Celery result backend URL"""
        if self.CELERY_RESULT_BACKEND:
            return self.CELERY_RESULT_BACKEND
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/2"
    
    def get_db_query_url(self) -> str:
        """Get database query URL (same as main database)"""
        return self.DB_QUERY_URL or self.get_database_url()
    
    def detect_hardware_profile(self) -> str:
        """Detect optimal deployment profile based on hardware"""
        try:
            # Check for Apple Silicon
            if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
                return 'apple-silicon'
            
            # Check for NVIDIA GPU
            try:
                import torch
                if torch.cuda.is_available():
                    return 'nvidia-gpu'
            except ImportError:
                pass
            
            # Check for NVIDIA GPU via nvidia-smi
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return 'nvidia-gpu'
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Default to CPU-only
            return 'cpu-only'
            
        except Exception:
            return 'cpu-only'
    
    def auto_detect_hardware(self):
        """Enhanced auto-detect optimal hardware settings"""
        try:
            # Detect hardware profile
            hardware_profile = self.detect_hardware_profile()
            
            # Get system information
            cpu_count = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Auto-detect CPU workers based on hardware profile and system specs
            if self.CPU_WORKERS is None:
                if hardware_profile == 'apple-silicon':
                    # Apple Silicon: Use more workers due to efficient cores
                    self.CPU_WORKERS = max(2, min(cpu_count - 1, int(cpu_count * 0.8)))
                elif hardware_profile == 'nvidia-gpu':
                    # GPU systems: Conservative CPU usage, let GPU handle heavy work
                    self.CPU_WORKERS = max(2, min(cpu_physical, 8))
                else:
                    # CPU-only: More aggressive CPU usage
                    self.CPU_WORKERS = max(2, min(cpu_count - 1, int(cpu_count * 0.9)))
            
            # Auto-detect max concurrent requests
            if self.MAX_CONCURRENT_REQUESTS is None:
                if hardware_profile == 'apple-silicon':
                    # Apple Silicon can handle more concurrent requests efficiently
                    self.MAX_CONCURRENT_REQUESTS = max(4, min(self.CPU_WORKERS * 2, 16))
                elif hardware_profile == 'nvidia-gpu':
                    # GPU systems: Higher concurrency for parallel processing
                    self.MAX_CONCURRENT_REQUESTS = max(8, min(self.CPU_WORKERS * 3, 32))
                else:
                    # CPU-only: Conservative concurrency
                    self.MAX_CONCURRENT_REQUESTS = max(4, min(self.CPU_WORKERS, 12))
            
            # Auto-detect GPU settings
            if self.GPU_ENABLED is None:
                if hardware_profile == 'nvidia-gpu':
                    try:
                        import torch
                        self.GPU_ENABLED = torch.cuda.is_available()
                        if self.GPU_ENABLED and self.MIXED_PRECISION is None:
                            self.MIXED_PRECISION = True
                    except ImportError:
                        self.GPU_ENABLED = False
                        self.MIXED_PRECISION = False
                else:
                    self.GPU_ENABLED = False
                    self.MIXED_PRECISION = False
            
            # Auto-detect embedding batch sizes based on hardware and memory
            if self.EMBEDDING_BATCH_SIZE is None:
                if hardware_profile == 'apple-silicon':
                    # Apple Silicon: Optimized for unified memory
                    if memory_gb >= 16:
                        self.EMBEDDING_BATCH_SIZE = 64
                        self.EMBEDDING_BATCH_SIZE_GPU = 32
                    elif memory_gb >= 8:
                        self.EMBEDDING_BATCH_SIZE = 48
                        self.EMBEDDING_BATCH_SIZE_GPU = 24
                    else:
                        self.EMBEDDING_BATCH_SIZE = 32
                        self.EMBEDDING_BATCH_SIZE_GPU = 16
                elif hardware_profile == 'nvidia-gpu':
                    # NVIDIA GPU: Higher batch sizes for GPU acceleration
                    if memory_gb >= 16:
                        self.EMBEDDING_BATCH_SIZE = 128
                        self.EMBEDDING_BATCH_SIZE_GPU = 256
                    elif memory_gb >= 8:
                        self.EMBEDDING_BATCH_SIZE = 96
                        self.EMBEDDING_BATCH_SIZE_GPU = 192
                    else:
                        self.EMBEDDING_BATCH_SIZE = 64
                        self.EMBEDDING_BATCH_SIZE_GPU = 128
                else:
                    # CPU-only: Conservative batch sizes
                    if memory_gb >= 16:
                        self.EMBEDDING_BATCH_SIZE = 48
                        self.EMBEDDING_BATCH_SIZE_GPU = 16
                    elif memory_gb >= 8:
                        self.EMBEDDING_BATCH_SIZE = 32
                        self.EMBEDDING_BATCH_SIZE_GPU = 16
                    else:
                        self.EMBEDDING_BATCH_SIZE = 24
                        self.EMBEDDING_BATCH_SIZE_GPU = 12
            
            # Set deployment profile for reference
            if self.DEPLOYMENT_PROFILE is None:
                self.DEPLOYMENT_PROFILE = hardware_profile
            
            # Auto-detect embedding device
            if self.EMBEDDING_DEVICE == "auto":
                if hardware_profile == 'nvidia-gpu' and self.GPU_ENABLED:
                    self.EMBEDDING_DEVICE = "cuda"
                elif hardware_profile == 'apple-silicon':
                    # Apple Silicon: Use MPS if available, otherwise CPU
                    try:
                        import torch
                        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            self.EMBEDDING_DEVICE = "mps"
                        else:
                            self.EMBEDDING_DEVICE = "cpu"
                    except ImportError:
                        self.EMBEDDING_DEVICE = "cpu"
                else:
                    self.EMBEDDING_DEVICE = "cpu"
            
            # Set URLs if not already set
            if not self.DATABASE_URL:
                self.DATABASE_URL = self.get_database_url()
            if not self.REDIS_URL:
                self.REDIS_URL = self.get_redis_url()
            if not self.OLLAMA_API:
                self.OLLAMA_API = self.get_ollama_api()
            if not self.CELERY_BROKER_URL:
                self.CELERY_BROKER_URL = self.get_celery_broker_url()
            if not self.CELERY_RESULT_BACKEND:
                self.CELERY_RESULT_BACKEND = self.get_celery_result_backend()
            if not self.DB_QUERY_URL:
                self.DB_QUERY_URL = self.get_db_query_url()
                
        except Exception as e:
            # Fallback to conservative defaults if auto-detection fails
            logging.warning(f"Hardware auto-detection failed: {e}. Using conservative defaults.")
            if self.CPU_WORKERS is None:
                self.CPU_WORKERS = 4
            if self.MAX_CONCURRENT_REQUESTS is None:
                self.MAX_CONCURRENT_REQUESTS = 8
            if self.GPU_ENABLED is None:
                self.GPU_ENABLED = False
            if self.MIXED_PRECISION is None:
                self.MIXED_PRECISION = False
            if self.EMBEDDING_BATCH_SIZE is None:
                self.EMBEDDING_BATCH_SIZE = 32
                self.EMBEDDING_BATCH_SIZE_GPU = 16
            if self.EMBEDDING_DEVICE == "auto":
                self.EMBEDDING_DEVICE = "cpu"
    
    def get_hardware_info(self) -> dict:
        """Get comprehensive hardware detection information"""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
            
            return {
                'deployment_profile': getattr(self, 'DEPLOYMENT_PROFILE', 'unknown'),
                'platform': {
                    'system': platform.system(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                },
                'cpu': {
                    'logical_cores': cpu_count,
                    'physical_cores': cpu_physical,
                    'workers_configured': self.CPU_WORKERS,
                },
                'memory': {
                    'total_gb': round(memory_gb, 2),
                    'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                },
                'gpu': {
                    'enabled': self.GPU_ENABLED,
                    'mixed_precision': self.MIXED_PRECISION,
                    'device_id': self.GPU_DEVICE_ID,
                },
                'optimization': {
                    'embedding_device': self.EMBEDDING_DEVICE,
                    'embedding_batch_size': self.EMBEDDING_BATCH_SIZE,
                    'embedding_batch_size_gpu': self.EMBEDDING_BATCH_SIZE_GPU,
                    'max_concurrent_requests': self.MAX_CONCURRENT_REQUESTS,
                },
                'urls': {
                    'database': self.get_database_url(),
                    'redis': self.get_redis_url(),
                    'ollama': self.get_ollama_api(),
                }
            }
        except Exception as e:
            return {'error': f'Failed to get hardware info: {e}'}
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }

# Global settings instance
settings = Settings()
# Auto-detect hardware and generate URLs on initialization
settings.auto_detect_hardware()

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        filename=settings.LOG_FILE
    )
    
    # Set specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

def get_settings() -> Settings:
    """Get application settings"""
    return settings

def validate_environment():
    """Validate critical environment variables"""
    # Check essential variables that must be set
    if not settings.POSTGRES_PASSWORD:
        raise ValueError("POSTGRES_PASSWORD must be set")
    
    # Check that auto-generated URLs are valid
    try:
        database_url = settings.get_database_url()
        ollama_api = settings.get_ollama_api()
        redis_url = settings.get_redis_url()
        
        logger.info(f"Database URL: {database_url}")
        logger.info(f"Ollama API: {ollama_api}")
        logger.info(f"Redis URL: {redis_url}")
        logger.info(f"CPU Workers: {settings.CPU_WORKERS}")
        logger.info(f"GPU Enabled: {settings.GPU_ENABLED}")
        logger.info(f"Max Concurrent Requests: {settings.MAX_CONCURRENT_REQUESTS}")
        
    except Exception as e:
        raise ValueError(f"Error generating configuration URLs: {e}")
    
    logger.info("Environment validation successful")
    return True