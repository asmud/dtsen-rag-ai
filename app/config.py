import os
from typing import Optional, List
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
import logging

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = "postgresql+psycopg://rag_user:rag_pass@postgres:5432/rag_db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Vector Store Configuration
    COLLECTION_NAME: str = "data_rag_kb"
    VECTOR_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension
    
    # LLM Configuration
    OLLAMA_API: str = "http://ollama:11434"
    LLM_MODEL: str = "gemma2:2b"
    LLM_FALLBACK_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512
    LLM_TIMEOUT: int = 60
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_FALLBACK_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "auto"  # auto, cpu, cuda, or cuda:0
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_BATCH_SIZE_GPU: int = 64  # Larger batch size for GPU
    
    # GPU Resource Management
    GPU_ENABLED: bool = True  # Enable GPU detection and usage
    GPU_DEVICE_ID: Optional[int] = None  # Specific GPU device (None = auto)
    GPU_MEMORY_FRACTION: float = 0.8  # Fraction of GPU memory to use
    CPU_WORKERS: int = 4  # Number of CPU workers for parallel processing
    MIXED_PRECISION: bool = True  # Use mixed precision for GPU operations
    
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
    
    # Redis Configuration (for caching and task queue)
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    REDIS_MAX_CONNECTIONS: int = 10
    
    # Task Queue Configuration
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"
    TASK_QUEUE_ENABLED: bool = True
    
    # Performance Settings
    ENABLE_CACHING: bool = True
    BATCH_PROCESSING: bool = True
    ASYNC_PROCESSING: bool = True
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300
    
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
    MCP_ENABLED: bool = False
    MCP_SERVER_URL: Optional[str] = None
    MCP_API_KEY: Optional[str] = None
    
    # Database Query Settings
    DB_QUERY_ENABLED: bool = False
    DB_QUERY_URL: Optional[str] = None
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
    
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }

# Global settings instance
settings = Settings()

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
    critical_vars = [
        'DATABASE_URL',
        'OLLAMA_API',
        'COLLECTION_NAME'
    ]
    
    missing_vars = []
    for var in critical_vars:
        if not getattr(settings, var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing critical environment variables: {missing_vars}")
    
    logger.info("Environment validation successful")
    return True