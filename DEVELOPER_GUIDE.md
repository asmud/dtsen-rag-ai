# DEVELOPER GUIDE

This file provides comprehensive development guidance for working with the DTSEN RAG AI codebase.

## 🦙 RAG Chatbot System Overview

A comprehensive multi-source Retrieval-Augmented Generation (RAG) system with the following capabilities:

- 📁 **Multi-source Data Ingestion**: Documents, web crawling, APIs, databases, MCP
- 🤖 **Dynamic System Prompts**: Configurable AI behavior with templates and variables
- 📊 **Interactive API Documentation**: Comprehensive Swagger UI and ReDoc interfaces
- ⚡ **Performance Optimized**: Low-resource design with intelligent caching
- 🔧 **Environment Driven**: Fully configurable via environment variables
- 🏥 **Health Monitoring**: Real-time system status and diagnostics

## Common Development Commands

### Docker Services
```bash
# Start all services (optimized for low resources)
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f chatbot
docker compose logs -f ollama

# Rebuild after changes
docker compose build chatbot
```

### Data Indexing Workflows
```bash
# Index documents from /data folder
docker compose exec chatbot python rag_pipeline.py

# Index with specific data source
docker compose exec chatbot python rag_pipeline.py --source documents
docker compose exec chatbot python rag_pipeline.py --source api
docker compose exec chatbot python rag_pipeline.py --source crawl
docker compose exec chatbot python rag_pipeline.py --source database

# Re-index all sources
docker compose exec chatbot python rag_pipeline.py --reindex-all
```

### Web Crawling with crawl4ai
```bash
# Crawl specific URLs
docker compose exec chatbot python -m crawl4ai.crawler --url "https://example.com"

# Batch crawl from URL file
docker compose exec chatbot python batch_crawler.py --input urls.txt
```

### API Testing
```bash
# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about topic XYZ"}'

# Health check
curl "http://localhost:8000/health"

# System information
curl "http://localhost:8000/info"

# API documentation metadata
curl "http://localhost:8000/api-info"

# Check available models
curl "http://localhost:11434/api/tags"
```

### API Documentation
```bash
# Access interactive Swagger UI documentation
open http://localhost:8000/docs

# Access alternative ReDoc documentation  
open http://localhost:8000/redoc

# Get OpenAPI schema JSON
curl "http://localhost:8000/openapi.json"

# API root with quick links
curl "http://localhost:8000/"

# System prompt management
curl "http://localhost:8000/system-prompt"
curl "http://localhost:8000/system-prompt/templates"

# Test chat with custom system prompt
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain machine learning",
    "system_prompt_template": "educational",
    "system_prompt_variables": {"domain": "AI/ML"}
  }'

# Test chat with custom system prompt
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze this document",
    "system_prompt": "You are a technical reviewer. Focus on key details and recommendations."
  }'
```

### Model Management
```bash
# Switch to primary model (gemma2:2b)
docker compose exec ollama ollama pull gemma2:2b

# Switch to fallback model (llama3)
docker compose exec ollama ollama pull llama3

# Test model performance
docker compose exec chatbot python test_models.py
```

### Database Operations
```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U rag_user -d rag_db

# Check vector storage
docker compose exec postgres psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM rag_kb;"

# Reset vector store
docker compose exec chatbot python reset_vectorstore.py

# Check Redis cache status
docker compose exec redis redis-cli info keyspace

# Clear cache
docker compose exec redis redis-cli flushall
```

### System Prompt Management
```bash
# View available system prompt templates
docker compose exec chatbot python -c "
from app.utils.prompts import get_prompt_manager
pm = get_prompt_manager()
templates = pm.get_available_templates()
for name, info in templates.items():
    print(f'{name}: {info[\"description\"]}')
"

# Test system prompt validation
curl -X POST "http://localhost:8000/system-prompt/test" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a helpful assistant specialized in technical documentation."}'

# Get current system prompt configuration
curl "http://localhost:8000/system-prompt" | jq '.'
```

## Architecture Overview

### RAG Pipeline Architecture
The system uses a multi-source RAG pipeline optimized for low-resource environments:

1. **Data Sources**:
   - Documents (PDF, TXT, MD) in `/data` folder
   - Web content via crawl4ai integration
   - API endpoints with structured data extraction
   - MCP (Model Context Protocol) connectors
   - RDBMS structured data queries

2. **Processing Flow**:
   ```
   Data Sources → Chunking → Embeddings (all-MiniLM-L6-v2) → PGVector → Query Engine + System Prompt → LLM (gemma2:2b) → Response
   ```

3. **Performance Optimizations**:
   - **Lightweight models**: gemma2:2b (2B params) + all-MiniLM-L6-v2 (22MB)
   - **Async processing**: Concurrent document processing and embedding generation
   - **Connection pooling**: Database connections optimized for concurrent access
   - **Embedding cache**: Redis-like caching for frequently accessed embeddings
   - **Smart chunking**: Context-aware text splitting with overlap optimization

### Service Dependencies
- **PostgreSQL + pgvector**: Vector storage with optimized indexing
- **Redis**: Caching layer for embeddings, queries, and document chunks
- **Ollama**: Local LLM serving (gemma2:2b primary, llama3 fallback)
- **FastAPI**: REST API with async request handling and comprehensive documentation
- **crawl4ai**: Web content extraction with rate limiting and content processing

### Key Components

#### `/app/rag_pipeline.py`
Multi-source data ingestion pipeline supporting:
- Document processing with async chunking
- Web crawling integration
- API data extraction
- Database query execution
- Embedding generation and storage

#### `/app/chatbot_api.py`
FastAPI server with:
- Async query processing with system prompt integration
- Model switching capability
- Health monitoring endpoints
- Performance metrics collection
- **Dynamic system prompt management** with templates and variables
- **Comprehensive Swagger UI documentation** with interactive examples
- **Interactive API exploration** at `/docs` with organized endpoint categories
- **Alternative ReDoc documentation** at `/redoc`
- **Custom OpenAPI schema** generation with enhanced metadata
- **API information endpoints** with system capabilities and features

#### `/app/data_connectors/`
Modular connectors for different data sources:
- `document_connector.py`: File system document processing
- `web_connector.py`: crawl4ai integration
- `api_connector.py`: REST API data extraction
- `mcp_connector.py`: Model Context Protocol implementation
- `db_connector.py`: RDBMS structured data queries

#### `/app/utils/`
Shared utility modules:
- `chunking.py`: Context-aware text chunking with overlap
- `caching.py`: Redis-based caching for embeddings and queries
- `health.py`: Comprehensive system health monitoring
- `embeddings.py`: Embedding generation with fallback models
- **`prompts.py`: Dynamic system prompt management with templates**

#### `/app/models/`
Pydantic data models:
- `schemas.py`: API request/response models with examples
- `document.py`: Document and metadata models
- **Enhanced models** with system prompt support and validation

### Environment Variables

#### Core System Configuration
- `DATABASE_URL`: PostgreSQL connection string with pgvector support
- `OLLAMA_API`: Ollama service endpoint for LLM inference
- `COLLECTION_NAME`: Vector store collection name (default: rag_kb)
- `REDIS_URL`: Redis connection string for caching
- `ENABLE_CACHING`: Enable Redis caching (default: true)

#### Model Configuration
- `LLM_MODEL`: Primary LLM model (default: gemma2:2b)
- `LLM_FALLBACK_MODEL`: Fallback LLM model (default: llama3)
- `EMBEDDING_MODEL`: Primary embedding model (default: all-MiniLM-L6-v2)
- `EMBEDDING_FALLBACK_MODEL`: Fallback embedding model (default: BAAI/bge-m3)
- `EMBEDDING_DEVICE`: Device for embeddings (default: cpu)

#### Processing Configuration
- `MAX_CHUNK_SIZE`: Document chunking size (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (default: 32)
- `MAX_CONCURRENT_REQUESTS`: Max concurrent API requests (default: 10)

#### Web Crawling Configuration
- `CRAWL_RATE_LIMIT`: Crawling rate limit (default: 1.0 req/sec)
- `CRAWL_MAX_PAGES`: Maximum pages per crawl session (default: 100)
- `CRAWL_TIMEOUT`: Timeout for crawl requests (default: 30)
- `CRAWL_MAX_DEPTH`: Maximum crawling depth (default: 3)

#### System Prompt Configuration
- `SYSTEM_PROMPT_ENABLED`: Enable system prompt functionality (default: true)
- `SYSTEM_PROMPT_DEFAULT`: Default system prompt text for RAG assistant
- `SYSTEM_PROMPT_OVERRIDE_ALLOWED`: Allow per-request prompt override (default: true)
- `SYSTEM_PROMPT_MAX_LENGTH`: Maximum prompt length (default: 2000)
- `SYSTEM_PROMPT_TEMPLATE_ENABLED`: Enable template variables (default: true)

#### API Documentation Configuration
- `SWAGGER_UI_ENABLED`: Enable Swagger UI documentation (default: true)
- `SWAGGER_UI_PATH`: Swagger UI endpoint path (default: /docs)
- `REDOC_PATH`: ReDoc documentation path (default: /redoc)
- `API_CONTACT_NAME`: API contact information
- `API_CONTACT_EMAIL`: API contact email

#### Security and Performance
- `RATE_LIMIT_ENABLED`: Enable API rate limiting (default: true)
- `RATE_LIMIT_REQUESTS`: Requests per window (default: 100)
- `API_KEY`: Optional API key for authentication
- `LOG_LEVEL`: Logging level (default: INFO)

## Development Workflow

### 1. Environment Setup
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env file with your specific values

# Start the application stack
docker-compose up -d

# Check services are healthy
docker-compose ps
curl http://localhost:8000/health
```

### 2. Adding New Documents
```bash
# Place documents in the data directory
mkdir -p data/documents
cp your-documents.pdf data/documents/

# The system will automatically index new files
# Or trigger manual indexing via API:
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "documents", "source_config": {"path": "/app/data/documents"}}'
```

### 3. Testing the System
```bash
# Test chat functionality
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

# Test search functionality
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 5}'
```

### 4. System Prompt Management
```bash
# Get current system prompt
curl http://localhost:8000/system-prompt

# Get available templates
curl http://localhost:8000/system-prompt/templates

# Test custom system prompt
curl -X POST "http://localhost:8000/system-prompt/test" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a specialized technical assistant."}'

# Use custom system prompt in chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain machine learning", "system_prompt": "You are a technical expert. Be concise."}'

# Use template in chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain this topic", "system_prompt_template": "technical_expert", "system_prompt_variables": {"domain": "AI"}}'
```

### 5. API Documentation Access
```bash
# Interactive Swagger UI
open http://localhost:8000/docs

# Alternative ReDoc documentation
open http://localhost:8000/redoc

# OpenAPI JSON schema
curl http://localhost:8000/openapi.json

# API information endpoint
curl http://localhost:8000/api-info
```

### 6. Multi-Source Data Ingestion
```bash
# Index web content
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "web", "source_config": {"urls": ["https://example.com"], "max_pages": 10}}'

# Index API data
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "api", "source_config": {"url": "https://api.example.com/data", "content_fields": ["content", "description"]}}'

# Index database results (if enabled)
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "database", "source_config": {"query": "SELECT * FROM articles", "content_columns": ["title", "body"]}}'
```

### 7. Monitoring and Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# View system information
curl http://localhost:8000/info

# Monitor logs
docker-compose logs -f chatbot

# Check specific component health
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f ollama
```

### 8. Performance Optimization
```bash
# Monitor resource usage
docker stats

# Check Redis cache status
docker exec -it rag_redis redis-cli info memory

# Monitor Ollama model performance
curl http://localhost:11434/api/tags

# Adjust performance settings in .env:
# - MAX_CHUNK_SIZE and CHUNK_OVERLAP for text processing
# - REDIS_CACHE_TTL for caching duration
# - MAX_CONCURRENT_REQUESTS for load handling
# - EMBEDDING_BATCH_SIZE for embedding efficiency
```

### 9. Background Task Management (Optional)
```bash
# Start with Celery worker for background processing
docker-compose --profile with-celery up -d

# Monitor background tasks
docker-compose logs -f celery-worker

# Submit large indexing jobs that run in background
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "documents", "source_config": {"path": "/app/data/large-dataset"}, "batch_size": 50}'
```

### 10. Debugging and Troubleshooting
```bash
# Check container health status
docker-compose ps

# Restart specific service
docker-compose restart chatbot

# View detailed logs with timestamps
docker-compose logs -t --tail=100 chatbot

# Connect to container for debugging
docker-compose exec chatbot bash

# Check Python dependencies
docker-compose exec chatbot pip list

# Test database connection
docker-compose exec chatbot python -c "
from app.config import get_settings
print('Database URL:', get_settings().DATABASE_URL)
"

# Validate configuration
docker-compose exec chatbot python -c "
from app.config import validate_environment
validate_environment()
print('Configuration validated successfully')
"
```

## Resource Optimization Notes
- System designed for 4GB+ RAM environments with gemma2:2b (2B parameter model)
- Models chosen for CPU inference capability with acceptable performance
- Database queries optimized with proper indexing and connection pooling
- Async operations prevent blocking on I/O operations
- Redis caching reduces redundant computation costs for embeddings and queries
- Batch processing for embeddings improves throughput
- Environment-driven configuration allows fine-tuning for specific hardware

## System Prompt Features
- **Dynamic Templates**: Use predefined prompt templates for different use cases
- **Variable Substitution**: Inject context-specific variables into prompts
- **Per-Request Override**: Custom prompts for individual chat requests
- **Validation**: Automatic prompt validation for length and content safety
- **Preview**: Test prompts before using them in production
- **Environment Configuration**: Set global default prompts via environment variables

## API Documentation Features
- **Interactive Swagger UI**: Full API exploration at `/docs` with live testing
- **ReDoc Alternative**: Clean documentation at `/redoc` for reference
- **Comprehensive Examples**: All endpoints include request/response examples
- **Organized Categories**: Endpoints grouped by functionality (Chat, Search, System, etc.)
- **Custom OpenAPI Schema**: Enhanced metadata with server information and contact details
- **API Information Endpoint**: Programmatic access to API capabilities at `/api-info`

## Performance Monitoring
- **Health Endpoints**: Real-time system health at `/health` with component status
- **System Information**: Detailed configuration and capabilities at `/info`
- **Resource Metrics**: Monitor CPU, memory, and disk usage with Docker stats
- **Cache Analytics**: Redis memory usage and hit rates
- **Processing Times**: Request timing metadata in API responses
- **Background Task Tracking**: Optional Celery integration for heavy workloads