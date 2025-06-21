import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from config import get_settings, validate_environment, logger
from rag_pipeline import RAGPipeline
from models.schemas import (
    ChatRequest, ChatResponse, HealthResponse, SystemInfo, 
    SearchRequest, SearchResponse, IndexRequest, IndexResponse,
    ErrorResponse, SystemPromptTemplatesResponse, SystemPromptRequest,
    SystemPromptResponse, SystemPromptInfo, SystemPromptTemplate
)
from utils.health import HealthChecker
from utils.embeddings import EmbeddingManager
from utils.caching import CacheManager
from utils.prompts import get_prompt_manager
from utils.resource_manager import get_resource_manager
from connectors import (
    DocumentConnector, WebConnector, APIConnector, 
    DatabaseConnector, MCPConnector
)

# Global state
app_state = {
    "index": None,
    "query_engine": None,
    "llm": None,
    "embedding_manager": None,
    "cache_manager": None,
    "health_checker": None,
    "rag_pipeline": None,
    "startup_time": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting RAG chatbot application...")
    app_state["startup_time"] = time.time()
    
    try:
        # Validate environment
        validate_environment()
        settings = get_settings()
        
        # Initialize components
        await initialize_components()
        
        logger.info("Application startup completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down application...")
    await cleanup_components()
    logger.info("Application shutdown completed")

async def initialize_components():
    """Initialize all application components"""
    settings = get_settings()
    
    # Initialize health checker
    app_state["health_checker"] = HealthChecker()
    
    # Initialize cache manager
    app_state["cache_manager"] = CacheManager()
    if settings.ENABLE_CACHING:
        await app_state["cache_manager"].connect()
    
    # Initialize embedding manager
    app_state["embedding_manager"] = EmbeddingManager()
    await app_state["embedding_manager"].initialize()
    
    # Initialize LLM with proper timeout and request timeout
    app_state["llm"] = Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_API,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
        request_timeout=settings.REQUEST_TIMEOUT
    )
    
    # Create LlamaIndex embedding model
    app_state["llama_embedding"] = HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE
    )
    
    # Initialize vector store and index
    await initialize_vector_store()
    
    # Initialize RAG pipeline
    app_state["rag_pipeline"] = RAGPipeline()
    await app_state["rag_pipeline"].initialize()
    
    logger.info("All components initialized successfully")

async def initialize_vector_store():
    """Initialize vector store and index"""
    settings = get_settings()
    
    try:
        # Create vector store with explicit async connection string
        sync_url = settings.DATABASE_URL
        async_url = sync_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        vector_store = PGVectorStore.from_params(
            connection_string=sync_url,
            async_connection_string=async_url,
            table_name=settings.COLLECTION_NAME,
            embed_dim=settings.VECTOR_DIMENSION
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Try to load existing index
        try:
            app_state["index"] = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=app_state["llama_embedding"]
            )
            logger.info("Loaded existing vector index")
        except Exception:
            # Create new index if none exists
            app_state["index"] = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=app_state["llama_embedding"]
            )
            logger.info("Created new vector index")
        
        # Create query engine
        app_state["query_engine"] = app_state["index"].as_query_engine(
            llm=app_state["llm"],
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise

async def cleanup_components():
    """Cleanup application components"""
    try:
        if app_state["rag_pipeline"]:
            await app_state["rag_pipeline"].cleanup()
            
        if app_state["embedding_manager"]:
            await app_state["embedding_manager"].cleanup()
        
        if app_state["cache_manager"]:
            await app_state["cache_manager"].disconnect()
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Get settings for app configuration
settings = get_settings()

# Create custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.SWAGGER_UI_TITLE,
        version="2.0.0",
        description="""
## 🦙 Multi-source RAG Chatbot API

A comprehensive Retrieval-Augmented Generation (RAG) system that supports multiple data sources including:

- 📁 **Local Documents** - PDF, TXT, MD, DOCX, HTML files
- 🌐 **Web Crawling** - Real-time web content extraction with crawl4ai
- 🔌 **API Integration** - REST API data ingestion
- 🗄️ **Database Queries** - Structured data from RDBMS
- 🤖 **MCP Protocol** - Model Context Protocol integration

### 🚀 Key Features

- **Optimized for Low Resources**: Uses gemma2:2b LLM + all-MiniLM-L6-v2 embeddings
- **Redis Caching**: Fast response times with intelligent caching
- **Health Monitoring**: Comprehensive system health checks
- **Async Processing**: High-performance concurrent operations
- **Environment-Driven**: Fully configurable via environment variables

### 🔧 Performance Optimizations

- **Smart Chunking**: Context-aware text splitting with overlap
- **Batch Processing**: Efficient embedding generation
- **Connection Pooling**: Optimized database connections
- **Background Tasks**: Non-blocking document indexing

### 📊 API Endpoints

Use the endpoints below to interact with the RAG system:

- **Chat**: Query the system with natural language
- **Search**: Direct vector similarity search
- **Index**: Add new documents from various sources
- **Health**: Monitor system status
- **Info**: Get system information and capabilities
        """,
        routes=app.routes,
        contact={
            "name": settings.API_CONTACT_NAME,
            "email": settings.API_CONTACT_EMAIL,
        },
        license_info={
            "name": settings.API_LICENSE_NAME,
            "url": settings.API_LICENSE_URL,
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.rag-chatbot.com",
                "description": "Production server"
            }
        ]
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add API tags
    openapi_schema["tags"] = [
        {
            "name": "Chat",
            "description": "Conversational AI endpoints for querying the RAG system",
        },
        {
            "name": "Search",
            "description": "Direct search and retrieval operations",
        },
        {
            "name": "Indexing",
            "description": "Document ingestion and processing from multiple sources",
        },
        {
            "name": "System",
            "description": "System health, monitoring, and information endpoints",
        },
        {
            "name": "Documentation",
            "description": "API documentation and schema endpoints",
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Create FastAPI app
app = FastAPI(
    title=settings.SWAGGER_UI_TITLE,
    description=settings.SWAGGER_UI_DESCRIPTION,
    version="2.0.0",
    lifespan=lifespan,
    docs_url=settings.SWAGGER_UI_PATH if settings.SWAGGER_UI_ENABLED else None,
    redoc_url=settings.REDOC_PATH if settings.SWAGGER_UI_ENABLED else None,
    openapi_url=settings.OPENAPI_JSON_PATH if settings.SWAGGER_UI_ENABLED else None
)

# Set custom OpenAPI schema
app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred"
        ).dict()
    )

# Dependency to get query engine
async def get_query_engine():
    if not app_state["query_engine"]:
        raise HTTPException(
            status_code=503,
            detail="Query engine not initialized"
        )
    return app_state["query_engine"]

# API Endpoints
@app.post(
    "/chat", 
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat with RAG System",
    description="""
    Submit a question to the RAG system and receive an AI-generated response based on indexed documents.
    
    **Features:**
    - Natural language processing
    - Context-aware responses
    - Source document citations
    - Conversation context support
    - Customizable response parameters
    
    **Use Cases:**
    - Ask questions about uploaded documents
    - Get summaries of specific topics
    - Retrieve information with source citations
    """,
    responses={
        200: {
            "description": "Successful response with answer and sources",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Based on the documents, RAG systems combine retrieval and generation to provide accurate responses.",
                        "conversation_id": "conv_12345",
                        "sources": [
                            {
                                "id": "doc_12345",
                                "title": "Introduction to RAG",
                                "content": "RAG systems...",
                                "source": "documents",
                                "similarity_score": 0.85
                            }
                        ],
                        "metadata": {"processing_time_ms": 150},
                        "timestamp": "2023-01-01T12:00:00"
                    }
                }
            }
        },
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)
async def chat(
    request: ChatRequest,
    query_engine = Depends(get_query_engine)
):
    """Chat with the RAG system to get AI-generated responses based on indexed documents"""
    try:
        start_time = time.time()
        
        # Get system prompt
        prompt_manager = get_prompt_manager()
        system_prompt = prompt_manager.get_system_prompt(
            user_prompt=request.system_prompt,
            template_name=request.system_prompt_template,
            variables=request.system_prompt_variables
        )
        
        # Create cache key that includes system prompt if it affects the response
        cache_key_extra = {}
        if system_prompt:
            cache_key_extra['system_prompt_hash'] = hash(system_prompt)
        
        # Check cache first
        cache_manager = app_state["cache_manager"]
        if cache_manager and settings.ENABLE_CACHING:
            cached_response = await cache_manager.get_query_results(
                request.question, 
                {**(request.filters or {}), **cache_key_extra}
            )
            if cached_response:
                logger.info(f"Cache hit for query: {request.question[:50]}...")
                return ChatResponse(**cached_response)
        
        # Enhance question with system prompt if provided
        if system_prompt and settings.SYSTEM_PROMPT_ENABLED:
            enhanced_question = f"System: {system_prompt}\n\nUser: {request.question}"
        else:
            enhanced_question = request.question
        
        # Query the system with timeout
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(query_engine.query, enhanced_question),
                timeout=settings.REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Query timeout after {settings.REQUEST_TIMEOUT} seconds"
            )
        
        # Extract source documents
        sources = []
        if request.include_sources and hasattr(response, 'source_nodes'):
            for i, node in enumerate(response.source_nodes[:5]):
                # Safely extract node data with proper null handling
                node_id = getattr(node, 'node_id', f'source_{i}')
                node_metadata = getattr(node, 'metadata', {}) or {}
                node_score = getattr(node, 'score', 0.0)
                
                # Ensure score is a float, not a string
                if isinstance(node_score, str):
                    try:
                        node_score = float(node_score) if node_score != 'None' else 0.0
                    except (ValueError, TypeError):
                        node_score = 0.0
                elif node_score is None:
                    node_score = 0.0
                
                sources.append({
                    "id": str(node_id) if node_id is not None else f'source_{i}',
                    "title": node_metadata.get('title', 'Unknown'),
                    "content": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                    "source": node_metadata.get('source', 'unknown'),
                    "url": node_metadata.get('url'),
                    "similarity_score": node_score,
                    "metadata": node_metadata
                })
        
        # Create response
        chat_response = ChatResponse(
            response=response.response,
            conversation_id=request.conversation_id,
            sources=sources,
            metadata={
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_used": settings.LLM_MODEL,
                "source_count": len(sources),
                "system_prompt_used": bool(system_prompt),
                "system_prompt_source": "custom" if request.system_prompt else 
                                      "template" if request.system_prompt_template else "default",
                "system_prompt_template": request.system_prompt_template,
                "system_prompt_preview": prompt_manager.format_prompt_for_display(system_prompt, 100) if system_prompt else None
            }
        )
        
        # Cache the response
        if cache_manager and settings.ENABLE_CACHING:
            await cache_manager.set_query_results(
                request.question,
                chat_response.dict(),
                request.filters,
                ttl=1800  # 30 minutes
            )
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post(
    "/search", 
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search Documents",
    description="""
    Perform vector similarity search on indexed documents without generating AI responses.
    
    **Features:**
    - Direct vector similarity search
    - Configurable result limits
    - Similarity threshold filtering
    - Fast retrieval without LLM processing
    
    **Use Cases:**
    - Find relevant documents quickly
    - Get similarity scores
    - Retrieve document excerpts
    - Filter by relevance threshold
    """
)
async def search_documents(request: SearchRequest):
    """Search documents in the vector store using similarity search"""
    try:
        start_time = time.time()
        
        if not app_state["index"]:
            raise HTTPException(
                status_code=503,
                detail="Search index not available"
            )
        
        # Create retriever
        retriever = app_state["index"].as_retriever(
            similarity_top_k=request.limit
        )
        
        # Perform search
        nodes = await asyncio.to_thread(retriever.retrieve, request.query)
        
        # Filter by similarity threshold
        filtered_nodes = [
            node for node in nodes 
            if getattr(node, 'score', 0.0) >= request.similarity_threshold
        ]
        
        # Convert to search results
        results = []
        for node in filtered_nodes:
            result = {
                "id": getattr(node, 'node_id', ''),
                "title": getattr(node, 'metadata', {}).get('title', 'Unknown'),
                "content": node.text if request.include_content else None,
                "similarity_score": getattr(node, 'score', 0.0),
                "source": getattr(node, 'metadata', {}).get('source', 'unknown'),
                "metadata": getattr(node, 'metadata', {})
            }
            results.append(result)
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )

@app.post(
    "/index", 
    response_model=IndexResponse,
    tags=["Indexing"],
    summary="Index Documents",
    description="""
    Submit documents for indexing from various data sources.
    
    **Supported Sources:**
    - **Documents**: Local files (PDF, TXT, MD, DOCX, HTML)
    - **Web**: Web pages via URL crawling
    - **API**: REST API endpoints
    - **Database**: SQL query results
    - **MCP**: Model Context Protocol servers
    
    **Features:**
    - Background processing
    - Batch operations
    - Error handling and reporting
    - Progress tracking
    """
)
async def index_documents(request: IndexRequest):
    """Index documents from various data sources for retrieval"""
    try:
        # Get the RAG pipeline
        if not app_state["rag_pipeline"]:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized"
            )
        
        pipeline = app_state["rag_pipeline"]
        task_id = f"task_{int(time.time())}"
        
        # Process the indexing request
        logger.info(f"Processing indexing request for {request.source_type}")
        
        # Convert source config to kwargs
        kwargs = {}
        if request.source_config:
            if request.source_type == "database":
                # For database sources, pass queries configuration
                kwargs.update(request.source_config)
            elif request.source_type == "web":
                # For web sources, extract URLs
                if "urls" in request.source_config:
                    kwargs["urls"] = request.source_config["urls"]
            elif request.source_type == "api":
                # For API sources, pass the configuration
                kwargs.update(request.source_config)
            elif request.source_type == "mcp":
                # For MCP sources, pass the configuration
                kwargs.update(request.source_config)
        
        # Call the appropriate pipeline method
        if request.source_type == "web" and "urls" in kwargs:
            stats = await pipeline.process_web_content(kwargs["urls"])
        else:
            stats = await pipeline.process_documents(request.source_type, **kwargs)
        
        return IndexResponse(
            task_id=task_id,
            status="completed",
            message=f"Indexing completed for {request.source_type}",
            documents_processed=stats.get('documents_processed', 0),
            chunks_created=stats.get('chunks_created', 0),
            errors=stats.get('errors', [])
        )
        
    except Exception as e:
        logger.error(f"Error processing index request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing index request: {str(e)}"
        )

@app.get(
    "/health", 
    response_model=HealthResponse,
    tags=["System"],
    summary="System Health Check",
    description="""
    Get comprehensive health status of all system components.
    
    **Checks Include:**
    - Database connectivity (PostgreSQL + pgvector)
    - LLM service availability (Ollama)
    - Cache system status (Redis)
    - Vector store health
    - Data connector status
    - System resource usage
    
    **Status Levels:**
    - **Healthy**: All systems operational
    - **Degraded**: Some issues but functional
    - **Unhealthy**: Critical issues present
    """
)
async def health_check():
    """Get comprehensive health status of all system components"""
    try:
        health_checker = app_state["health_checker"]
        if not health_checker:
            raise HTTPException(status_code=503, detail="Health checker not available")
        
        health_data = await health_checker.check_all_systems()
        
        # Convert to response format
        components = {}
        for component_name, component_data in health_data.get('checks', {}).items():
            components[component_name] = {
                "status": component_data.get('status', 'unknown'),
                "message": component_data.get('message'),
                "details": component_data,
                "last_check": health_data['timestamp']
            }
        
        # Calculate uptime
        uptime_seconds = None
        if app_state["startup_time"]:
            uptime_seconds = time.time() - app_state["startup_time"]
        
        return HealthResponse(
            overall_status=health_data['overall_status'],
            components=components,
            uptime_seconds=uptime_seconds
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            overall_status="unhealthy",
            components={
                "health_check": {
                    "status": "unhealthy",
                    "message": f"Health check failed: {str(e)}",
                    "details": {},
                    "last_check": time.time()
                }
            }
        )

@app.get(
    "/info", 
    response_model=SystemInfo,
    tags=["System"],
    summary="System Information",
    description="""
    Get detailed information about the RAG system configuration and capabilities.
    
    **Information Includes:**
    - Available data connectors and their status
    - Configured models (LLM and embedding)
    - System settings and performance parameters
    - Enabled features and capabilities
    - API version and build information
    
    **Use Cases:**
    - Verify system configuration
    - Check available features
    - Monitor connector status
    - API client initialization
    """
)
async def system_info():
    """Get detailed system information and capabilities"""
    try:
        settings = get_settings()
        
        # Get connector info
        connectors = []
        connector_classes = [
            DocumentConnector, WebConnector, APIConnector,
            DatabaseConnector, MCPConnector
        ]
        
        for connector_class in connector_classes:
            try:
                connector = connector_class()
                info = connector.get_connector_info()
                connectors.append(info)
            except Exception as e:
                logger.error(f"Error getting connector info for {connector_class.__name__}: {e}")
        
        return SystemInfo(
            version="2.0.0",
            connectors=connectors,
            models={
                "llm_primary": settings.LLM_MODEL,
                "llm_fallback": settings.LLM_FALLBACK_MODEL,
                "embedding_primary": settings.EMBEDDING_MODEL,
                "embedding_fallback": settings.EMBEDDING_FALLBACK_MODEL
            },
            settings={
                "chunk_size": settings.MAX_CHUNK_SIZE,
                "cache_enabled": settings.ENABLE_CACHING,
                "async_processing": settings.ASYNC_PROCESSING,
                "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
            },
            capabilities=[
                "multi_source_indexing",
                "web_crawling",
                "api_integration", 
                "caching",
                "health_monitoring",
                "background_processing"
            ]
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system info: {str(e)}"
        )

@app.get(
    "/resources",
    tags=["System"],
    summary="Resource Utilization",
    description="""
    Get real-time system resource utilization including CPU, GPU, and memory usage.
    
    **Resource Information:**
    - CPU utilization and core count
    - GPU availability and memory usage
    - System memory statistics
    - Device optimization recommendations
    
    **Use Cases:**
    - Monitor system performance
    - Optimize resource allocation
    - Track hardware utilization
    - Debug performance issues
    """
)
async def get_resources():
    """Get real-time resource utilization and system performance metrics"""
    try:
        resource_manager = await get_resource_manager()
        
        # Get resource utilization
        utilization = await resource_manager.get_resource_utilization()
        
        # Get system information
        system_info = resource_manager.get_system_info()
        
        # Get optimization recommendations
        embedding_optimization = await resource_manager.optimize_for_task("embedding", "medium")
        llm_optimization = await resource_manager.optimize_for_task("llm_inference", "medium")
        
        return {
            "utilization": utilization,
            "system_info": system_info,
            "optimizations": {
                "embedding": embedding_optimization,
                "llm_inference": llm_optimization
            },
            "recommendations": {
                "current_device": resource_manager.get_device(),
                "gpu_available": resource_manager.is_gpu_available(),
                "optimal_batch_size": resource_manager.get_optimal_batch_size("embedding"),
                "worker_count": resource_manager.get_worker_count("cpu")
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting resource information: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting resource information: {str(e)}"
        )

# Documentation endpoints
@app.get(
    "/api-info",
    tags=["Documentation"],
    summary="API Information",
    description="Get comprehensive API documentation and endpoint information"
)
async def api_info():
    """Get API documentation metadata and endpoint information"""
    return {
        "api": {
            "name": "RAG Chatbot API",
            "version": "2.0.0",
            "description": "Multi-source RAG chatbot with web crawling, API integration, and caching",
            "contact": {
                "name": settings.API_CONTACT_NAME,
                "email": settings.API_CONTACT_EMAIL
            },
            "license": {
                "name": settings.API_LICENSE_NAME,
                "url": settings.API_LICENSE_URL
            }
        },
        "endpoints": {
            "chat": {
                "path": "/chat",
                "method": "POST",
                "description": "Submit questions to the RAG system",
                "authentication": "Optional"
            },
            "search": {
                "path": "/search",
                "method": "POST", 
                "description": "Search documents without AI generation",
                "authentication": "Optional"
            },
            "index": {
                "path": "/index",
                "method": "POST",
                "description": "Index documents from various sources",
                "authentication": "Optional"
            },
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Get system health status",
                "authentication": "None"
            },
            "info": {
                "path": "/info",
                "method": "GET",
                "description": "Get system information",
                "authentication": "None"
            },
            "docs": {
                "path": "/docs",
                "method": "GET",
                "description": "Interactive API documentation (Swagger UI)",
                "authentication": "None"
            },
            "redoc": {
                "path": "/redoc",
                "method": "GET",
                "description": "Alternative API documentation (ReDoc)",
                "authentication": "None"
            }
        },
        "features": [
            "Multi-source data ingestion",
            "Web crawling with crawl4ai",
            "Real-time chat with context",
            "Vector similarity search", 
            "Redis caching",
            "Health monitoring",
            "Background processing",
            "Environment-driven configuration"
        ],
        "data_sources": [
            "Local documents (PDF, TXT, MD, DOCX, HTML)",
            "Web pages via URL crawling",
            "REST API endpoints",
            "Database query results",
            "MCP (Model Context Protocol) servers"
        ],
        "models": {
            "llm_primary": settings.LLM_MODEL,
            "llm_fallback": settings.LLM_FALLBACK_MODEL,
            "embedding_primary": settings.EMBEDDING_MODEL,
            "embedding_fallback": settings.EMBEDDING_FALLBACK_MODEL
        }
    }

# Root endpoint
@app.get(
    "/",
    tags=["Documentation"],
    summary="API Root",
    description="Root endpoint with basic API information and quick links"
)
async def root():
    """Root endpoint with basic API information and navigation links"""
    return {
        "name": "🦙 RAG Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "description": "Multi-source RAG chatbot with optimized performance for low-resource environments",
        "features": [
            "📁 Local document processing",
            "🌐 Web crawling with crawl4ai", 
            "🔌 API integration",
            "🗄️ Database queries",
            "🤖 MCP protocol support",
            "⚡ Redis caching",
            "📊 Health monitoring"
        ],
        "quick_links": {
            "chat": "/chat",
            "search": "/search",
            "index": "/index", 
            "health": "/health",
            "system_info": "/info",
            "api_documentation": "/api-info",
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "models": {
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL
        },
        "getting_started": {
            "1": "Check system health at /health",
            "2": "View API documentation at /docs", 
            "3": "Index your documents via /index",
            "4": "Start chatting at /chat"
        }
    }

# System Prompt Management Endpoints
@app.get(
    "/system-prompt/templates",
    response_model=SystemPromptTemplatesResponse,
    tags=["System"],
    summary="Get System Prompt Templates",
    description="""
    Get available system prompt templates and configuration information.
    
    **Features:**
    - List all available prompt templates
    - Template descriptions and previews
    - Available template variables
    - Configuration settings
    """
)
async def get_system_prompt_templates():
    """Get available system prompt templates"""
    try:
        prompt_manager = get_prompt_manager()
        templates_info = prompt_manager.get_available_templates()
        
        # Convert to response format
        templates = {}
        for name, info in templates_info.items():
            templates[name] = SystemPromptTemplate(
                name=info["name"],
                description=info["description"],
                preview=info["preview"],
                length=info["length"],
                variables=info["variables"]
            )
        
        return SystemPromptTemplatesResponse(
            templates=templates,
            default_template="default",
            override_allowed=settings.SYSTEM_PROMPT_OVERRIDE_ALLOWED,
            max_length=settings.SYSTEM_PROMPT_MAX_LENGTH
        )
        
    except Exception as e:
        logger.error(f"Error getting system prompt templates: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system prompt templates: {str(e)}"
        )

@app.get(
    "/system-prompt",
    response_model=SystemPromptResponse,
    tags=["System"],
    summary="Get Current System Prompt",
    description="""
    Get the current default system prompt and its information.
    
    **Information Includes:**
    - Current prompt content
    - Source of the prompt (default, template, custom)
    - Prompt statistics and validation info
    - Template variables if applicable
    """
)
async def get_current_system_prompt():
    """Get current system prompt configuration"""
    try:
        prompt_manager = get_prompt_manager()
        current_prompt = prompt_manager.get_system_prompt()
        prompt_info_data = prompt_manager.get_prompt_info(current_prompt)
        
        prompt_info = SystemPromptInfo(
            content=current_prompt,
            length=prompt_info_data["length"],
            word_count=prompt_info_data["word_count"],
            variables=prompt_info_data["variables"],
            is_valid=prompt_info_data["is_valid"],
            preview=prompt_info_data["preview"]
        )
        
        return SystemPromptResponse(
            prompt=current_prompt,
            source="default",
            template_name=None,
            info=prompt_info,
            last_updated=None
        )
        
    except Exception as e:
        logger.error(f"Error getting current system prompt: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting current system prompt: {str(e)}"
        )

@app.post(
    "/system-prompt/test",
    tags=["System"],
    summary="Test System Prompt",
    description="""
    Test a system prompt without saving it. Useful for validation and preview.
    
    **Features:**
    - Validate prompt content and length
    - Preview how prompt will be processed
    - Check for template variables
    - Security validation
    """
)
async def test_system_prompt(request: SystemPromptRequest):
    """Test a system prompt for validation and preview"""
    try:
        prompt_manager = get_prompt_manager()
        
        # Validate the prompt
        is_valid = prompt_manager._validate_prompt(request.prompt)
        prompt_info_data = prompt_manager.get_prompt_info(request.prompt)
        
        return {
            "valid": is_valid,
            "info": {
                "length": prompt_info_data["length"],
                "word_count": prompt_info_data["word_count"],
                "variables": prompt_info_data["variables"],
                "preview": prompt_info_data["preview"]
            },
            "processed_prompt": prompt_manager.get_system_prompt(
                user_prompt=request.prompt,
                template_name=request.template_name
            ) if is_valid else None,
            "warnings": [] if is_valid else ["Prompt failed validation - check length and content"]
        }
        
    except Exception as e:
        logger.error(f"Error testing system prompt: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error testing system prompt: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "chatbot_api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.API_RELOAD,
        log_level=settings.API_LOG_LEVEL.lower()
    )
