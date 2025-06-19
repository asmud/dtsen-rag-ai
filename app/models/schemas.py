from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json

class SourceType(str, Enum):
    """Enumeration of data source types"""
    DOCUMENTS = "documents"
    WEB = "web"
    API = "api"
    DATABASE = "database"
    MCP = "mcp"

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=10000, 
        description="User question or query",
        example="What is the main topic discussed in the uploaded documents?"
    )
    conversation_id: Optional[str] = Field(
        None, 
        description="Optional conversation ID for maintaining chat context",
        example="conv_12345"
    )
    include_sources: bool = Field(
        True, 
        description="Whether to include source documents in the response",
        example=True
    )
    max_tokens: Optional[int] = Field(
        None, 
        ge=1, 
        le=2048, 
        description="Maximum number of tokens in the response",
        example=512
    )
    temperature: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=2.0, 
        description="Temperature for response generation (0.0 = deterministic, 2.0 = very creative)",
        example=0.1
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional filters for document search",
        example={"source": "web", "date_after": "2023-01-01"}
    )
    system_prompt: Optional[str] = Field(
        None,
        max_length=2000,
        description="Custom system prompt to override the default (if allowed by configuration)",
        example="You are a specialized technical assistant. Focus on providing detailed technical explanations."
    )
    system_prompt_template: Optional[str] = Field(
        None,
        description="Name of system prompt template to use",
        example="technical_expert"
    )
    system_prompt_variables: Optional[Dict[str, Any]] = Field(
        None,
        description="Variables to substitute in system prompt template",
        example={"user_expertise": "beginner", "domain": "machine learning"}
    )

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError('System prompt cannot be empty if provided')
            if len(v) > 2000:
                raise ValueError('System prompt exceeds maximum length of 2000 characters')
        return v.strip() if v else None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the key features of the RAG system?",
                "conversation_id": "conv_12345",
                "include_sources": True,
                "max_tokens": 512,
                "temperature": 0.1,
                "filters": {"source": "documents"},
                "system_prompt_template": "technical_expert",
                "system_prompt_variables": {"domain": "AI/ML"}
            }
        }
    }

class SourceDocument(BaseModel):
    """Model for source document in response"""
    id: str = Field(..., description="Unique document identifier", example="doc_12345")
    title: Optional[str] = Field(None, description="Document title", example="Introduction to RAG Systems")
    content: str = Field(..., description="Document content excerpt", example="Retrieval-Augmented Generation combines...")
    source: str = Field(..., description="Source type", example="documents")
    url: Optional[str] = Field(None, description="Source URL if applicable", example="https://example.com/doc.pdf")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score", example=0.85)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata", example={"file_type": ".pdf", "created_at": "2023-01-01"})

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Generated response to the user's question", example="Based on the documents, RAG systems combine retrieval and generation...")
    conversation_id: Optional[str] = Field(None, description="Conversation ID", example="conv_12345")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used for generating the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata including processing time", example={"processing_time_ms": 150, "model_used": "gemma2:2b"})
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Based on the documents, RAG systems combine retrieval and generation to provide accurate, contextual responses.",
                "conversation_id": "conv_12345",
                "sources": [
                    {
                        "id": "doc_12345",
                        "title": "Introduction to RAG",
                        "content": "RAG systems combine retrieval...",
                        "source": "documents",
                        "similarity_score": 0.85,
                        "metadata": {"file_type": ".pdf"}
                    }
                ],
                "metadata": {
                    "processing_time_ms": 150,
                    "model_used": "gemma2:2b",
                    "source_count": 1
                },
                "timestamp": "2023-01-01T12:00:00"
            }
        }
    }

class IndexRequest(BaseModel):
    """Request model for indexing endpoint"""
    source_type: SourceType = Field(..., description="Type of data source")
    source_config: Dict[str, Any] = Field(..., description="Source-specific configuration")
    force_reindex: bool = Field(False, description="Force reindexing of existing documents")
    chunk_size: Optional[int] = Field(None, ge=100, le=2000, description="Custom chunk size")
    batch_size: Optional[int] = Field(None, ge=1, le=100, description="Batch processing size")

class IndexResponse(BaseModel):
    """Response model for indexing endpoint"""
    task_id: str = Field(..., description="Background task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(0, description="Number of documents processed")
    chunks_created: int = Field(0, description="Number of chunks created")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")

class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"
    UNKNOWN = "unknown"

class ComponentHealth(BaseModel):
    """Health information for a system component"""
    status: HealthStatus = Field(..., description="Component health status")
    message: Optional[str] = Field(None, description="Health status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check time")

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    overall_status: HealthStatus = Field(..., description="Overall system health")
    components: Dict[str, ComponentHealth] = Field(..., description="Individual component health")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")

class ConnectorInfo(BaseModel):
    """Information about a data connector"""
    name: str = Field(..., description="Connector name")
    type: str = Field(..., description="Connector type")
    status: str = Field(..., description="Connection status")
    config: Optional[Dict[str, Any]] = Field(None, description="Connector configuration")
    stats: Optional[Dict[str, Any]] = Field(None, description="Connector statistics")

class SystemInfo(BaseModel):
    """System information response"""
    version: str = Field(..., description="Application version")
    connectors: List[ConnectorInfo] = Field(..., description="Available connectors")
    models: Dict[str, str] = Field(..., description="Configured models")
    settings: Dict[str, Any] = Field(..., description="System settings")
    capabilities: List[str] = Field(..., description="System capabilities")

class SearchRequest(BaseModel):
    """Request model for document search"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    include_content: bool = Field(True, description="Include document content in results")

class SearchResult(BaseModel):
    """Individual search result"""
    id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    content: Optional[str] = Field(None, description="Document content")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    source: str = Field(..., description="Document source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of matching documents")
    query: str = Field(..., description="Original search query")
    processing_time_ms: float = Field(..., description="Query processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class ConfigurationRequest(BaseModel):
    """Request to update system configuration"""
    settings: Dict[str, Any] = Field(..., description="Settings to update")
    restart_required: bool = Field(False, description="Whether restart is required")

class ConfigurationResponse(BaseModel):
    """Response for configuration update"""
    success: bool = Field(..., description="Whether update was successful")
    updated_settings: List[str] = Field(..., description="List of updated setting keys")
    restart_required: bool = Field(..., description="Whether restart is required")
    errors: List[str] = Field(default_factory=list, description="Configuration errors")

class TaskStatus(str, Enum):
    """Background task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskInfo(BaseModel):
    """Background task information"""
    task_id: str = Field(..., description="Task identifier")
    task_type: str = Field(..., description="Type of task")
    status: TaskStatus = Field(..., description="Task status")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Task progress percentage")
    result: Optional[Any] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Task creation time")
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")

class TaskResponse(BaseModel):
    """Response for task status query"""
    task: TaskInfo = Field(..., description="Task information")
    
class BatchTaskResponse(BaseModel):
    """Response for multiple task status query"""
    tasks: List[TaskInfo] = Field(..., description="List of tasks")
    total_tasks: int = Field(..., description="Total number of tasks")

# Web crawler specific models
class CrawlRequest(BaseModel):
    """Request for web crawling"""
    urls: List[str] = Field(..., min_items=1, description="URLs to crawl")
    max_depth: int = Field(1, ge=1, le=5, description="Maximum crawl depth")
    max_pages: int = Field(10, ge=1, le=100, description="Maximum pages to crawl")
    respect_robots: bool = Field(True, description="Respect robots.txt")
    rate_limit: float = Field(1.0, ge=0.1, le=10.0, description="Requests per second")

# API connector specific models
class APISourceConfig(BaseModel):
    """Configuration for API data source"""
    url: str = Field(..., description="API endpoint URL")
    method: str = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    data: Optional[Dict[str, Any]] = Field(None, description="Request body data")
    auth: Optional[Dict[str, str]] = Field(None, description="Authentication config")
    response_path: Optional[str] = Field(None, description="JSON path to extract data")
    content_fields: List[str] = Field(default_factory=list, description="Fields containing content")

# Database connector specific models
class DatabaseQueryConfig(BaseModel):
    """Configuration for database query"""
    query: str = Field(..., description="SQL query")
    params: List[Any] = Field(default_factory=list, description="Query parameters")
    id_column: str = Field("id", description="Column to use as document ID")
    content_columns: List[str] = Field(default_factory=list, description="Columns containing content")
    title_column: Optional[str] = Field(None, description="Column containing title")
    metadata_columns: List[str] = Field(default_factory=list, description="Columns to include as metadata")

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")

# System Prompt Models
class SystemPromptInfo(BaseModel):
    """Information about a system prompt"""
    content: str = Field(..., description="System prompt content")
    length: int = Field(..., description="Character length of prompt")
    word_count: int = Field(..., description="Word count of prompt")
    variables: List[str] = Field(default_factory=list, description="Template variables found in prompt")
    is_valid: bool = Field(..., description="Whether the prompt passes validation")
    preview: str = Field(..., description="Preview of the prompt")

class SystemPromptTemplate(BaseModel):
    """System prompt template information"""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    preview: str = Field(..., description="Preview of template content")
    length: int = Field(..., description="Template character length")
    variables: List[str] = Field(default_factory=list, description="Available template variables")

class SystemPromptTemplatesResponse(BaseModel):
    """Response with available system prompt templates"""
    templates: Dict[str, SystemPromptTemplate] = Field(..., description="Available templates")
    default_template: str = Field(..., description="Name of default template")
    override_allowed: bool = Field(..., description="Whether custom prompts are allowed")
    max_length: int = Field(..., description="Maximum allowed prompt length")

class SystemPromptRequest(BaseModel):
    """Request to update system prompt"""
    prompt: str = Field(..., max_length=2000, description="New system prompt content")
    template_name: Optional[str] = Field(None, description="Optional template name")
    description: Optional[str] = Field(None, description="Description of the prompt")

class SystemPromptResponse(BaseModel):
    """Response with current system prompt"""
    prompt: str = Field(..., description="Current system prompt")
    source: str = Field(..., description="Source of prompt (default, template, custom)")
    template_name: Optional[str] = Field(None, description="Template name if applicable")
    info: SystemPromptInfo = Field(..., description="Prompt information")
    last_updated: Optional[datetime] = Field(None, description="When prompt was last updated")