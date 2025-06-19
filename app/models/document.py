from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class DocumentSource(str, Enum):
    """Document source types"""
    LOCAL_FILE = "local_file"
    WEB_CRAWL = "web_crawl"
    API = "api"
    DATABASE = "database"
    MCP = "mcp"

class DocumentMetadata(BaseModel):
    """Metadata for documents"""
    
    # Source information
    source: DocumentSource = Field(..., description="Document source type")
    title: Optional[str] = Field(None, description="Document title")
    
    # File-related metadata
    file_path: Optional[str] = Field(None, description="Original file path")
    file_name: Optional[str] = Field(None, description="File name")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    file_type: Optional[str] = Field(None, description="File extension or MIME type")
    
    # URL-related metadata
    url: Optional[str] = Field(None, description="Source URL")
    
    # API-related metadata
    api_endpoint: Optional[str] = Field(None, description="API endpoint URL")
    
    # Database-related metadata
    database_table: Optional[str] = Field(None, description="Database table name")
    database_query: Optional[str] = Field(None, description="SQL query used")
    
    # MCP-related metadata
    mcp_endpoint: Optional[str] = Field(None, description="MCP endpoint")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Document creation time")
    modified_at: Optional[datetime] = Field(None, description="Document modification time")
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    
    # Content metadata
    content_length: Optional[int] = Field(None, ge=0, description="Content length in characters")
    language: Optional[str] = Field(None, description="Detected language")
    
    # Processing metadata
    chunk_count: Optional[int] = Field(None, ge=0, description="Number of chunks created")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    
    # Additional metadata
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('file_type')
    def normalize_file_type(cls, v):
        if v and not v.startswith('.'):
            return f'.{v.lower()}'
        return v.lower() if v else v

class Document(BaseModel):
    """Document model for RAG system"""
    
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Vector embedding (optional, computed separately)
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    
    # Computed properties
    @property
    def source_type(self) -> DocumentSource:
        """Get the document source type"""
        return self.metadata.source
    
    @property
    def display_title(self) -> str:
        """Get display title, falling back to filename or ID"""
        if self.metadata.title:
            return self.metadata.title
        elif self.metadata.file_name:
            return self.metadata.file_name
        elif self.metadata.url:
            from urllib.parse import urlparse
            return urlparse(self.metadata.url).path.split('/')[-1] or self.metadata.url
        else:
            return f"Document {self.id[:8]}"
    
    @property
    def display_source(self) -> str:
        """Get human-readable source description"""
        if self.metadata.source == DocumentSource.LOCAL_FILE:
            return f"File: {self.metadata.file_name or 'Unknown'}"
        elif self.metadata.source == DocumentSource.WEB_CRAWL:
            return f"Web: {self.metadata.url or 'Unknown URL'}"
        elif self.metadata.source == DocumentSource.API:
            return f"API: {self.metadata.api_endpoint or 'Unknown endpoint'}"
        elif self.metadata.source == DocumentSource.DATABASE:
            return f"DB: {self.metadata.database_table or 'Unknown table'}"
        elif self.metadata.source == DocumentSource.MCP:
            return f"MCP: {self.metadata.mcp_endpoint or 'Unknown endpoint'}"
        else:
            return str(self.metadata.source)
    
    def to_search_result(self, similarity_score: float = 0.0, content_preview_length: int = 200) -> Dict[str, Any]:
        """Convert document to search result format"""
        content_preview = self.content[:content_preview_length]
        if len(self.content) > content_preview_length:
            content_preview += "..."
        
        return {
            "id": self.id,
            "title": self.display_title,
            "content": content_preview,
            "similarity_score": similarity_score,
            "source": self.display_source,
            "metadata": {
                "source_type": self.metadata.source,
                "processed_at": self.metadata.processed_at.isoformat(),
                "content_length": len(self.content),
                **self.metadata.extra_metadata
            }
        }
    
    def to_source_document(self, similarity_score: float = 0.0, content_excerpt_length: int = 300) -> Dict[str, Any]:
        """Convert document to source document format for chat responses"""
        content_excerpt = self.content[:content_excerpt_length]
        if len(self.content) > content_excerpt_length:
            content_excerpt += "..."
        
        return {
            "id": self.id,
            "title": self.display_title,
            "content": content_excerpt,
            "source": self.metadata.source,
            "url": self.metadata.url,
            "similarity_score": similarity_score,
            "metadata": {
                "file_name": self.metadata.file_name,
                "file_type": self.metadata.file_type,
                "processed_at": self.metadata.processed_at.isoformat(),
                "content_length": len(self.content)
            }
        }
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v
    
    @validator('id')
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError('ID cannot be empty')
        return v.strip()

class DocumentChunk(BaseModel):
    """Chunk of a document with metadata"""
    
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., min_length=1, description="Chunk content")
    chunk_index: int = Field(..., ge=0, description="Chunk index within document")
    
    # Chunk-specific metadata
    chunk_size: int = Field(..., ge=1, description="Chunk size in characters")
    overlap_with_previous: int = Field(0, ge=0, description="Overlap with previous chunk")
    overlap_with_next: int = Field(0, ge=0, description="Overlap with next chunk")
    
    # Position within document
    start_position: Optional[int] = Field(None, ge=0, description="Start position in original document")
    end_position: Optional[int] = Field(None, ge=0, description="End position in original document")
    
    # Vector embedding
    embedding: Optional[List[float]] = Field(None, description="Chunk embedding vector")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Chunk creation time")
    
    # Additional metadata from parent document
    document_metadata: Optional[DocumentMetadata] = Field(None, description="Parent document metadata")
    
    @property
    def display_title(self) -> str:
        """Get display title for chunk"""
        if self.document_metadata and self.document_metadata.title:
            return f"{self.document_metadata.title} (Chunk {self.chunk_index + 1})"
        else:
            return f"Chunk {self.chunk_index + 1} of Document {self.document_id[:8]}"
    
    def to_search_result(self, similarity_score: float = 0.0) -> Dict[str, Any]:
        """Convert chunk to search result format"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "title": self.display_title,
            "content": self.content,
            "similarity_score": similarity_score,
            "chunk_index": self.chunk_index,
            "metadata": {
                "chunk_size": self.chunk_size,
                "start_position": self.start_position,
                "end_position": self.end_position,
                "created_at": self.created_at.isoformat()
            }
        }
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Chunk content cannot be empty')
        return v
    
    @validator('end_position')
    def validate_positions(cls, v, values):
        start_pos = values.get('start_position')
        if start_pos is not None and v is not None and v <= start_pos:
            raise ValueError('End position must be greater than start position')
        return v

class DocumentCollection(BaseModel):
    """Collection of documents with metadata"""
    
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    documents: List[Document] = Field(default_factory=list, description="Documents in collection")
    
    # Collection metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Collection creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Collection update time")
    
    # Statistics
    total_documents: int = Field(0, ge=0, description="Total number of documents")
    total_chunks: int = Field(0, ge=0, description="Total number of chunks")
    total_size_bytes: int = Field(0, ge=0, description="Total content size")
    
    def add_document(self, document: Document):
        """Add document to collection"""
        self.documents.append(document)
        self.total_documents = len(self.documents)
        self.total_size_bytes += len(document.content)
        self.updated_at = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_size_bytes": 0,
                "avg_document_size": 0,
                "source_breakdown": {}
            }
        
        source_counts = {}
        total_size = 0
        
        for doc in self.documents:
            source = doc.metadata.source
            source_counts[source] = source_counts.get(source, 0) + 1
            total_size += len(doc.content)
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": self.total_chunks,
            "total_size_bytes": total_size,
            "avg_document_size": total_size / len(self.documents),
            "source_breakdown": source_counts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }