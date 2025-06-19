import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiofiles
import logging
from datetime import datetime

from connectors.base_connector import BaseConnector
from models.document import Document, DocumentMetadata
from utils.chunking import TextChunker

logger = logging.getLogger(__name__)

class DocumentConnector(BaseConnector):
    """Connector for local document files"""
    
    def __init__(self):
        super().__init__()
        self.data_dir = Path(self.settings.DATA_DIR)
        self.supported_extensions = self.settings.SUPPORTED_EXTENSIONS
        self.max_file_size = self.settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.chunker = TextChunker()
        self._connected = False
    
    async def connect(self) -> bool:
        """Check if data directory exists and is accessible"""
        try:
            if not self.data_dir.exists():
                logger.warning(f"Data directory {self.data_dir} does not exist, creating it")
                self.data_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.data_dir.is_dir():
                logger.error(f"Data path {self.data_dir} is not a directory")
                return False
            
            # Check if directory is readable
            if not os.access(self.data_dir, os.R_OK):
                logger.error(f"Data directory {self.data_dir} is not readable")
                return False
            
            self._connected = True
            logger.info(f"Connected to document source: {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to document source: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from document source"""
        self._connected = False
        return True
    
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch documents from the data directory"""
        if not self._connected:
            await self.connect()
        
        documents = []
        file_paths = kwargs.get('file_paths', None)
        
        try:
            if file_paths:
                # Process specific files
                for file_path in file_paths:
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        doc = await self._process_file(path)
                        if doc:
                            documents.append(doc)
            else:
                # Process all files in data directory
                for file_path in self._get_all_files():
                    doc = await self._process_file(file_path)
                    if doc:
                        documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} documents from local files")
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []
    
    def _get_all_files(self) -> List[Path]:
        """Get all supported files from data directory"""
        files = []
        for ext in self.supported_extensions:
            files.extend(self.data_dir.rglob(f"*{ext}"))
        return files
    
    async def _process_file(self, file_path: Path) -> Optional[Document]:
        """Process a single file and return Document"""
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                logger.warning(f"File {file_path} exceeds maximum size limit")
                return None
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.debug(f"Skipping unsupported file: {file_path}")
                return None
            
            # Read file content
            content = await self._read_file_content(file_path)
            if not content:
                logger.warning(f"Empty or unreadable file: {file_path}")
                return None
            
            # Create document metadata
            metadata = DocumentMetadata(
                source="local_file",
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_path.stat().st_size,
                file_type=file_path.suffix.lower(),
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                modified_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                processed_at=datetime.now()
            )
            
            # Create document
            document = Document(
                id=str(hash(f"{file_path}_{file_path.stat().st_mtime}")),
                content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    async def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content based on file type"""
        try:
            if file_path.suffix.lower() == '.pdf':
                return await self._read_pdf(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md', '.html']:
                return await self._read_text_file(file_path)
            elif file_path.suffix.lower() == '.docx':
                return await self._read_docx(file_path)
            else:
                # Try to read as text
                return await self._read_text_file(file_path)
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    async def _read_text_file(self, file_path: Path) -> str:
        """Read text-based files"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return await f.read()
    
    async def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file content"""
        # Note: This is a placeholder - in production, you'd use PyPDF2 or similar
        logger.warning(f"PDF reading not implemented for {file_path}")
        return f"PDF content from {file_path.name} (not implemented)"
    
    async def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file content"""
        # Note: This is a placeholder - in production, you'd use python-docx
        logger.warning(f"DOCX reading not implemented for {file_path}")
        return f"DOCX content from {file_path.name} (not implemented)"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of document connector"""
        try:
            if not self.data_dir.exists():
                return {
                    "status": "unhealthy",
                    "message": f"Data directory {self.data_dir} does not exist"
                }
            
            file_count = len(self._get_all_files())
            total_size = sum(f.stat().st_size for f in self._get_all_files())
            
            return {
                "status": "healthy",
                "data_directory": str(self.data_dir),
                "file_count": file_count,
                "total_size_bytes": total_size,
                "supported_extensions": self.supported_extensions,
                "max_file_size_mb": self.settings.MAX_FILE_SIZE_MB
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Return connector metadata"""
        return {
            "name": "DocumentConnector",
            "type": "local_files",
            "status": "connected" if self._connected else "disconnected",
            "data_directory": str(self.data_dir),
            "supported_extensions": self.supported_extensions
        }