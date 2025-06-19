#!/usr/bin/env python3
"""
Enhanced RAG Pipeline with multi-source support
Supports documents, web crawling, APIs, databases, and MCP
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from llama_index.core import VectorStoreIndex, Document as LlamaDocument
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import get_settings, validate_environment, logger
from connectors import (
    DocumentConnector, WebConnector, APIConnector,
    DatabaseConnector, MCPConnector
)
from utils.chunking import TextChunker
from utils.embeddings import EmbeddingManager
from utils.caching import CacheManager
from models.document import Document, DocumentMetadata

class RAGPipeline:
    """Enhanced RAG pipeline with multi-source support"""
    
    def __init__(self):
        self.settings = get_settings()
        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.cache_manager = CacheManager()
        self.connectors = {}
        self.vector_store = None
        self.index = None
        self.llama_embedding = None
        
    async def initialize(self):
        """Initialize the pipeline"""
        logger.info("Initializing RAG pipeline...")
        
        # Validate environment
        validate_environment()
        
        # Initialize components
        await self.embedding_manager.initialize()
        
        if self.settings.ENABLE_CACHING:
            await self.cache_manager.connect()
        
        # Create LlamaIndex embedding model
        self.llama_embedding = HuggingFaceEmbedding(
            model_name=self.settings.EMBEDDING_MODEL,
            device=self.settings.EMBEDDING_DEVICE
        )
        
        # Initialize connectors
        await self._initialize_connectors()
        
        # Initialize vector store
        await self._initialize_vector_store()
        
        logger.info("RAG pipeline initialized successfully")
    
    async def _initialize_connectors(self):
        """Initialize all data connectors"""
        connector_classes = {
            'documents': DocumentConnector,
            'web': WebConnector,
            'api': APIConnector,
            'database': DatabaseConnector,
            'mcp': MCPConnector
        }
        
        for name, connector_class in connector_classes.items():
            try:
                connector = connector_class()
                await connector.connect()
                self.connectors[name] = connector
                logger.info(f"Initialized {name} connector")
            except Exception as e:
                logger.error(f"Failed to initialize {name} connector: {e}")
                # Continue with other connectors
    
    async def _initialize_vector_store(self):
        """Initialize vector store and index"""
        try:
            # Create vector store with explicit async connection string
            sync_url = self.settings.DATABASE_URL
            async_url = sync_url.replace('postgresql://', 'postgresql+asyncpg://')
            
            self.vector_store = PGVectorStore.from_params(
                connection_string=sync_url,
                async_connection_string=async_url,
                table_name=self.settings.COLLECTION_NAME,
                embed_dim=self.settings.VECTOR_DIMENSION
            )
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Try to load existing index or create new one
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=self.llama_embedding
                )
                logger.info("Loaded existing vector index")
            except Exception:
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=self.llama_embedding
                )
                logger.info("Created new vector index")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def process_documents(self, source_type: str = "documents", **kwargs) -> Dict[str, Any]:
        """Process documents from specified source"""
        start_time = time.time()
        stats = {
            'source_type': source_type,
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': [],
            'processing_time': 0
        }
        
        try:
            if source_type not in self.connectors:
                raise ValueError(f"Unknown source type: {source_type}")
            
            connector = self.connectors[source_type]
            
            # Fetch documents from source
            logger.info(f"Fetching documents from {source_type} source...")
            documents = await connector.fetch_data(**kwargs)
            
            if not documents:
                logger.warning(f"No documents found from {source_type} source")
                return stats
            
            logger.info(f"Found {len(documents)} documents from {source_type}")
            
            # Process each document
            for doc in documents:
                try:
                    await self._process_single_document(doc)
                    stats['documents_processed'] += 1
                    
                    # Update chunk count (estimate)
                    estimated_chunks = len(doc.content) // self.settings.MAX_CHUNK_SIZE + 1
                    stats['chunks_created'] += estimated_chunks
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc.id}: {str(e)}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
            # Persist index
            if self.index:
                self.index.storage_context.persist()
                logger.info("Index persisted successfully")
            
            stats['processing_time'] = time.time() - start_time
            logger.info(f"Processing completed: {stats}")
            
            return stats
            
        except Exception as e:
            error_msg = f"Error processing {source_type} documents: {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            stats['processing_time'] = time.time() - start_time
            return stats
    
    async def _process_single_document(self, document: Document):
        """Process a single document into the index"""
        try:
            # Check if document already exists (if caching enabled)
            if self.settings.ENABLE_CACHING:
                cached_chunks = await self.cache_manager.get_document_chunks(document.id)
                if cached_chunks:
                    logger.debug(f"Using cached chunks for document {document.id}")
                    # Convert cached chunks to LlamaIndex nodes and add to index
                    await self._add_chunks_to_index(cached_chunks, document)
                    return
            
            # Chunk the document
            chunks = self.text_chunker.chunk_text(
                document.content,
                metadata=document.metadata.dict()
            )
            
            if not chunks:
                logger.warning(f"No chunks created for document {document.id}")
                return
            
            # Cache chunks if enabled
            if self.settings.ENABLE_CACHING:
                await self.cache_manager.set_document_chunks(document.id, chunks)
            
            # Add chunks to index
            await self._add_chunks_to_index(chunks, document)
            
            logger.debug(f"Processed document {document.id} into {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            raise
    
    async def _add_chunks_to_index(self, chunks: List[Dict[str, Any]], source_document: Document):
        """Add chunks to the vector index"""
        try:
            llama_documents = []
            
            for chunk in chunks:
                # Create LlamaIndex document
                chunk_metadata = chunk['metadata'].copy()
                chunk_metadata.update({
                    'source_document_id': source_document.id,
                    'source_type': source_document.metadata.source,
                    'title': source_document.display_title,
                    'source_display': source_document.display_source
                })
                
                llama_doc = LlamaDocument(
                    text=chunk['content'],
                    metadata=chunk_metadata
                )
                llama_documents.append(llama_doc)
            
            # Add to index in batches
            batch_size = self.settings.EMBEDDING_BATCH_SIZE
            for i in range(0, len(llama_documents), batch_size):
                batch = llama_documents[i:i + batch_size]
                
                # Insert documents into index
                for doc in batch:
                    self.index.insert(doc)
                
                # Small delay to avoid overwhelming the system
                if len(llama_documents) > batch_size:
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error adding chunks to index: {e}")
            raise
    
    async def process_web_content(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Process web content from URLs"""
        return await self.process_documents(
            source_type="web",
            urls=urls,
            **kwargs
        )
    
    async def process_api_data(self, api_configs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Process data from API endpoints"""
        return await self.process_documents(
            source_type="api", 
            api_configs=api_configs,
            **kwargs
        )
    
    async def process_database_queries(self, queries: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Process data from database queries"""
        return await self.process_documents(
            source_type="database",
            queries=queries,
            **kwargs
        )
    
    async def process_mcp_data(self, mcp_requests: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Process data from MCP servers"""
        return await self.process_documents(
            source_type="mcp",
            mcp_requests=mcp_requests,
            **kwargs
        )
    
    async def reindex_all(self) -> Dict[str, Any]:
        """Reindex all sources"""
        logger.info("Starting full reindex of all sources...")
        
        total_stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_errors': [],
            'source_stats': {},
            'total_time': 0
        }
        
        start_time = time.time()
        
        # Note: Full reindex would clear existing data
        logger.warning("Full reindex not implemented for data safety")
        
        # Process each source type
        source_types = ['documents', 'web', 'api', 'database', 'mcp']
        
        for source_type in source_types:
            if source_type in self.connectors:
                try:
                    logger.info(f"Processing {source_type} source...")
                    stats = await self.process_documents(source_type)
                    
                    total_stats['source_stats'][source_type] = stats
                    total_stats['total_documents'] += stats['documents_processed']
                    total_stats['total_chunks'] += stats['chunks_created']
                    total_stats['total_errors'].extend(stats['errors'])
                    
                except Exception as e:
                    error_msg = f"Error processing {source_type}: {str(e)}"
                    logger.error(error_msg)
                    total_stats['total_errors'].append(error_msg)
        
        total_stats['total_time'] = time.time() - start_time
        logger.info(f"Full reindex completed: {total_stats}")
        
        return total_stats
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'vector_store_initialized': self.vector_store is not None,
            'index_initialized': self.index is not None,
            'connectors_initialized': list(self.connectors.keys()),
            'embedding_model': self.settings.EMBEDDING_MODEL,
            'chunk_size': self.settings.MAX_CHUNK_SIZE,
            'cache_enabled': self.settings.ENABLE_CACHING
        }
        
        # Get vector store stats if available
        if self.vector_store:
            try:
                # This would require implementing a method to get collection stats
                stats['vector_collection'] = self.settings.COLLECTION_NAME
            except Exception as e:
                logger.error(f"Error getting vector store stats: {e}")
        
        return stats
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("Cleaning up pipeline resources...")
        
        try:
            # Disconnect connectors
            for name, connector in self.connectors.items():
                try:
                    await connector.disconnect()
                    logger.debug(f"Disconnected {name} connector")
                except Exception as e:
                    logger.error(f"Error disconnecting {name} connector: {e}")
            
            # Cleanup other components
            if self.embedding_manager:
                await self.embedding_manager.cleanup()
            
            if self.cache_manager:
                await self.cache_manager.disconnect()
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")

async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="RAG Pipeline - Multi-source document indexing")
    parser.add_argument("--source", choices=["documents", "web", "api", "database", "mcp", "all"], 
                       default="documents", help="Source type to process")
    parser.add_argument("--reindex-all", action="store_true", help="Reindex all sources")
    parser.add_argument("--urls", nargs="*", help="URLs to crawl (for web source)")
    parser.add_argument("--config-file", help="Configuration file for API/DB sources")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    try:
        await pipeline.initialize()
        
        if args.stats:
            stats = await pipeline.get_pipeline_stats()
            print("Pipeline Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        if args.reindex_all:
            logger.info("Starting full reindex...")
            stats = await pipeline.reindex_all()
        elif args.source == "web" and args.urls:
            logger.info(f"Processing web URLs: {args.urls}")
            stats = await pipeline.process_web_content(args.urls)
        elif args.source == "documents":
            logger.info("Processing local documents...")
            stats = await pipeline.process_documents("documents")
        elif args.source == "database" and args.config_file:
            logger.info("Processing database source...")
            import json
            with open(args.config_file, 'r') as f:
                config = json.load(f)
            stats = await pipeline.process_documents("database", **config)
        else:
            logger.info(f"Processing {args.source} source...")
            stats = await pipeline.process_documents(args.source)
        
        print("\nProcessing Results:")
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    
    finally:
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
