-- DTSEN RAG AI - Sample Test Database Schema
-- Complete DDL schema for testing the RAG system
-- Compatible with PostgreSQL 16 + pgvector extension

-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the main vector storage table for RAG documents
-- Table name matches COLLECTION_NAME from config.py: "data_rag_kb"
-- Vector dimension matches VECTOR_DIMENSION from config.py: 384 (all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS data_rag_kb (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for optimal vector similarity search performance
-- IVFFlat index for cosine similarity (most common for RAG applications)
CREATE INDEX IF NOT EXISTS data_rag_kb_embedding_cosine_idx 
ON data_rag_kb USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- IVFFlat index for L2 distance (alternative similarity metric)
CREATE INDEX IF NOT EXISTS data_rag_kb_embedding_l2_idx 
ON data_rag_kb USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- GIN index on metadata for fast metadata filtering
CREATE INDEX IF NOT EXISTS data_rag_kb_metadata_idx 
ON data_rag_kb USING GIN (metadata);

-- B-tree index on created_at for temporal queries
CREATE INDEX IF NOT EXISTS data_rag_kb_created_at_idx 
ON data_rag_kb (created_at);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at on row changes
CREATE TRIGGER update_data_rag_kb_updated_at 
    BEFORE UPDATE ON data_rag_kb 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create system metadata table
CREATE TABLE IF NOT EXISTS rag_system_info (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert system configuration information
INSERT INTO rag_system_info (key, value) VALUES 
    ('vector_dimension', '384'),
    ('embedding_model', '"sentence-transformers/all-MiniLM-L6-v2"'),
    ('collection_name', '"data_rag_kb"'),
    ('schema_version', '1.0.0'),
    ('created_by', '"DTSEN RAG AI Test Suite"')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = CURRENT_TIMESTAMP;

-- Insert sample test data matching the test documents in /tests/data/documents/
INSERT INTO data_rag_kb (text, metadata, embedding) VALUES 
(
    'DTSEN (Satu Data) is Indonesia''s national data governance initiative to create a unified, integrated, and interoperable data ecosystem across government institutions.',
    jsonb_build_object(
        'source', 'DTSEN_Overview.md',
        'source_type', 'documents',
        'title', 'DTSEN Overview',
        'chunk_id', 'chunk_dtsen_001',
        'source_document_id', 'doc_dtsen_overview',
        'file_path', '/app/tests/data/documents/DTSEN_Overview.md',
        'file_type', '.md'
    ),
    array_fill(0.1, ARRAY[384])::vector
),
(
    'The DTSEN initiative aims to eliminate data silos, improve data quality, and enable evidence-based decision making through standardized data sharing protocols.',
    jsonb_build_object(
        'source', 'DTSEN_Strategy.md',
        'source_type', 'documents',
        'title', 'DTSEN Strategic Framework',
        'chunk_id', 'chunk_dtsen_002',
        'source_document_id', 'doc_dtsen_strategy',
        'file_path', '/app/tests/data/documents/DTSEN_Strategy.md',
        'file_type', '.md'
    ),
    array_fill(0.15, ARRAY[384])::vector
),
(
    'DTSEN services include data cataloging, metadata management, API gateway services, and real-time data synchronization capabilities for government agencies.',
    jsonb_build_object(
        'source', 'DTSEN_Services.txt',
        'source_type', 'documents',
        'title', 'DTSEN Services Catalog',
        'chunk_id', 'chunk_dtsen_003',
        'source_document_id', 'doc_dtsen_services',
        'file_path', '/app/tests/data/documents/DTSEN_Services.txt',
        'file_type', '.txt'
    ),
    array_fill(0.2, ARRAY[384])::vector
),
(
    'The technology stack of DTSEN includes cloud infrastructure, microservices architecture, containerization with Docker and Kubernetes, and modern API management platforms.',
    jsonb_build_object(
        'source', 'DTSEN_Technology.html',
        'source_type', 'documents',
        'title', 'DTSEN Technology Stack',
        'chunk_id', 'chunk_dtsen_004',
        'source_document_id', 'doc_dtsen_technology',
        'file_path', '/app/tests/data/documents/DTSEN_Technology.html',
        'file_type', '.html'
    ),
    array_fill(0.25, ARRAY[384])::vector
),
(
    'Presidential Instruction Number 4 of 2025 mandates the implementation of DTSEN across all government institutions to enhance digital governance and public service delivery.',
    jsonb_build_object(
        'source', 'Inpres Nomor 4 Tahun 2025.pdf',
        'source_type', 'documents',
        'title', 'Presidential Instruction No. 4/2025',
        'chunk_id', 'chunk_dtsen_005',
        'source_document_id', 'doc_inpres_2025',
        'file_path', '/app/tests/data/documents/Inpres Nomor 4 Tahun 2025.pdf',
        'file_type', '.pdf'
    ),
    array_fill(0.3, ARRAY[384])::vector
),
(
    'Retrieval-Augmented Generation (RAG) combines information retrieval with natural language generation to provide contextually accurate responses based on document collections.',
    jsonb_build_object(
        'source', 'https://example.com/rag-guide',
        'source_type', 'web',
        'title', 'RAG Technology Guide',
        'chunk_id', 'chunk_web_001',
        'source_document_id', 'web_rag_guide',
        'url', 'https://example.com/rag-guide'
    ),
    array_fill(0.35, ARRAY[384])::vector
),
(
    'PostgreSQL with pgvector extension provides efficient storage and similarity search capabilities for high-dimensional embedding vectors in RAG applications.',
    jsonb_build_object(
        'source', 'api_endpoint',
        'source_type', 'api',
        'title', 'Vector Database Documentation',
        'chunk_id', 'chunk_api_001',
        'source_document_id', 'api_vector_docs',
        'endpoint', '/api/vector-db/docs',
        'api_url', 'https://jsonplaceholder.typicode.com/posts/1'
    ),
    array_fill(0.4, ARRAY[384])::vector
),
(
    'The all-MiniLM-L6-v2 sentence transformer model generates 384-dimensional embeddings optimized for semantic similarity search and document retrieval tasks.',
    jsonb_build_object(
        'source', 'database_query',
        'source_type', 'database',
        'title', 'Embedding Model Specifications',
        'chunk_id', 'chunk_db_001',
        'source_document_id', 'db_embeddings_spec',
        'database_table', 'model_specifications',
        'database_query', 'SELECT * FROM models WHERE type = ''embedding'''
    ),
    array_fill(0.45, ARRAY[384])::vector
),
(
    'Model Context Protocol (MCP) enables standardized communication between AI models and external data sources for enhanced retrieval capabilities.',
    jsonb_build_object(
        'source', 'mcp_endpoint',
        'source_type', 'mcp',
        'title', 'MCP Protocol Overview',
        'chunk_id', 'chunk_mcp_001',
        'source_document_id', 'mcp_protocol_overview',
        'mcp_endpoint', 'https://jsonplaceholder.typicode.com/posts/2'
    ),
    array_fill(0.5, ARRAY[384])::vector
),
(
    'FastAPI provides asynchronous REST API capabilities with automatic OpenAPI documentation generation for the DTSEN RAG chatbot system architecture.',
    jsonb_build_object(
        'source', 'system_architecture.md',
        'source_type', 'documents',
        'title', 'System Architecture Overview',
        'chunk_id', 'chunk_arch_001',
        'source_document_id', 'doc_system_arch',
        'file_path', '/app/docs/architecture.md',
        'file_type', '.md'
    ),
    array_fill(0.55, ARRAY[384])::vector
);

-- Create a view for test statistics
CREATE OR REPLACE VIEW test_vector_store_stats AS
SELECT 
    COUNT(*) as total_test_vectors,
    AVG(LENGTH(text)) as avg_text_length,
    MIN(created_at) as first_test_document,
    MAX(created_at) as last_test_document,
    COUNT(DISTINCT metadata->>'source_type') as unique_source_types,
    COUNT(DISTINCT metadata->>'source') as unique_sources,
    pg_size_pretty(pg_total_relation_size('data_rag_kb')) as table_size
FROM data_rag_kb;

-- Test validation queries

-- 1. Verify vector dimensions are correct (should all be 384)
SELECT 
    'Vector Dimension Check' as test_name,
    COUNT(*) as total_vectors,
    COUNT(CASE WHEN vector_dims(embedding) = 384 THEN 1 END) as correct_dimension,
    COUNT(CASE WHEN vector_dims(embedding) != 384 THEN 1 END) as incorrect_dimension,
    CASE WHEN COUNT(CASE WHEN vector_dims(embedding) != 384 THEN 1 END) = 0 THEN 'PASS' ELSE 'FAIL' END as status
FROM data_rag_kb;

-- 2. Verify all source types are represented
SELECT 
    'Source Type Coverage' as test_name,
    STRING_AGG(DISTINCT metadata->>'source_type', ', ' ORDER BY metadata->>'source_type') as available_source_types,
    COUNT(DISTINCT metadata->>'source_type') as source_type_count,
    CASE WHEN COUNT(DISTINCT metadata->>'source_type') >= 5 THEN 'PASS' ELSE 'FAIL' END as status
FROM data_rag_kb;

-- 3. Test cosine similarity search (typical RAG query)
SELECT 
    'Similarity Search Test' as test_name,
    COUNT(*) as results_found,
    AVG(1 - (embedding <=> array_fill(0.2, ARRAY[384])::vector)) as avg_similarity,
    CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END as status
FROM (
    SELECT embedding
    FROM data_rag_kb
    ORDER BY embedding <=> array_fill(0.2, ARRAY[384])::vector
    LIMIT 5
) similarity_test;

-- 4. Test metadata filtering (RAG system requirement)
SELECT 
    'Metadata Filtering Test' as test_name,
    COUNT(*) as document_source_count,
    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'documents') as documents_count,
    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'web') as web_count,
    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'api') as api_count,
    CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END as status
FROM data_rag_kb;

-- 5. Test system configuration
SELECT 
    'System Configuration Test' as test_name,
    COUNT(*) as config_entries,
    COUNT(*) FILTER (WHERE key = 'vector_dimension' AND value::text = '384') as dimension_ok,
    COUNT(*) FILTER (WHERE key = 'embedding_model') as model_configured,
    CASE WHEN COUNT(*) >= 3 THEN 'PASS' ELSE 'FAIL' END as status
FROM rag_system_info;

-- Test data summary for verification
SELECT 
    '=== TEST DATA SUMMARY ===' as summary,
    (SELECT COUNT(*) FROM data_rag_kb) as total_documents,
    (SELECT COUNT(DISTINCT metadata->>'source_type') FROM data_rag_kb) as source_types,
    (SELECT DISTINCT vector_dims(embedding) FROM data_rag_kb LIMIT 1) as vector_dimension,
    (SELECT pg_size_pretty(pg_total_relation_size('data_rag_kb'))) as table_size;

-- Display final test summary
DO $$
DECLARE
    doc_count INTEGER;
    source_types INTEGER;
    vector_dim INTEGER;
BEGIN
    SELECT COUNT(*), COUNT(DISTINCT metadata->>'source_type'), vector_dims(embedding)
    INTO doc_count, source_types, vector_dim
    FROM data_rag_kb LIMIT 1;
    
    RAISE NOTICE '=== DTSEN RAG AI Test Database Setup Complete ===';
    RAISE NOTICE 'Test documents loaded: %', doc_count;
    RAISE NOTICE 'Source types available: %', source_types;
    RAISE NOTICE 'Vector dimension verified: %', vector_dim;
    RAISE NOTICE 'Tables created: data_rag_kb, rag_system_info';
    RAISE NOTICE 'Indexes created: cosine, l2, metadata, timestamp';
    RAISE NOTICE 'Ready for RAG system testing!';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Run: SELECT * FROM test_vector_store_stats;';
    RAISE NOTICE '2. Test similarity: SELECT text, metadata FROM data_rag_kb ORDER BY embedding <=> array_fill(0.2, ARRAY[384])::vector LIMIT 3;';
    RAISE NOTICE '3. Start RAG system with: docker-compose up -d';
END $$;