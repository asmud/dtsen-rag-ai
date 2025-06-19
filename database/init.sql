-- DTSEN RAG AI - PostgreSQL Database Initialization Script
-- This script creates the database schema that matches LlamaIndex's PGVectorStore structure
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

-- Create additional tables for system metadata (optional)
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
    ('created_by', '"DTSEN RAG AI System"')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = CURRENT_TIMESTAMP;

-- Create a view for easy vector store statistics
CREATE OR REPLACE VIEW vector_store_stats AS
SELECT 
    COUNT(*) as total_vectors,
    AVG(LENGTH(text)) as avg_text_length,
    MIN(created_at) as first_document,
    MAX(created_at) as last_document,
    COUNT(DISTINCT metadata->>'source') as unique_sources,
    COUNT(DISTINCT metadata->>'source_type') as unique_source_types,
    pg_size_pretty(pg_total_relation_size('data_rag_kb')) as table_size
FROM data_rag_kb;

-- Function to check vector store health
CREATE OR REPLACE FUNCTION check_vector_store_health()
RETURNS TABLE (
    component TEXT,
    status TEXT,
    message TEXT,
    details JSONB
) AS $$
BEGIN
    -- Check pgvector extension
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RETURN QUERY SELECT 
            'pgvector_extension'::TEXT,
            'healthy'::TEXT,
            'pgvector extension is installed'::TEXT,
            jsonb_build_object('version', (SELECT extversion FROM pg_extension WHERE extname = 'vector'));
    ELSE
        RETURN QUERY SELECT 
            'pgvector_extension'::TEXT,
            'unhealthy'::TEXT,
            'pgvector extension is not installed'::TEXT,
            '{}'::JSONB;
    END IF;

    -- Check main table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'data_rag_kb') THEN
        RETURN QUERY SELECT 
            'main_table'::TEXT,
            'healthy'::TEXT,
            'Main vector table exists'::TEXT,
            (SELECT jsonb_build_object(
                'row_count', COUNT(*),
                'has_vectors', COUNT(*) > 0,
                'avg_vector_dim', CASE WHEN COUNT(*) > 0 THEN vector_dims(embedding) ELSE 0 END
            ) FROM data_rag_kb LIMIT 1);
    ELSE
        RETURN QUERY SELECT 
            'main_table'::TEXT,
            'unhealthy'::TEXT,
            'Main vector table does not exist'::TEXT,
            '{}'::JSONB;
    END IF;

    -- Check indexes
    IF EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'data_rag_kb') THEN
        RETURN QUERY SELECT 
            'indexes'::TEXT,
            'healthy'::TEXT,
            'Vector indexes exist'::TEXT,
            (SELECT jsonb_agg(indexname) FROM pg_indexes WHERE tablename = 'data_rag_kb');
    ELSE
        RETURN QUERY SELECT 
            'indexes'::TEXT,
            'degraded'::TEXT,
            'No indexes found for main table'::TEXT,
            '{}'::JSONB;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Grant appropriate permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON data_rag_kb TO rag_user;
-- GRANT SELECT ON vector_store_stats TO rag_user;
-- GRANT EXECUTE ON FUNCTION check_vector_store_health() TO rag_user;

-- Display initialization summary
DO $$
BEGIN
    RAISE NOTICE '=== DTSEN RAG AI Database Initialization Complete ===';
    RAISE NOTICE 'Vector table: data_rag_kb';
    RAISE NOTICE 'Vector dimension: 384';
    RAISE NOTICE 'Embedding model: sentence-transformers/all-MiniLM-L6-v2';
    RAISE NOTICE 'Indexes created for optimal vector search performance';
    RAISE NOTICE 'Use SELECT * FROM check_vector_store_health(); for health checks';
    RAISE NOTICE 'Use SELECT * FROM vector_store_stats; for statistics';
END $$;