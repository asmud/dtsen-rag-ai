# DTSEN RAG AI - Database Schema Documentation

This directory contains the SQL DDL statements and database setup scripts for the DTSEN RAG AI vector store. The system uses PostgreSQL with the pgvector extension for efficient vector storage and similarity search.

## Files Overview

- **`init.sql`** - Complete database initialization script with schema creation
- **`test_setup.sql`** - Test data and verification queries for database testing
- **`README.md`** - This documentation file

## Database Configuration

### Connection Details
- **Database**: `rag_db`
- **User**: `rag_user` 
- **Password**: `rag_pass`
- **Host**: `postgres` (Docker container)
- **Port**: `5432`
- **Connection String**: `postgresql://rag_user:rag_pass@postgres:5432/rag_db`

### Vector Store Configuration
- **Table Name**: `data_rag_kb`
- **Vector Dimension**: `384` (matches all-MiniLM-L6-v2 embedding model)
- **Extension**: pgvector for PostgreSQL 16

## Schema Structure

### Main Vector Table: `data_rag_kb`

```sql
CREATE TABLE data_rag_kb (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,                    -- Document text content
    metadata JSONB DEFAULT '{}',           -- Document metadata (source, title, etc.)
    embedding VECTOR(384) NOT NULL,        -- 384-dimensional embedding vector
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes for Performance

1. **Vector Similarity Indexes**:
   - `data_rag_kb_embedding_cosine_idx` - IVFFlat index for cosine similarity
   - `data_rag_kb_embedding_l2_idx` - IVFFlat index for L2 distance

2. **Metadata Indexes**:
   - `data_rag_kb_metadata_idx` - GIN index for JSONB metadata queries
   - `data_rag_kb_created_at_idx` - B-tree index for temporal queries

### System Metadata Table: `rag_system_info`

Stores system configuration and version information:

```sql
CREATE TABLE rag_system_info (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Utility Views and Functions

- **`vector_store_stats`** - View providing vector store statistics
- **`check_vector_store_health()`** - Function for health checking
- **`update_updated_at_column()`** - Trigger function for automatic timestamp updates

## Setup Instructions

### 1. Initialize Database

Run the initialization script to create the complete schema:

```bash
# Using psql
psql -h localhost -p 5432 -U rag_user -d rag_db -f init.sql

# Using Docker
docker exec -i dtsen_rag_postgres psql -U rag_user -d rag_db < init.sql
```

### 2. Load Test Data (Optional)

For testing purposes, load sample data:

```bash
# Using psql
psql -h localhost -p 5432 -U rag_user -d rag_db -f test_setup.sql

# Using Docker
docker exec -i dtsen_rag_postgres psql -U rag_user -d rag_db < test_setup.sql
```

### 3. Verify Installation

Check that everything is working correctly:

```sql
-- Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check table exists
\dt data_rag_kb

-- Run health check
SELECT * FROM check_vector_store_health();

-- View statistics
SELECT * FROM vector_store_stats;
```

## Common Queries

### Vector Similarity Search

```sql
-- Find most similar documents (cosine similarity)
SELECT 
    text,
    metadata->>'title' as title,
    1 - (embedding <=> $1) as similarity
FROM data_rag_kb
ORDER BY embedding <=> $1
LIMIT 5;
```

### Filtered Search

```sql
-- Search within specific source type
SELECT text, metadata->>'title'
FROM data_rag_kb
WHERE metadata->>'source_type' = 'documents'
ORDER BY embedding <=> $1
LIMIT 5;
```

### Metadata Queries

```sql
-- Find documents by source
SELECT DISTINCT metadata->>'source' FROM data_rag_kb;

-- Count documents by type
SELECT 
    metadata->>'source_type' as type,
    COUNT(*) as count
FROM data_rag_kb
GROUP BY metadata->>'source_type';
```

## Metadata Schema

The `metadata` JSONB field contains structured information about each document chunk:

```json
{
  "source": "document_filename.pdf",
  "source_type": "documents|web|api|database|mcp",
  "title": "Document Title",
  "chunk_id": "unique_chunk_identifier",
  "source_document_id": "parent_document_id",
  "url": "http://example.com",  // for web sources
  "endpoint": "/api/endpoint",  // for API sources
  "custom_field": "value"       // connector-specific fields
}
```

## Performance Considerations

### Index Configuration

The IVFFlat indexes are configured with `lists = 100`, which is suitable for:
- Small to medium datasets (up to 1M vectors)
- Good balance between index build time and query performance
- Adjust based on your dataset size:
  - `lists = sqrt(rows)` for larger datasets
  - `lists = rows/1000` for very large datasets

### Query Optimization

1. **Use appropriate operators**:
   - `<=>` for cosine similarity (most common for RAG)
   - `<->` for L2 distance
   - `<#>` for inner product

2. **Filter before similarity search** when possible:
   ```sql
   WHERE metadata->>'source_type' = 'documents'
   ORDER BY embedding <=> $1
   ```

3. **Limit results** to improve performance:
   ```sql
   LIMIT 10  -- or appropriate number
   ```

## Health Monitoring

### Built-in Health Checks

Use the provided health check function:

```sql
SELECT * FROM check_vector_store_health();
```

This checks:
- pgvector extension status
- Main table existence and content
- Index availability
- Vector dimension consistency

### Manual Health Checks

```sql
-- Check vector dimensions
SELECT DISTINCT vector_dims(embedding) FROM data_rag_kb;

-- Check for null embeddings
SELECT COUNT(*) FROM data_rag_kb WHERE embedding IS NULL;

-- Check table size
SELECT pg_size_pretty(pg_total_relation_size('data_rag_kb'));

-- Check index usage
EXPLAIN (ANALYZE) 
SELECT * FROM data_rag_kb 
ORDER BY embedding <=> array_fill(0.1, ARRAY[384])::vector 
LIMIT 5;
```

## Compatibility with LlamaIndex

This schema is designed to be **fully compatible** with LlamaIndex's PGVectorStore implementation:

- Table structure matches LlamaIndex expectations
- Vector dimensions align with embedding model
- Metadata format supports LlamaIndex's document structure
- Automatic table creation by LlamaIndex will work with this schema

## Migration and Upgrades

### Schema Versioning

The system includes version tracking in `rag_system_info`:

```sql
SELECT value FROM rag_system_info WHERE key = 'schema_version';
```

### Backup Recommendations

Before any schema changes:

```bash
# Backup entire database
pg_dump -h localhost -p 5432 -U rag_user rag_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup just the vector data
pg_dump -h localhost -p 5432 -U rag_user -t data_rag_kb rag_db > vectors_backup.sql
```

## Troubleshooting

### Common Issues

1. **pgvector extension not found**:
   ```sql
   CREATE EXTENSION vector;
   ```

2. **Vector dimension mismatch**:
   ```sql
   SELECT DISTINCT vector_dims(embedding) FROM data_rag_kb;
   -- Should return 384
   ```

3. **Slow similarity queries**:
   - Check if indexes exist: `\di data_rag_kb*`
   - Verify index usage in query plans
   - Consider adjusting `lists` parameter

4. **Memory issues with large vectors**:
   - Increase `shared_buffers` in postgresql.conf
   - Consider `effective_cache_size` tuning
   - Monitor `work_mem` for sort operations

### Performance Tuning

```sql
-- PostgreSQL configuration for vector workloads
-- Add to postgresql.conf:

shared_buffers = 256MB              -- Increase for larger datasets
effective_cache_size = 1GB          -- Set to available system memory
work_mem = 64MB                     -- For sorting operations
maintenance_work_mem = 512MB        -- For index building
random_page_cost = 1.1              -- Lower for SSD storage
```

## Integration with DTSEN RAG AI

This database schema integrates seamlessly with the DTSEN RAG AI application:

- **Configuration**: Matches settings in `config.py`
- **Health Checks**: Compatible with `utils/health.py`
- **Vector Operations**: Supports `rag_pipeline.py` operations
- **API Endpoints**: Enables all `/chat`, `/search`, and `/index` functionality

The system will automatically use this schema when the database is properly configured and the pgvector extension is available.