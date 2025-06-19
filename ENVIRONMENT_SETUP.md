# DTSEN RAG AI - Environment Setup Guide

This guide explains how to properly configure environment variables for the DTSEN RAG AI system.

## Quick Start

### 1. Choose Your Hardware Profile

**Apple Silicon (M1/M2/M3/M4):**
```bash
cp .env.apple-silicon .env
docker-compose up -d
```

**NVIDIA GPU Systems:**
```bash
cp .env.nvidia-gpu .env
docker-compose --profile nvidia-gpu up -d
```

**CPU-Only Systems:**
```bash
cp .env.cpu-only .env
docker-compose --profile cpu-only up -d
```

### 2. Validate Configuration

```bash
# Validate your environment setup
python validate_env.py

# Check all configuration files
python validate_env.py --check-all

# Generate summary report
python validate_env.py --summary
```

## Environment Files Overview

### `.env.apple-silicon`
- Optimized for Mac M1/M2/M3/M4 systems
- CPU-based processing with unified memory optimizations
- 8 CPU workers, 32 batch size for embeddings

### `.env.nvidia-gpu` 
- Optimized for NVIDIA GPU systems
- CUDA acceleration enabled
- Higher batch sizes (64-128) and concurrent requests (16)
- GPU memory management configurations

### `.env.cpu-only`
- Conservative settings for CPU-only systems
- Lower resource usage and batch sizes
- Suitable for development or low-resource environments

## Key Configuration Variables

### Database & Vector Store
```bash
DATABASE_URL=postgresql://rag_user:rag_pass@postgres:5432/rag_db
COLLECTION_NAME=data_rag_kb          # Vector collection name
VECTOR_DIMENSION=384                 # Embedding dimension
```

### LLM Configuration
```bash
OLLAMA_API=http://ollama:11434       # Ollama service endpoint
LLM_MODEL=gemma2:2b                  # Primary language model
LLM_FALLBACK_MODEL=llama3            # Fallback model
```

### Embedding Configuration
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu                 # cpu, cuda, or auto
EMBEDDING_BATCH_SIZE=32              # Adjust based on hardware
```

### Hardware Optimization
```bash
GPU_ENABLED=false                    # Enable GPU acceleration
CPU_WORKERS=4                        # Number of CPU workers
MAX_CONCURRENT_REQUESTS=8            # Concurrent request limit
```

### Feature Flags
```bash
MCP_ENABLED=true                     # Model Context Protocol
DB_QUERY_ENABLED=true                # Database query connector
FEATURE_WEB_CRAWLING=true            # Web crawling capability
```

## Docker Compose Profiles

The system supports three deployment profiles:

### Default (Apple Silicon)
```bash
docker-compose up -d
```
- Uses `.env.apple-silicon` configuration
- ARM64 platform optimizations
- Unified memory management

### NVIDIA GPU Profile
```bash
docker-compose --profile nvidia-gpu up -d
```
- Uses `.env.nvidia-gpu` configuration
- CUDA GPU acceleration
- Higher resource limits

### CPU-Only Profile
```bash
docker-compose --profile cpu-only up -d
```
- Uses `.env.cpu-only` configuration
- Conservative resource usage
- Maximum compatibility

## Customization

### Creating Custom Environment
1. Copy the closest matching profile file:
   ```bash
   cp .env.apple-silicon .env.custom
   ```

2. Modify values as needed:
   ```bash
   nano .env.custom
   ```

3. Use custom environment:
   ```bash
   cp .env.custom .env
   docker-compose up -d
   ```

### Available Configuration Templates
- **`.env.apple-silicon`** - Complete template with all variables for Mac systems
- **`.env.nvidia-gpu`** - Template optimized for NVIDIA GPU systems  
- **`.env.cpu-only`** - Conservative template for CPU-only systems

### Common Customizations

**Increase Memory Limits:**
```bash
MAX_CONCURRENT_REQUESTS=16
EMBEDDING_BATCH_SIZE=64
DB_POOL_SIZE=20
```

**Development Mode:**
```bash
LOG_LEVEL=DEBUG
API_RELOAD=true
ENVIRONMENT=development
```

**Production Security:**
```bash
API_KEY=your_secure_api_key
RATE_LIMIT_ENABLED=true
CORS_ORIGINS=https://your-domain.com
```

## Validation & Troubleshooting

### Environment Validation
```bash
# Basic validation
python validate_env.py

# Comprehensive check
python validate_env.py --check-all

# Validate specific file
python validate_env.py --env-file .env.nvidia-gpu

# Validate current .env file
python validate_env.py --env-file .env
```

### Common Issues

**Collection Name Mismatch:**
```bash
# Ensure consistent collection name
COLLECTION_NAME=data_rag_kb  # Must match database schema
```

**Database Connection Issues:**
```bash
# Use correct database URL format
DATABASE_URL=postgresql://rag_user:rag_pass@postgres:5432/rag_db
```

**GPU Not Detected:**
```bash
# Check GPU configuration
GPU_ENABLED=true
NVIDIA_VISIBLE_DEVICES=all
EMBEDDING_DEVICE=cuda
```

**Memory Issues:**
```bash
# Reduce batch sizes
EMBEDDING_BATCH_SIZE=16
MAX_CONCURRENT_REQUESTS=4
```

### Health Check
```bash
# Verify system health
curl http://localhost:8000/health | jq '.overall_status'

# Check component status
curl http://localhost:8000/health | jq '.components'
```

## Migration from Old Configuration

If you have an existing `.env` file:

1. **Backup existing configuration:**
   ```bash
   cp .env .env.backup
   ```

2. **Check for issues:**
   ```bash
   python validate_env.py --env-file .env.backup
   ```

3. **Choose appropriate profile:**
   ```bash
   cp .env.apple-silicon .env  # or appropriate profile
   ```

4. **Merge custom settings:**
   - Compare `.env.backup` with new `.env`
   - Copy any custom values to new environment file

## Advanced Configuration

### Multi-Environment Setup
```bash
# Development
cp .env.apple-silicon .env.dev
# Modify development-specific settings

# Production  
cp .env.apple-silicon .env.prod
# Modify production-specific settings

# Use environment
cp .env.dev .env && docker-compose up -d
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats

# Check API resource metrics
curl http://localhost:8000/resources | jq '.'
```

### Performance Tuning
```bash
# High-performance configuration
MAX_CONCURRENT_REQUESTS=20
EMBEDDING_BATCH_SIZE=128
DB_POOL_SIZE=25
API_WORKERS=4
```

## Support

- **Validation Issues**: Run `python validate_env.py --check-all`
- **Configuration Questions**: Check `.env.example` for all options
- **Performance Issues**: Try appropriate hardware profile
- **Connection Issues**: Verify service dependencies in docker-compose.yml

---

**Environment Setup Version**: 2.0.0  
**Last Updated**: June 2025  
**Next Review**: Quarterly