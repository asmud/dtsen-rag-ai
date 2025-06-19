# 🧪 How to Test DTSEN RAG AI System

## 📋 Quick Start Testing (5 minutes)

Essential validation steps to verify system functionality:

### 1. System Health Check
```bash
# Start the system
docker-compose up -d

# Wait for services to start (30 seconds)
sleep 30

# Check overall health
curl http://localhost:8000/health | jq '.'

# Verify all components are "healthy"
curl http://localhost:8000/health | jq '.overall_status'
```

### 2. Basic Document Indexing
```bash
# Index sample documents (using test data)
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "documents", "source_config": {"path": "/app/tests/data/documents"}}'

# Wait for processing (10 seconds)
sleep 10

# Verify indexing completed
curl "http://localhost:8000/info" | jq '.capabilities'
```

### 3. Quick Chat Test
```bash
# Test basic chat functionality
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is DTSEN?", "include_sources": true}' | jq '.'

# Should return response with sources from indexed documents
```

## 🔍 Comprehensive Testing (30 minutes)

Complete system verification across all components:

### Phase 1: Multi-Source Data Ingestion (15 minutes)

#### A. Document Processing Test
```bash
# Test various document formats
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "documents",
    "source_config": {
      "path": "/app/tests/data/documents"
    }
  }'

# Verify different file formats were processed
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "DTSEN", "limit": 10}' | jq '.results[].source'
```

#### B. Web Crawling Test
```bash
# Test web content extraction
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "web",
    "source_config": {
      "urls": ["https://httpbin.org/json"],
      "max_pages": 1
    }
  }'

# Verify web content was indexed
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "httpbin", "limit": 5}' | jq '.results'
```

#### C. API Integration Test
```bash
# Test JSON API data ingestion
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "api",
    "source_config": {
      "url": "https://jsonplaceholder.typicode.com/posts/1",
      "content_fields": ["body", "title"],
      "title_fields": ["title"],
      "id_fields": ["id"]
    }
  }'

# Test with authentication
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "api",
    "source_config": {
      "url": "https://httpbin.org/bearer",
      "auth": {"type": "bearer", "token": "test_token"},
      "content_fields": ["authenticated", "token"]
    }
  }'
```

#### D. Database Integration Test (if enabled)
```bash
# Test database query indexing
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "database",
    "source_config": {
      "queries": [{
        "name": "test_query",
        "query": "SELECT 1 as id, '\''Test Content'\'' as content, '\''Test Title'\'' as title",
        "content_columns": ["content"],
        "title_column": "title",
        "id_column": "id"
      }]
    }
  }'
```

#### E. MCP Protocol Test
```bash
# Test MCP data ingestion
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "mcp",
    "source_config": {
      "mcp_requests": [
        {"endpoint": "/posts/1", "format": "resources"},
        {"endpoint": "/posts/2", "format": "tools"}
      ]
    }
  }'
```

### Phase 2: Search and Chat Integration (10 minutes)

#### A. Vector Search Testing
```bash
# Test similarity search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "government data system",
    "limit": 5,
    "similarity_threshold": 0.1,
    "include_content": true
  }' | jq '.results[] | {title: .title, score: .similarity_score, source: .source}'

# Test search filters
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "technology",
    "limit": 3,
    "filters": {"source": "documents"}
  }' | jq '.'
```

#### B. Chat Integration Testing
```bash
# Test basic chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the purpose of DTSEN?",
    "include_sources": true
  }' | jq '{response: .response, source_count: (.sources | length)}'

# Test with system prompt
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key features",
    "system_prompt": "You are a technical analyst. Provide concise, structured responses.",
    "include_sources": true
  }' | jq '.metadata.system_prompt_used'

# Test with prompt templates
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the system architecture",
    "system_prompt_template": "technical_expert",
    "system_prompt_variables": {"domain": "government systems"}
  }' | jq '.metadata'
```

### Phase 3: System Monitoring (5 minutes)

#### A. Health Check Validation
```bash
# Comprehensive health check
curl "http://localhost:8000/health" | jq '{
  status: .overall_status,
  components: (.components | to_entries | map({key: .key, status: .value.status}))
}'

# Check system information
curl "http://localhost:8000/info" | jq '{
  version: .version,
  models: .models,
  connectors: (.connectors | map(.name))
}'

# Monitor resource usage
curl "http://localhost:8000/resources" | jq '{
  cpu_percent: .utilization.cpu_percent,
  memory_mb: .utilization.memory_mb,
  gpu_available: .recommendations.gpu_available
}'
```

## ⚡ Performance Testing (15 minutes)

### Load Testing
```bash
# Concurrent request testing (requires 'ab' or similar tool)
# Test 10 concurrent requests
ab -n 20 -c 5 -H "Content-Type: application/json" \
   -p <(echo '{"question": "What is DTSEN?"}') \
   http://localhost:8000/chat

# Memory stress test - index larger dataset
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "api",
    "source_config": {
      "url": "https://jsonplaceholder.typicode.com/posts",
      "content_fields": ["body", "title"]
    }
  }'
```

### Response Time Benchmarking
```bash
# Measure response times
time curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key points about DTSEN"}' > /dev/null

# Test search performance
time curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "data system", "limit": 10}' > /dev/null
```

## 🧪 Manual Testing Procedures

### Interactive Testing Workflow

#### 1. API Documentation Testing
```bash
# Access interactive documentation
open http://localhost:8000/docs

# Test each endpoint manually through Swagger UI:
# - /chat: Try different questions and system prompts
# - /search: Test various queries and filters
# - /index: Test different source types
# - /health: Verify component status
# - /info: Check system configuration
```

#### 2. Error Handling Testing
```bash
# Test invalid requests
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"invalid_field": "test"}' | jq '.'

# Test malformed JSON
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{invalid json}' | jq '.'

# Test timeout handling
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "web", "source_config": {"urls": ["http://httpbin.org/delay/30"]}}' | jq '.'
```

#### 3. Data Quality Validation
```bash
# Check indexed content quality
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "include_content": true, "limit": 5}' | \
  jq '.results[] | {title: .title, content_preview: (.content[:100])}'

# Verify source attribution
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What sources are available?", "include_sources": true}' | \
  jq '.sources[] | {source: .source, title: .title}'
```

## 🔧 Troubleshooting Common Issues

### Service Startup Issues
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs -f chatbot
docker-compose logs -f ollama
docker-compose logs -f postgres

# Restart services
docker-compose restart chatbot
```

### Database Connection Issues
```bash
# Test database connectivity
curl "http://localhost:8000/health" | jq '.components.database'

# Check PostgreSQL status
docker-compose exec postgres psql -U rag_user -d rag_db -c "SELECT version();"
```

### Model Loading Issues
```bash
# Check Ollama status
curl "http://localhost:11434/api/tags" | jq '.'

# Pull required models
docker-compose exec ollama ollama pull gemma2:2b
```

### Memory Issues
```bash
# Monitor resource usage
docker stats

# Check available memory
curl "http://localhost:8000/resources" | jq '.utilization'
```

## 📊 Expected Test Results

### Successful Test Indicators
- **Health Status**: All components report "healthy"
- **Response Times**: < 30 seconds for complex queries
- **Memory Usage**: < 1GB under normal load
- **Error Rate**: 0% for valid requests
- **Search Quality**: Similarity scores > 0.1 for relevant content

### Performance Benchmarks
- **Document Processing**: < 5 seconds per document
- **API Indexing**: > 5 documents per second
- **Search Response**: < 1 second
- **Chat Response**: < 15 seconds (depending on complexity)

## 📝 Test Logging and Reporting

### Automated Test Execution
```bash
#!/bin/bash
# save as tests/run_tests.sh

echo "Starting DTSEN RAG AI Test Suite..."

# Health check
echo "1. Health Check"
curl -s "http://localhost:8000/health" | jq -r '.overall_status'

# Basic indexing
echo "2. Document Indexing"
curl -s -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "documents", "source_config": {"path": "/app/tests/data/documents"}}' | \
  jq -r '.status'

# Search test
echo "3. Search Test"
curl -s -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "DTSEN", "limit": 1}' | \
  jq -r '.total_results'

# Chat test
echo "4. Chat Test"
curl -s -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is DTSEN?"}' | \
  jq -r '.response' | head -c 50

echo "Test suite completed!"
```

### Manual Test Checklist

- [ ] System health check passes
- [ ] All 5 data sources can be indexed
- [ ] Search returns relevant results
- [ ] Chat generates coherent responses
- [ ] System prompts work correctly
- [ ] Error handling is graceful
- [ ] Performance meets benchmarks
- [ ] Resource usage is reasonable
- [ ] API documentation is accessible
- [ ] All components report healthy status

## 🎯 Best Practices

### Before Testing
1. Ensure all Docker services are running
2. Wait 30-60 seconds after startup for model loading
3. Check system resources (4GB+ RAM recommended)
4. Verify network connectivity for external API tests

### During Testing
1. Monitor system resources with `docker stats`
2. Check logs for any warnings or errors
3. Test one component at a time for troubleshooting
4. Save test results for comparison with future tests

### After Testing
1. Clean up test data if needed
2. Document any issues or unexpected behavior
3. Update test procedures based on findings
4. Archive test results for future reference

---

**Testing Guide Version**: 2.0.0  
**Last Updated**: June 2025  
**Next Review**: Quarterly