import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config import get_settings
from utils.caching import CacheManager
from utils.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manager for embedding generation with caching and fallback models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.primary_model = None
        self.fallback_model = None
        self.cache_manager = CacheManager()
        self.resource_manager = None
        self.device = "cpu"  # Will be set by resource manager
        self.batch_size = self.settings.EMBEDDING_BATCH_SIZE
        self._models_loaded = False
    
    async def initialize(self) -> bool:
        """Initialize embedding models and cache"""
        try:
            # Initialize resource manager
            self.resource_manager = await get_resource_manager()
            
            # Get optimal device and batch size from resource manager
            self.device = self.resource_manager.get_device()
            self.batch_size = self.resource_manager.get_optimal_batch_size("embedding")
            
            # Connect to cache
            if self.settings.ENABLE_CACHING:
                await self.cache_manager.connect()
            
            # Load models
            await self.load_models()
            
            logger.info(f"Embedding manager initialized - Device: {self.device}, Batch size: {self.batch_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {e}")
            return False
    
    async def load_models(self) -> bool:
        """Load embedding models"""
        try:
            # Load primary model
            logger.info(f"Loading primary embedding model: {self.settings.EMBEDDING_MODEL}")
            self.primary_model = await asyncio.to_thread(
                self._load_model, self.settings.EMBEDDING_MODEL
            )
            
            # Load fallback model if different
            if self.settings.EMBEDDING_FALLBACK_MODEL != self.settings.EMBEDDING_MODEL:
                logger.info(f"Loading fallback embedding model: {self.settings.EMBEDDING_FALLBACK_MODEL}")
                self.fallback_model = await asyncio.to_thread(
                    self._load_model, self.settings.EMBEDDING_FALLBACK_MODEL
                )
            else:
                self.fallback_model = self.primary_model
            
            self._models_loaded = True
            logger.info("Embedding models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            return False
    
    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Load a single embedding model"""
        try:
            model = SentenceTransformer(model_name, device=self.device)
            
            # Optimize for inference
            if hasattr(model, 'eval'):
                model.eval()
            
            # GPU-specific optimizations
            if self.device.startswith('cuda') and self.resource_manager:
                if self.settings.MIXED_PRECISION:
                    # Enable mixed precision if supported
                    try:
                        model.half()
                        logger.info(f"Enabled mixed precision for {model_name}")
                    except Exception as e:
                        logger.warning(f"Mixed precision not supported for {model_name}: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    async def get_embedding(self, text: str, use_cache: bool = True) -> Optional[List[float]]:
        """Get embedding for a single text"""
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Check cache first
        if use_cache and self.settings.ENABLE_CACHING:
            cached_embedding = await self.cache_manager.get_embedding(
                text, self.settings.EMBEDDING_MODEL
            )
            if cached_embedding:
                return cached_embedding
        
        if not self._models_loaded:
            await self.load_models()
        
        try:
            # Try primary model first
            embedding = await self._generate_embedding(text, self.primary_model)
            
            if embedding is None and self.fallback_model != self.primary_model:
                logger.warning("Primary model failed, trying fallback model")
                embedding = await self._generate_embedding(text, self.fallback_model)
            
            # Cache the result
            if embedding and use_cache and self.settings.ENABLE_CACHING:
                await self.cache_manager.set_embedding(
                    text, self.settings.EMBEDDING_MODEL, embedding
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts"""
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not cleaned_texts:
            return [None] * len(texts)
        
        results = []
        
        # Check cache for all texts first
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache and self.settings.ENABLE_CACHING:
            for i, text in enumerate(cleaned_texts):
                cached_embedding = await self.cache_manager.get_embedding(
                    text, self.settings.EMBEDDING_MODEL
                )
                if cached_embedding:
                    cached_results[i] = cached_embedding
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = cleaned_texts
            uncached_indices = list(range(len(cleaned_texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            if not self._models_loaded:
                await self.load_models()
            
            # Process in batches
            uncached_embeddings = []
            
            for i in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[i:i + self.batch_size]
                batch_embeddings = await self._generate_embeddings_batch(batch_texts)
                uncached_embeddings.extend(batch_embeddings)
            
            # Cache new embeddings
            if use_cache and self.settings.ENABLE_CACHING:
                for text, embedding in zip(uncached_texts, uncached_embeddings):
                    if embedding:
                        await self.cache_manager.set_embedding(
                            text, self.settings.EMBEDDING_MODEL, embedding
                        )
        else:
            uncached_embeddings = []
        
        # Combine cached and new results
        final_results = [None] * len(cleaned_texts)
        
        # Add cached results
        for i, embedding in cached_results.items():
            final_results[i] = embedding
        
        # Add new results
        for i, embedding in zip(uncached_indices, uncached_embeddings):
            final_results[i] = embedding
        
        return final_results
    
    async def _generate_embedding(self, text: str, model: SentenceTransformer) -> Optional[List[float]]:
        """Generate embedding using specified model"""
        try:
            # Generate embedding in thread pool to avoid blocking
            embedding = await asyncio.to_thread(model.encode, text, convert_to_numpy=True)
            
            # Convert to list and ensure float32
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32).tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return None
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts"""
        try:
            # Try primary model first
            embeddings = await self._batch_encode(texts, self.primary_model)
            
            if embeddings is None and self.fallback_model != self.primary_model:
                logger.warning("Primary model batch failed, trying fallback model")
                embeddings = await self._batch_encode(texts, self.fallback_model)
            
            if embeddings is None:
                return [None] * len(texts)
            
            # Convert to list format
            result = []
            for embedding in embeddings:
                if isinstance(embedding, np.ndarray):
                    result.append(embedding.astype(np.float32).tolist())
                else:
                    result.append(embedding)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    async def _batch_encode(self, texts: List[str], model: SentenceTransformer) -> Optional[np.ndarray]:
        """Encode batch of texts with specified model"""
        try:
            embeddings = await asyncio.to_thread(
                model.encode, 
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return None
    
    async def get_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Calculate cosine similarity between two texts"""
        try:
            embeddings = await self.get_embeddings_batch([text1, text2])
            
            if not embeddings[0] or not embeddings[1]:
                return None
            
            # Calculate cosine similarity
            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return None
    
    async def find_most_similar(self, query_text: str, candidate_texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar texts to query"""
        try:
            # Get embeddings for all texts
            all_texts = [query_text] + candidate_texts
            embeddings = await self.get_embeddings_batch(all_texts)
            
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            if not query_embedding:
                return []
            
            # Calculate similarities
            similarities = []
            query_vec = np.array(query_embedding)
            
            for i, (text, embedding) in enumerate(zip(candidate_texts, candidate_embeddings)):
                if embedding:
                    candidate_vec = np.array(embedding)
                    similarity = np.dot(query_vec, candidate_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
                    )
                    similarities.append({
                        'text': text,
                        'similarity': float(similarity),
                        'index': i
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar texts: {e}")
            return []
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'primary_model': self.settings.EMBEDDING_MODEL,
            'fallback_model': self.settings.EMBEDDING_FALLBACK_MODEL,
            'device': self.device,
            'batch_size': self.batch_size,
            'models_loaded': self._models_loaded,
            'cache_enabled': self.settings.ENABLE_CACHING
        }
        
        if self._models_loaded and self.primary_model:
            try:
                # Get model dimension
                test_embedding = await self._generate_embedding("test", self.primary_model)
                if test_embedding:
                    info['embedding_dimension'] = len(test_embedding)
                    info['expected_dimension'] = self.settings.VECTOR_DIMENSION
                    info['dimension_match'] = len(test_embedding) == self.settings.VECTOR_DIMENSION
                
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding system"""
        try:
            if not self._models_loaded:
                return {
                    'status': 'unhealthy',
                    'message': 'Models not loaded'
                }
            
            # Test embedding generation
            test_text = "Health check test embedding"
            embedding = await self.get_embedding(test_text, use_cache=False)
            
            if embedding and len(embedding) == self.settings.VECTOR_DIMENSION:
                return {
                    'status': 'healthy',
                    'models_loaded': True,
                    'primary_model': self.settings.EMBEDDING_MODEL,
                    'embedding_dimension': len(embedding),
                    'expected_dimension': self.settings.VECTOR_DIMENSION,
                    'cache_enabled': self.settings.ENABLE_CACHING,
                    'device': self.device
                }
            else:
                return {
                    'status': 'degraded',
                    'message': 'Embedding generation failed or dimension mismatch',
                    'models_loaded': True
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f"Health check failed: {e}"
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.cache_manager:
                await self.cache_manager.disconnect()
            
            # Clear model references to free memory
            self.primary_model = None
            self.fallback_model = None
            self._models_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU memory using resource manager
            if self.resource_manager:
                self.resource_manager.cleanup_gpu_memory()
            
            logger.info("Embedding manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")