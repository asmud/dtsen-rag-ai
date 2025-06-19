import re
import logging
from typing import List, Dict, Any, Optional
from config import get_settings

logger = logging.getLogger(__name__)

class TextChunker:
    """Utility for chunking text into smaller segments"""
    
    def __init__(self):
        self.settings = get_settings()
        self.chunk_size = self.settings.MAX_CHUNK_SIZE
        self.overlap = self.settings.CHUNK_OVERLAP
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller segments with overlap
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Choose chunking strategy based on text type
        if self._is_structured_text(cleaned_text):
            chunks = self._chunk_structured_text(cleaned_text)
        else:
            chunks = self._chunk_by_sentences(cleaned_text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_content in enumerate(chunks):
            if len(chunk_content.strip()) < 20:  # Skip very small chunks
                continue
                
            chunk_metadata = {
                'chunk_index': i,
                'chunk_count': len(chunks),
                'chunk_size': len(chunk_content),
                'overlap_size': self.overlap
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            chunk_objects.append({
                'content': chunk_content.strip(),
                'metadata': chunk_metadata
            })
        
        logger.debug(f"Created {len(chunk_objects)} chunks from text of length {len(text)}")
        return chunk_objects
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        return text.strip()
    
    def _is_structured_text(self, text: str) -> bool:
        """Determine if text has structure (headers, sections, etc.)"""
        # Check for markdown headers
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return True
        
        # Check for numbered sections
        if re.search(r'^\d+\.\s+', text, re.MULTILINE):
            return True
        
        # Check for bullet points
        if re.search(r'^[\*\-\+]\s+', text, re.MULTILINE):
            return True
        
        # Check for section separators
        if re.search(r'^[-=]{3,}$', text, re.MULTILINE):
            return True
        
        return False
    
    def _chunk_structured_text(self, text: str) -> List[str]:
        """Chunk text that has clear structure"""
        chunks = []
        
        # Split by major sections first
        sections = self._split_by_sections(text)
        
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Further split large sections
                sub_chunks = self._chunk_by_sentences(section)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by structural elements"""
        # Split by markdown headers
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return re.split(r'\n(?=#{1,6}\s+)', text)
        
        # Split by numbered sections
        if re.search(r'^\d+\.\s+', text, re.MULTILINE):
            return re.split(r'\n(?=\d+\.\s+)', text)
        
        # Split by double line breaks (paragraphs)
        sections = text.split('\n\n')
        return [s.strip() for s in sections if s.strip()]
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences with overlap"""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_size = len(sentence)
            
            # If single sentence is larger than chunk size, split it
            if sentence_size > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # Split large sentence by words
                word_chunks = self._chunk_by_words(sentence)
                chunks.extend(word_chunks)
                i += 1
                continue
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_size = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size = len(current_chunk)
            
            i += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Use regex to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _chunk_by_words(self, text: str) -> List[str]:
        """Chunk text by words when sentences are too large"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_words = self._get_overlap_words(current_chunk)
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if self.overlap <= 0:
            return ""
        
        # Take last N characters for overlap
        if len(text) <= self.overlap:
            return text
        
        # Try to end overlap at sentence boundary
        overlap_text = text[-self.overlap:]
        
        # Find the last sentence boundary in overlap
        sentence_end = max(
            overlap_text.rfind('.'),
            overlap_text.rfind('!'),
            overlap_text.rfind('?')
        )
        
        if sentence_end > self.overlap // 2:  # If we found a good boundary
            return overlap_text[sentence_end + 1:].strip()
        
        return overlap_text
    
    def _get_overlap_words(self, words: List[str]) -> List[str]:
        """Get overlap words from the end of current chunk"""
        if self.overlap <= 0:
            return []
        
        # Calculate overlap in terms of characters
        total_chars = sum(len(w) + 1 for w in words)
        
        # Work backwards to find overlap words
        overlap_words = []
        char_count = 0
        
        for word in reversed(words):
            char_count += len(word) + 1
            overlap_words.insert(0, word)
            
            if char_count >= self.overlap:
                break
        
        return overlap_words
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(sizes),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'target_chunk_size': self.chunk_size,
            'overlap_size': self.overlap
        }