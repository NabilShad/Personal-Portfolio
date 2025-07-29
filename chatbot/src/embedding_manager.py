"""Embedding management using HuggingFace sentence-transformers."""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from config.settings import ChatbotConfig


class EmbeddingManager:
    """Manages text embeddings using HuggingFace sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or ChatbotConfig.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load embedding model {self.model_name}: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        
        try:
            embedding = self.model.encode([text])[0]
            return embedding
        except Exception as e:
            raise Exception(f"Failed to embed text: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided for embedding")
        
        try:
            embeddings = self.model.encode(valid_texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to embed texts: {str(e)}")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to document chunks."""
        if not chunks:
            return []
        
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            chunk['embedding_model'] = self.model_name
        
        return chunks
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """Find the most similar embeddings to the query."""
        if not candidate_embeddings:
            return []
        
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "status": "Loaded"
        }