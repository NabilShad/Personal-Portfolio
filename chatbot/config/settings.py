"""Configuration settings for the conversational chatbot."""

import os
from typing import Optional

class ChatbotConfig:
    """Configuration class for the document-aware chatbot."""
    
    # Embedding model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    
    # ChromaDB settings
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_embeddings")
    
    # Retrieval settings
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Document processing settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Conversation settings
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))
    
    # Language model settings
    LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")
    USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # UI settings
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
    GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        if cls.USE_OPENAI and not cls.OPENAI_API_KEY:
            print("Warning: OpenAI API key not provided while USE_OPENAI is True")
            return False
        return True