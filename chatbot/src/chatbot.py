"""Core chatbot implementation with ChromaDB integration."""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None

from config.settings import ChatbotConfig
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.conversation_manager import ConversationManager


class DocumentAwareChatbot:
    """A conversational chatbot that answers queries based on uploaded documents."""
    
    def __init__(self, config: ChatbotConfig = None):
        self.config = config or ChatbotConfig()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.conversation_manager = ConversationManager()
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self._init_chromadb()
        
        # Initialize language model
        self.llm_pipeline = None
        self._init_language_model()
        
        # Document storage
        self.document_count = 0
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        if chromadb is None:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        try:
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_DB_PATH,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"ChromaDB initialized with collection: {self.config.COLLECTION_NAME}")
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def _init_language_model(self):
        """Initialize the language model for response generation."""
        if self.config.USE_OPENAI and self.config.OPENAI_API_KEY:
            # OpenAI will be handled separately in generate_response
            print("Using OpenAI API for response generation")
            return
        
        if pipeline is None:
            print("Warning: transformers not available. Response generation will be limited.")
            return
        
        try:
            print(f"Loading language model: {self.config.LLM_MODEL}")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.config.LLM_MODEL,
                device=-1  # CPU
            )
            print("Language model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load language model: {str(e)}")
            self.llm_pipeline = None
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the knowledge base."""
        if not file_paths:
            return {"status": "error", "message": "No file paths provided"}
        
        try:
            # Process documents
            all_chunks = self.document_processor.process_multiple_documents(file_paths)
            
            if not all_chunks:
                return {"status": "error", "message": "No content extracted from documents"}
            
            # Generate embeddings
            chunks_with_embeddings = self.embedding_manager.embed_chunks(all_chunks)
            
            # Store in ChromaDB
            self._store_chunks_in_chromadb(chunks_with_embeddings)
            
            self.document_count += len(file_paths)
            
            return {
                "status": "success",
                "message": f"Successfully processed {len(file_paths)} documents",
                "chunks_created": len(chunks_with_embeddings),
                "total_documents": self.document_count
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error adding documents: {str(e)}"}
    
    def _store_chunks_in_chromadb(self, chunks: List[Dict[str, Any]]):
        """Store document chunks with embeddings in ChromaDB."""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            embeddings.append(chunk['embedding'].tolist())
            documents.append(chunk['content'])
            
            # Prepare metadata (ChromaDB doesn't accept nested dicts)
            metadata = {
                'filename': chunk.get('filename', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'word_count': chunk.get('word_count', 0),
                'file_type': chunk.get('file_type', ''),
                'embedding_model': chunk.get('embedding_model', '')
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Stored {len(chunks)} chunks in ChromaDB")
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query."""
        top_k = top_k or self.config.TOP_K_RESULTS
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_text(query)
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= self.config.SIMILARITY_THRESHOLD:
                        relevant_chunks.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'rank': i + 1
                        })
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]], context: str = "") -> str:
        """Generate a response based on query, retrieved chunks, and conversation context."""
        if not relevant_chunks:
            return "The answer is not found in the provided documents."
        
        # Prepare context from retrieved chunks
        source_context = "\n\n".join([
            f"Source {i+1} (from {chunk['metadata'].get('filename', 'unknown')}): {chunk['content']}"
            for i, chunk in enumerate(relevant_chunks[:3])  # Use top 3 chunks
        ])
        
        # Build prompt
        prompt = self._build_prompt(query, source_context, context)
        
        if self.config.USE_OPENAI and self.config.OPENAI_API_KEY:
            return self._generate_openai_response(prompt)
        elif self.llm_pipeline:
            return self._generate_local_response(prompt)
        else:
            # Fallback: basic template response
            return self._generate_template_response(query, relevant_chunks)
    
    def _build_prompt(self, query: str, source_context: str, conversation_context: str = "") -> str:
        """Build a prompt for the language model."""
        prompt = f"""You are a helpful assistant that answers questions based on provided documents. 
Use only the information from the sources below to answer the question. If the answer cannot be found in the sources, say "The answer is not found in the provided documents."

{conversation_context}

Sources:
{source_context}

Question: {query}

Answer:"""
        return prompt
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            openai.api_key = self.config.OPENAI_API_KEY
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with OpenAI API: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
    
    def _generate_local_response(self, prompt: str) -> str:
        """Generate response using local language model."""
        try:
            response = self.llm_pipeline(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            return response[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            print(f"Error with local model: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
    
    def _generate_template_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate a basic template response when no LLM is available."""
        if not relevant_chunks:
            return "The answer is not found in the provided documents."
        
        # Simple extraction-based response
        top_chunk = relevant_chunks[0]
        source_info = f"from {top_chunk['metadata'].get('filename', 'uploaded document')}"
        
        response = f"Based on the {source_info}, here's what I found:\n\n{top_chunk['content'][:300]}..."
        if len(top_chunk['content']) > 300:
            response += "\n\n[Answer truncated for brevity]"
        
        return response
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Main chat interface method."""
        if not user_input or not user_input.strip():
            return {
                "response": "Please provide a question or message.",
                "sources": [],
                "error": None
            }
        
        try:
            # Add user message to conversation history
            self.conversation_manager.add_user_message(user_input)
            
            # Build context-aware query
            context_aware_query = self.conversation_manager.build_context_aware_query(user_input)
            
            # Retrieve relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(context_aware_query)
            
            # Get conversation context
            conversation_context = self.conversation_manager.get_recent_context(2)
            
            # Generate response
            response = self._generate_response(user_input, relevant_chunks, conversation_context)
            
            # Add assistant message to history
            self.conversation_manager.add_assistant_message(
                response, 
                sources=relevant_chunks[:3]  # Include top 3 sources
            )
            
            return {
                "response": response,
                "sources": relevant_chunks[:3],
                "error": None,
                "conversation_id": self.conversation_manager.session_id
            }
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            return {
                "response": "I'm sorry, I encountered an error while processing your request.",
                "sources": [],
                "error": error_msg
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_manager.get_conversation_history()
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_manager.clear_history()
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base."""
        try:
            collection_count = self.collection.count()
            return {
                "documents_added": self.document_count,
                "total_chunks": collection_count,
                "embedding_model": self.embedding_manager.model_name,
                "collection_name": self.config.COLLECTION_NAME
            }
        except Exception as e:
            return {"error": str(e)}
    
    def reset_knowledge_base(self):
        """Reset the ChromaDB collection."""
        try:
            self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self.document_count = 0
            print("Knowledge base reset successfully")
        except Exception as e:
            print(f"Error resetting knowledge base: {str(e)}")