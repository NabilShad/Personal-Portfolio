#!/usr/bin/env python3
"""
Example usage script for the Document-Aware Chatbot.
Demonstrates how to use the chatbot programmatically.
"""

import os
import sys
from pathlib import Path

# Add the chatbot directory to the path
chatbot_dir = Path(__file__).parent
sys.path.append(str(chatbot_dir))

from config.settings import ChatbotConfig
from src.chatbot import DocumentAwareChatbot


def example_basic_usage():
    """Demonstrate basic chatbot usage."""
    print("🤖 Basic Chatbot Usage Example")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = DocumentAwareChatbot()
    
    # Add example documents
    example_docs = [
        str(chatbot_dir / "examples" / "example_documents" / "python_basics.txt"),
        str(chatbot_dir / "examples" / "example_documents" / "machine_learning.txt")
    ]
    
    print("📚 Adding example documents...")
    result = chatbot.add_documents(example_docs)
    print(f"Status: {result}")
    
    # Example queries
    queries = [
        "What is Python?",
        "What are the types of machine learning?",
        "Tell me about Python data types",
        "How does supervised learning work?",
        "What libraries are mentioned for machine learning?"
    ]
    
    print("\n💬 Example Conversations:")
    print("-" * 40)
    
    for i, query in enumerate(queries, 1):
        print(f"\n🙋 Query {i}: {query}")
        response = chatbot.chat(query)
        print(f"🤖 Response: {response['response']}")
        
        if response['sources']:
            print("📖 Sources:")
            for j, source in enumerate(response['sources'], 1):
                filename = source['metadata'].get('filename', 'Unknown')
                similarity = source.get('similarity', 0)
                print(f"  {j}. {filename} (similarity: {similarity:.2f})")
    
    # Show knowledge base info
    print("\n📊 Knowledge Base Information:")
    kb_info = chatbot.get_knowledge_base_info()
    for key, value in kb_info.items():
        print(f"  {key}: {value}")
    
    return chatbot


def example_conversation_flow():
    """Demonstrate conversation flow with context."""
    print("\n🗣️ Conversation Flow Example")
    print("=" * 40)
    
    chatbot = DocumentAwareChatbot()
    
    # Add documents
    example_docs = [
        str(chatbot_dir / "examples" / "example_documents" / "python_basics.txt"),
        str(chatbot_dir / "examples" / "example_documents" / "machine_learning.txt")
    ]
    chatbot.add_documents(example_docs)
    
    # Simulate a conversation with follow-up questions
    conversation = [
        "What is machine learning?",
        "What are its main types?",
        "Can you tell me more about supervised learning?",
        "What algorithms are used for it?",
        "How is it different from unsupervised learning?"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\n🙋 Turn {i}: {message}")
        response = chatbot.chat(message)
        print(f"🤖 Response: {response['response'][:200]}...")  # Truncate for display
    
    # Show conversation history
    print("\n📜 Conversation History:")
    history = chatbot.get_conversation_history()
    for entry in history[-4:]:  # Show last 4 entries
        role = entry['role'].title()
        content = entry['content'][:100] + "..." if len(entry['content']) > 100 else entry['content']
        print(f"  {role}: {content}")


def example_document_processing():
    """Demonstrate document processing capabilities."""
    print("\n📄 Document Processing Example")
    print("=" * 40)
    
    from src.document_processor import DocumentProcessor
    from src.embedding_manager import EmbeddingManager
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    embedding_manager = EmbeddingManager()
    
    # Process a document
    example_file = str(chatbot_dir / "examples" / "example_documents" / "python_basics.txt")
    
    print(f"📖 Processing: {Path(example_file).name}")
    
    # Load and chunk document
    chunks = doc_processor.process_document(example_file)
    print(f"📑 Created {len(chunks)} chunks")
    
    # Show chunk information
    for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
        print(f"\nChunk {i}:")
        print(f"  Word count: {chunk['word_count']}")
        print(f"  Content: {chunk['content'][:100]}...")
    
    # Generate embeddings
    print(f"\n🧮 Generating embeddings...")
    chunks_with_embeddings = embedding_manager.embed_chunks(chunks)
    
    for i, chunk in enumerate(chunks_with_embeddings[:2], 1):
        embedding_shape = chunk['embedding'].shape
        print(f"  Chunk {i} embedding shape: {embedding_shape}")
    
    # Test similarity search
    query = "What is Python?"
    query_embedding = embedding_manager.embed_text(query)
    
    print(f"\n🔍 Similarity search for: '{query}'")
    candidate_embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
    similar_chunks = embedding_manager.find_most_similar(query_embedding, candidate_embeddings, top_k=3)
    
    for rank, (idx, similarity) in enumerate(similar_chunks, 1):
        chunk_content = chunks_with_embeddings[idx]['content'][:100]
        print(f"  {rank}. Similarity: {similarity:.3f} - {chunk_content}...")


def example_configuration():
    """Demonstrate configuration options."""
    print("\n⚙️ Configuration Example")
    print("=" * 40)
    
    config = ChatbotConfig()
    
    print("Current Configuration:")
    config_attrs = [
        'EMBEDDING_MODEL', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 
        'TOP_K_RESULTS', 'MAX_HISTORY_LENGTH', 'COLLECTION_NAME'
    ]
    
    for attr in config_attrs:
        value = getattr(config, attr, 'Not set')
        print(f"  {attr}: {value}")
    
    # Show model information
    from src.embedding_manager import EmbeddingManager
    embedding_manager = EmbeddingManager()
    model_info = embedding_manager.get_model_info()
    
    print(f"\nEmbedding Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("🚀 Document-Aware Chatbot Examples")
    print("=" * 50)
    
    try:
        # Basic usage
        chatbot = example_basic_usage()
        
        # Conversation flow
        example_conversation_flow()
        
        # Document processing
        example_document_processing()
        
        # Configuration
        example_configuration()
        
        print("\n✅ All examples completed successfully!")
        print("\nTo launch the web interface, run:")
        print("python ui/gradio_interface.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("python setup.py")


if __name__ == "__main__":
    main()