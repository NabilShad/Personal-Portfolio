#!/usr/bin/env python3
"""
Simple demo showing the chatbot components working together.
This demo works without installing heavy dependencies like sentence-transformers.
"""

import os
import sys
from pathlib import Path

# Add the chatbot directory to the path
chatbot_dir = Path(__file__).parent
sys.path.append(str(chatbot_dir))

from config.settings import ChatbotConfig
from src.document_processor import DocumentProcessor
from src.conversation_manager import ConversationManager

def demo_document_processing():
    """Demo document processing capabilities."""
    print("📄 Document Processing Demo")
    print("-" * 30)
    
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    # Process example document
    example_file = chatbot_dir / "examples" / "example_documents" / "python_basics.txt"
    
    print(f"Processing: {example_file.name}")
    chunks = processor.process_document(str(example_file))
    
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:2], 1):  # Show first 2 chunks
        print(f"\nChunk {i}:")
        print(f"  Words: {chunk['word_count']}")
        print(f"  Content: {chunk['content'][:150]}...")
    
    return chunks

def demo_conversation_management():
    """Demo conversation management."""
    print("\n💬 Conversation Management Demo")
    print("-" * 35)
    
    cm = ConversationManager(max_history_length=6)
    
    # Simulate a conversation
    conversation_turns = [
        ("user", "What is Python?"),
        ("assistant", "Python is a high-level programming language known for its simplicity."),
        ("user", "What are its main features?"),
        ("assistant", "Python features include easy syntax, interpreted execution, and extensive libraries."),
        ("user", "Tell me about data types"),
        ("assistant", "Python has numbers, strings, lists, tuples, dictionaries, and sets as main data types.")
    ]
    
    for role, message in conversation_turns:
        if role == "user":
            cm.add_user_message(message)
        else:
            cm.add_assistant_message(message)
    
    # Show conversation history
    history = cm.get_conversation_history()
    print(f"Conversation has {len(history)} messages")
    
    for entry in history[-4:]:  # Show last 4 messages
        role = entry['role'].title()
        content = entry['content'][:60] + "..." if len(entry['content']) > 60 else entry['content']
        print(f"  {role}: {content}")
    
    # Show context-aware query building
    print(f"\nContext-aware query example:")
    current_query = "Can you give me examples?"
    context_query = cm.build_context_aware_query(current_query, num_context_turns=2)
    print(f"Original: {current_query}")
    print(f"With context: {context_query[:100]}...")
    
    return cm

def demo_configuration():
    """Demo configuration system."""
    print("\n⚙️ Configuration Demo")
    print("-" * 25)
    
    config = ChatbotConfig()
    
    settings = [
        ("Embedding Model", config.EMBEDDING_MODEL),
        ("Chunk Size", config.CHUNK_SIZE),
        ("Chunk Overlap", config.CHUNK_OVERLAP),
        ("Top K Results", config.TOP_K_RESULTS),
        ("Max History", config.MAX_HISTORY_LENGTH),
        ("Collection Name", config.COLLECTION_NAME)
    ]
    
    for setting, value in settings:
        print(f"  {setting}: {value}")
    
    return config

def demo_integration():
    """Demo integration between components."""
    print("\n🔗 Integration Demo")
    print("-" * 20)
    
    # Create components
    processor = DocumentProcessor()
    cm = ConversationManager()
    
    # Process document and extract key information
    example_file = chatbot_dir / "examples" / "example_documents" / "machine_learning.txt"
    chunks = processor.process_document(str(example_file))
    
    print(f"Loaded {len(chunks)} chunks from {Path(example_file).name}")
    
    # Simulate finding relevant chunk (without embeddings)
    query = "types of machine learning"
    query_words = set(query.lower().split())
    
    # Simple keyword matching for demo
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        content_words = set(chunk['content'].lower().split())
        overlap = len(query_words.intersection(content_words))
        chunk_scores.append((i, overlap, chunk))
    
    # Sort by overlap score
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    if chunk_scores[0][1] > 0:  # If we found relevant chunks
        best_chunk = chunk_scores[0][2]
        print(f"\nQuery: '{query}'")
        print(f"Best matching chunk (score: {chunk_scores[0][1]}):")
        print(f"  {best_chunk['content'][:200]}...")
        
        # Add to conversation
        cm.add_user_message(query)
        response = f"Based on the document, {best_chunk['content'][:150]}..."
        cm.add_assistant_message(response, sources=[{'content': best_chunk['content'][:100]}])
        
        print(f"\nConversation now has {len(cm.get_conversation_history())} messages")

def main():
    """Run all demos."""
    print("🤖 Document-Aware Chatbot Component Demo")
    print("=" * 50)
    print("This demo shows the chatbot components working without heavy ML dependencies.")
    print()
    
    try:
        # Run demos
        chunks = demo_document_processing()
        cm = demo_conversation_management()
        config = demo_configuration()
        demo_integration()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print()
        print("🚀 Next Steps:")
        print("1. Install full dependencies: python setup.py")
        print("2. Try the complete example: python examples/example_usage.py")
        print("3. Launch the web interface: python ui/gradio_interface.py")
        print()
        print("📋 Features demonstrated:")
        print("• Document loading and chunking")
        print("• Conversation history management")
        print("• Configuration system")
        print("• Component integration")
        print("• Keyword-based retrieval (demo version)")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)