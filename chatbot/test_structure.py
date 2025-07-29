#!/usr/bin/env python3
"""
Simple test to verify the chatbot structure without heavy dependencies.
"""

import os
import sys
from pathlib import Path

# Add the chatbot directory to the path
chatbot_dir = Path(__file__).parent
sys.path.append(str(chatbot_dir))

def test_imports():
    """Test basic imports."""
    print("🧪 Testing imports...")
    
    try:
        from config.settings import ChatbotConfig
        print("✅ Config module")
        
        from src.document_processor import DocumentProcessor  
        print("✅ Document processor")
        
        from src.conversation_manager import ConversationManager
        print("✅ Conversation manager")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from config.settings import ChatbotConfig
        from src.conversation_manager import ConversationManager
        from src.document_processor import DocumentProcessor
        
        # Test config
        config = ChatbotConfig()
        print(f"✅ Config: {config.EMBEDDING_MODEL}")
        
        # Test conversation manager
        cm = ConversationManager()
        cm.add_user_message("Hello")
        cm.add_assistant_message("Hi there!")
        history = cm.get_conversation_history()
        print(f"✅ Conversation: {len(history)} messages")
        
        # Test document processor
        processor = DocumentProcessor()
        test_text = "This is a test document. It has multiple sentences."
        chunks = processor.chunk_text(test_text)
        print(f"✅ Document processing: {len(chunks)} chunks")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_file_structure():
    """Test file structure."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "config/settings.py",
        "src/chatbot.py", 
        "src/document_processor.py",
        "src/embedding_manager.py",
        "src/conversation_manager.py",
        "ui/gradio_interface.py",
        "requirements.txt",
        "README.md",
        "setup.py"
    ]
    
    base_path = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_example_documents():
    """Test example documents."""
    print("\n🧪 Testing example documents...")
    
    example_dir = Path(__file__).parent / "examples" / "example_documents"
    
    if not example_dir.exists():
        print("❌ Example documents directory missing")
        return False
    
    example_files = list(example_dir.glob("*.txt"))
    
    for file_path in example_files:
        with open(file_path, 'r') as f:
            content = f.read()
            print(f"✅ {file_path.name}: {len(content)} characters")
    
    return len(example_files) > 0

def main():
    """Run all tests."""
    print("🚀 Document-Aware Chatbot Structure Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Example Documents", test_example_documents)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} passed")
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All structure tests passed!")
        print("📋 To run with full dependencies:")
        print("   1. Install dependencies: python setup.py")
        print("   2. Run examples: python examples/example_usage.py")  
        print("   3. Launch interface: python ui/gradio_interface.py")
    else:
        print("❌ Some tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)