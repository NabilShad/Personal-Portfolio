#!/usr/bin/env python3
"""
Setup script for the Document-Aware Chatbot.
This script helps set up the environment and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "chroma_db",
        "uploads",
        "logs"
    ]
    
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_dependencies():
    """Check if all required packages are available."""
    required_packages = [
        "sentence_transformers",
        "chromadb", 
        "gradio",
        "transformers",
        "torch",
        "PyPDF2",
        "docx",
        "numpy",
        "pandas"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def setup_environment():
    """Set up environment variables."""
    env_file = Path(__file__).parent / ".env.example"
    
    env_content = """# Environment configuration for Document-Aware Chatbot

# Embedding Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB Settings  
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=document_embeddings

# Retrieval Settings
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Conversation Settings
MAX_HISTORY_LENGTH=10

# Language Model Settings
LLM_MODEL=microsoft/DialoGPT-medium
USE_OPENAI=false
# OPENAI_API_KEY=your_openai_api_key_here

# UI Settings
GRADIO_PORT=7860
GRADIO_HOST=0.0.0.0
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created example environment file: {env_file}")
    print("Copy .env.example to .env and modify as needed.")


def run_quick_test():
    """Run a quick test to verify installation."""
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        from sentence_transformers import SentenceTransformer
        import chromadb
        import gradio as gr
        
        # Test embedding model loading
        print("Testing embedding model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode(["Test sentence"])
        print(f"✅ Embedding generated: shape {test_embedding.shape}")
        
        # Test ChromaDB
        print("Testing ChromaDB...")
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        print("✅ ChromaDB working")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Document-Aware Chatbot...")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Create directories
    if success:
        create_directories()
    
    # Check dependencies
    if success and not check_dependencies():
        success = False
    
    # Setup environment
    if success:
        setup_environment()
    
    # Run quick test
    if success and not run_quick_test():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure as needed")
        print("2. Run: python ui/gradio_interface.py")
        print("3. Upload documents and start chatting!")
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()