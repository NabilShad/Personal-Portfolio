# 🤖 Document-Aware Conversational Chatbot

A sophisticated conversational AI chatbot that can answer questions based on uploaded documents using HuggingFace embeddings and ChromaDB vector database.

## ✨ Features

- **📚 Multi-format Document Support**: PDF, TXT, and DOCX files
- **🧠 Intelligent Embeddings**: Uses HuggingFace sentence-transformers for semantic understanding
- **🔍 Vector Database**: ChromaDB for efficient similarity search and retrieval
- **💭 Conversation Memory**: Maintains context across conversation turns
- **🌐 Web Interface**: Beautiful Gradio-based chat interface
- **📊 Source Attribution**: Shows which documents answers come from
- **⚙️ Configurable**: Extensive configuration options via environment variables
- **🔄 Persistent Storage**: ChromaDB provides persistent vector storage

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   Embeddings     │    │   ChromaDB      │
│  (PDF/TXT/DOCX) │───▶│ (HuggingFace)    │───▶│ Vector Storage  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  Gradio UI      │    │  Chatbot Core    │◀────────────┘
│  (Web Interface)│◀──▶│ (Conversation    │
└─────────────────┘    │  Management)     │
                       └──────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the chatbot directory
cd chatbot

# Install dependencies and setup
python setup.py
```

### 2. Launch the Web Interface

```bash
python ui/gradio_interface.py
```

### 3. Upload Documents and Start Chatting!

1. Go to the "Document Management" tab
2. Upload your PDF, TXT, or DOCX files
3. Switch to the "Chat" tab
4. Ask questions about your documents!

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for complete dependency list

### Core Dependencies

```
sentence-transformers>=2.2.2
chromadb>=0.4.15
gradio>=4.0.0
transformers>=4.35.0
torch>=2.0.0
PyPDF2>=3.0.1
python-docx>=0.8.11
```

## 🔧 Configuration

Create a `.env` file (copy from `.env.example`) to customize settings:

```env
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

# UI Settings
GRADIO_PORT=7860
GRADIO_HOST=0.0.0.0
```

## 💻 Programmatic Usage

```python
from config.settings import ChatbotConfig
from src.chatbot import DocumentAwareChatbot

# Initialize chatbot
chatbot = DocumentAwareChatbot()

# Add documents
result = chatbot.add_documents(['document1.pdf', 'document2.txt'])
print(result)

# Chat with the bot
response = chatbot.chat("What is the main topic of the documents?")
print(response['response'])
print("Sources:", response['sources'])

# View conversation history
history = chatbot.get_conversation_history()
```

## 📁 Project Structure

```
chatbot/
├── src/
│   ├── __init__.py
│   ├── chatbot.py                 # Core chatbot implementation
│   ├── document_processor.py      # Document loading and chunking
│   ├── embedding_manager.py       # HuggingFace embeddings
│   └── conversation_manager.py    # Conversation history
├── ui/
│   ├── __init__.py
│   └── gradio_interface.py        # Gradio web interface
├── config/
│   ├── __init__.py
│   └── settings.py                # Configuration management
├── examples/
│   ├── example_documents/         # Sample documents
│   └── example_usage.py           # Usage examples
├── requirements.txt               # Dependencies
├── setup.py                      # Setup script
└── README.md                     # This file
```

## 🎯 How It Works

### 1. Document Processing
- Documents are loaded using appropriate parsers (PyPDF2, python-docx)
- Text is cleaned and split into overlapping chunks
- Each chunk maintains metadata (filename, position, etc.)

### 2. Embedding Generation
- Text chunks are converted to dense vectors using HuggingFace sentence-transformers
- Default model: `all-MiniLM-L6-v2` (384 dimensions, multilingual)
- Embeddings capture semantic meaning of text

### 3. Vector Storage
- ChromaDB stores embeddings with metadata
- Supports cosine similarity search
- Persistent storage across sessions

### 4. Query Processing
- User queries are embedded using the same model
- Vector similarity search retrieves relevant chunks
- Conversation context is maintained for follow-up questions

### 5. Response Generation
- Retrieved chunks provide context for response generation
- Supports multiple LLM backends (local models, OpenAI API)
- Fallback to template-based responses if no LLM available

## 🔍 Features in Detail

### Document Support
- **PDF**: Extracts text from all pages
- **TXT**: Handles various encodings (UTF-8, Latin-1)
- **DOCX**: Extracts text from paragraphs

### Conversation Management
- Maintains conversation history
- Context-aware query building
- Configurable history length
- Session management

### Vector Search
- Cosine similarity matching
- Configurable similarity threshold
- Top-k result retrieval
- Source attribution

### Web Interface
- Clean, intuitive Gradio interface
- Real-time chat interface
- Document upload management
- Knowledge base information display
- Conversation history visualization

## 🛠️ Advanced Usage

### Custom Embedding Models

```python
from config.settings import ChatbotConfig
from src.chatbot import DocumentAwareChatbot

# Configure custom embedding model
config = ChatbotConfig()
config.EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

chatbot = DocumentAwareChatbot(config)
```

### Using OpenAI API

```env
USE_OPENAI=true
OPENAI_API_KEY=your_api_key_here
```

### Batch Document Processing

```python
import os
from pathlib import Path

# Process all documents in a directory
doc_dir = Path("./documents")
doc_files = list(doc_dir.glob("*.pdf")) + list(doc_dir.glob("*.txt"))
file_paths = [str(f) for f in doc_files]

result = chatbot.add_documents(file_paths)
```

## 📊 Performance Notes

- **Embedding Model**: `all-MiniLM-L6-v2` is fast and lightweight (384D)
- **Chunk Size**: 500 words with 50-word overlap balances context and precision
- **Memory Usage**: Scales with document size and chunk count
- **Search Speed**: ChromaDB provides fast approximate nearest neighbor search

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **ChromaDB Permissions**
   ```bash
   chmod -R 755 ./chroma_db
   ```

3. **Model Download Issues**
   - Models are downloaded automatically on first use
   - Ensure stable internet connection
   - Check HuggingFace Hub status

4. **Memory Issues**
   - Reduce `CHUNK_SIZE` for large documents
   - Use lighter embedding models
   - Process documents in smaller batches

### Performance Tuning

- **Faster Inference**: Use quantized models or GPU acceleration
- **Better Quality**: Use larger models like `all-mpnet-base-v2`
- **Multilingual**: Use `paraphrase-multilingual-*` models
- **Domain-Specific**: Fine-tune embeddings for your domain

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows existing style
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **HuggingFace** for sentence-transformers
- **ChromaDB** for vector database
- **Gradio** for the web interface
- **PyPDF2** and **python-docx** for document processing

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run `python examples/example_usage.py` to test setup
3. Create an issue with detailed error messages

---

**Happy Chatting! 🎉**