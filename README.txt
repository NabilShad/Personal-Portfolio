# Project: Conversational Document-Aware Chatbot with HuggingFace Embeddings and ChromaDB

# Goal:
# Build an intelligent chatbot that can answer user queries conversationally based on the content of uploaded documents.
# The chatbot should:
# - Embed and store documents as dense vectors using HuggingFace sentence-transformers
# - Use ChromaDB as a vector database for efficient similarity search
# - Accept user input in natural language
# - Retrieve relevant document chunks based on vector similarity
# - Maintain conversation history to provide context-aware responses
# - Use a language model to generate precise, grounded answers based on retrieved context
# - Be accessible via a simple conversational interface (Gradio, Streamlit, or web frontend)

# Functional Requirements:
# 1. Document ingestion and embedding
#    - Load documents (PDF/TXT/DOCX)
#    - Chunk into sections if large
#    - Generate embeddings using a HuggingFace model (e.g., bge-small-en, e5-small-v2, all-MiniLM)
#    - Store in ChromaDB with metadata (doc title, chunk index, etc.)

# 2. Conversational query handling
#    - Accept user input as free-form questions
#    - Maintain a conversation history (list of user + assistant messages)
#    - On each new message:
#        a. Build a context-aware query (based on last N turns)
#        b. Embed the query and search ChromaDB for top-k relevant chunks
#        c. Use the retrieved chunks + history to generate an answer with a language model
#        d. Respond with a grounded, precise, non-hallucinated reply
#        e. If no answer is found, reply with: "The answer is not found in the provided documents."

# 3. Frontend Interface (optional but encouraged)
#    - Use Gradio or Streamlit to create a chat interface
#    - Display chat history, current answer, and source chunks on request

# Suggested Tech Stack:
# - sentence-transformers (HuggingFace)
# - chromadb
# - openai or transformers for LLM response (or local LLM like Mistral/phi)
# - gradio or streamlit for UI
# - fastapi (optional, for backend if building a web frontend)
# - langchain (optional, for memory handling or chaining)

# Copilot, please:
# - Scaffold the code to initialize ChromaDB and load embedding model
# - Write document ingestion and chunking pipeline
# - Write functions to handle conversation, memory, search, and generation
# - Build a minimal Gradio chatbot interface with history support