"""Document processing utilities for the chatbot."""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from config.settings import ChatbotConfig


class DocumentProcessor:
    """Handles document loading, processing, and chunking."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or ChatbotConfig.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or ChatbotConfig.CHUNK_OVERLAP
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a document and return its content with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                content = self._load_txt(file_path)
            elif file_extension == '.pdf':
                content = self._load_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                content = self._load_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return {
                'content': content,
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': file_extension
            }
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file content."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        content = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content.append(page.extract_text())
        
        return '\n'.join(content)
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file content."""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
        
        doc = DocxDocument(file_path)
        content = []
        for paragraph in doc.paragraphs:
            content.append(paragraph.text)
        
        return '\n'.join(content)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Text is small enough to be one chunk
            chunks.append({
                'content': text,
                'chunk_index': 0,
                'word_count': len(words),
                **( metadata or {})
            })
        else:
            # Split into overlapping chunks
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    'content': chunk_text,
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words),
                    'start_word': i,
                    'end_word': min(i + self.chunk_size, len(words)),
                    **(metadata or {})
                })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with embedding
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Strip and return
        return text.strip()
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document from loading to chunking."""
        # Load document
        doc_data = self.load_document(file_path)
        
        # Chunk the content
        chunks = self.chunk_text(
            doc_data['content'],
            {
                'filename': doc_data['filename'],
                'file_path': doc_data['file_path'],
                'file_type': doc_data['file_type'],
                'file_size': doc_data['file_size']
            }
        )
        
        return chunks
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                print(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return all_chunks