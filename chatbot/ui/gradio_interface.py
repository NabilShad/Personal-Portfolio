"""Gradio-based chat interface for the document-aware chatbot."""

import os
import sys
from typing import List, Tuple, Optional

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
except ImportError:
    gr = None

from config.settings import ChatbotConfig
from src.chatbot import DocumentAwareChatbot


class GradioInterface:
    """Gradio web interface for the document-aware chatbot."""
    
    def __init__(self):
        self.chatbot = DocumentAwareChatbot()
        self.config = ChatbotConfig()
        
    def upload_documents(self, files) -> str:
        """Handle document upload."""
        if not files:
            return "No files uploaded."
        
        try:
            file_paths = [file.name for file in files]
            result = self.chatbot.add_documents(file_paths)
            
            if result["status"] == "success":
                return f"✅ {result['message']}\nChunks created: {result['chunks_created']}"
            else:
                return f"❌ {result['message']}"
                
        except Exception as e:
            return f"❌ Error processing files: {str(e)}"
    
    def chat_response(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Handle chat interaction."""
        if not message.strip():
            return "", history
        
        # Get response from chatbot
        result = self.chatbot.chat(message)
        response = result["response"]
        
        # Add sources information if available
        if result["sources"]:
            sources_info = "\n\n**Sources:**\n"
            for i, source in enumerate(result["sources"], 1):
                filename = source["metadata"].get("filename", "Unknown")
                similarity = source.get("similarity", 0)
                sources_info += f"{i}. {filename} (similarity: {similarity:.2f})\n"
            response += sources_info
        
        # Update history
        history.append((message, response))
        
        return "", history
    
    def clear_chat(self) -> Tuple[List, str]:
        """Clear chat history."""
        self.chatbot.clear_conversation()
        return [], "Chat history cleared."
    
    def get_knowledge_base_info(self) -> str:
        """Get knowledge base information."""
        info = self.chatbot.get_knowledge_base_info()
        if "error" in info:
            return f"❌ Error: {info['error']}"
        
        return f"""📊 **Knowledge Base Information:**
- Documents added: {info.get('documents_added', 0)}
- Total chunks: {info.get('total_chunks', 0)}
- Embedding model: {info.get('embedding_model', 'Unknown')}
- Collection name: {info.get('collection_name', 'Unknown')}"""
    
    def reset_knowledge_base(self) -> str:
        """Reset the knowledge base."""
        try:
            self.chatbot.reset_knowledge_base()
            return "✅ Knowledge base reset successfully."
        except Exception as e:
            return f"❌ Error resetting knowledge base: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        if gr is None:
            raise ImportError("gradio is required. Install with: pip install gradio")
        
        with gr.Blocks(
            title="Document-Aware Chatbot",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-container {
                height: 500px;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 📚 Document-Aware Conversational Chatbot
            
            Upload documents and chat with an AI that can answer questions based on your documents.
            Supports PDF, TXT, and DOCX files.
            """)
            
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot_interface = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            show_label=True,
                            container=True
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a question about your documents...",
                                label="Your Message",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("Clear Chat", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Quick Actions")
                        
                        info_btn = gr.Button("📊 Knowledge Base Info", variant="secondary")
                        info_output = gr.Textbox(
                            label="Information",
                            lines=8,
                            interactive=False
                        )
                        
                        reset_btn = gr.Button("🗑️ Reset Knowledge Base", variant="stop")
                        reset_output = gr.Textbox(
                            label="Reset Status",
                            lines=2,
                            interactive=False
                        )
            
            with gr.Tab("📁 Document Management"):
                gr.Markdown("### Upload Documents")
                gr.Markdown("Upload PDF, TXT, or DOCX files to add them to the knowledge base.")
                
                file_upload = gr.Files(
                    label="Select Documents",
                    file_types=[".pdf", ".txt", ".docx"],
                    file_count="multiple"
                )
                
                upload_btn = gr.Button("Upload Documents", variant="primary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=4,
                    interactive=False
                )
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ### About This Chatbot
                
                This is a document-aware conversational chatbot built with:
                - **HuggingFace Sentence Transformers** for text embeddings
                - **ChromaDB** for vector storage and similarity search
                - **Gradio** for the web interface
                - **Conversation memory** for context-aware responses
                
                #### How it works:
                1. **Upload documents** - PDF, TXT, or DOCX files are processed and chunked
                2. **Embeddings generation** - Text chunks are converted to vector embeddings
                3. **Vector storage** - Embeddings are stored in ChromaDB for efficient search
                4. **Conversational queries** - Ask questions in natural language
                5. **Similarity search** - Find relevant document chunks based on your query
                6. **Response generation** - Generate answers grounded in your documents
                
                #### Features:
                - Multi-document support
                - Conversation history and context awareness
                - Source attribution for answers
                - Similarity scoring for relevance
                - Persistent knowledge base
                
                #### Supported file formats:
                - PDF files (.pdf)
                - Text files (.txt)
                - Word documents (.docx)
                """)
            
            # Event handlers
            def submit_message(message, history):
                return self.chat_response(message, history)
            
            # Chat interactions
            msg_input.submit(
                submit_message,
                inputs=[msg_input, chatbot_interface],
                outputs=[msg_input, chatbot_interface]
            )
            
            send_btn.click(
                submit_message,
                inputs=[msg_input, chatbot_interface],
                outputs=[msg_input, chatbot_interface]
            )
            
            clear_btn.click(
                self.clear_chat,
                outputs=[chatbot_interface, reset_output]
            )
            
            # Document management
            upload_btn.click(
                self.upload_documents,
                inputs=[file_upload],
                outputs=[upload_status]
            )
            
            # Information and reset
            info_btn.click(
                self.get_knowledge_base_info,
                outputs=[info_output]
            )
            
            reset_btn.click(
                self.reset_knowledge_base,
                outputs=[reset_output]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": self.config.GRADIO_HOST,
            "server_port": self.config.GRADIO_PORT,
            "share": False,
            "debug": False
        }
        
        # Update with user parameters
        launch_params.update(kwargs)
        
        print(f"Launching Gradio interface at http://{launch_params['server_name']}:{launch_params['server_port']}")
        
        interface.launch(**launch_params)


def main():
    """Main function to launch the interface."""
    try:
        app = GradioInterface()
        app.launch()
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        print("Make sure all required dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()