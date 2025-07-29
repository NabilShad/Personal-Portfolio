#!/usr/bin/env python3
"""
Simple launcher script for the Document-Aware Chatbot.
"""

import sys
import os
from pathlib import Path

# Add the chatbot directory to the path
chatbot_dir = Path(__file__).parent
sys.path.append(str(chatbot_dir))

def main():
    """Launch the chatbot interface."""
    print("🚀 Launching Document-Aware Chatbot...")
    
    try:
        from ui.gradio_interface import GradioInterface
        
        app = GradioInterface()
        app.launch(share=False, debug=True)
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: python setup.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()