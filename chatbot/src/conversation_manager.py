"""Conversation management for maintaining chat history and context."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from config.settings import ChatbotConfig


class ConversationManager:
    """Manages conversation history and context for the chatbot."""
    
    def __init__(self, max_history_length: int = None):
        self.max_history_length = max_history_length or ChatbotConfig.MAX_HISTORY_LENGTH
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_user_message(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Add a user message to the conversation history."""
        entry = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(entry)
        self._trim_history()
    
    def add_assistant_message(
        self, 
        message: str, 
        sources: List[Dict[str, Any]] = None, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add an assistant message to the conversation history."""
        entry = {
            'role': 'assistant',
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'sources': sources or [],
            'metadata': metadata or {}
        }
        self.conversation_history.append(entry)
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Trim conversation history to maximum length."""
        if len(self.conversation_history) > self.max_history_length:
            # Keep the most recent messages
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history.copy()
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation context as a formatted string."""
        if not self.conversation_history:
            return ""
        
        # Get the last num_turns * 2 messages (user + assistant pairs)
        recent_messages = self.conversation_history[-(num_turns * 2):]
        
        context_parts = []
        for entry in recent_messages:
            role = entry['role'].title()
            content = entry['content']
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def build_context_aware_query(self, current_query: str, num_context_turns: int = 2) -> str:
        """Build a context-aware query incorporating conversation history."""
        if not self.conversation_history:
            return current_query
        
        # Get recent context
        context = self.get_recent_context(num_context_turns)
        
        if not context:
            return current_query
        
        # Combine context with current query
        context_aware_query = f"Previous conversation:\n{context}\n\nCurrent question: {current_query}"
        
        return context_aware_query
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for entry in reversed(self.conversation_history):
            if entry['role'] == 'user':
                return entry['content']
        return None
    
    def get_last_assistant_message(self) -> Optional[Dict[str, Any]]:
        """Get the last assistant message with metadata."""
        for entry in reversed(self.conversation_history):
            if entry['role'] == 'assistant':
                return entry
        return None
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        self.session_id = self._generate_session_id()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        user_messages = [entry for entry in self.conversation_history if entry['role'] == 'user']
        assistant_messages = [entry for entry in self.conversation_history if entry['role'] == 'assistant']
        
        return {
            'session_id': self.session_id,
            'total_messages': len(self.conversation_history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'first_message_time': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
            'last_message_time': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export the full conversation for saving or analysis."""
        return {
            'session_id': self.session_id,
            'conversation_history': self.conversation_history,
            'summary': self.get_conversation_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_conversation(self, conversation_data: Dict[str, Any]) -> None:
        """Import a previously exported conversation."""
        if 'conversation_history' in conversation_data:
            self.conversation_history = conversation_data['conversation_history']
        if 'session_id' in conversation_data:
            self.session_id = conversation_data['session_id']