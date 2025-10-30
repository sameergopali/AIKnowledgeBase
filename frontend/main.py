import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests
import streamlit as st


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_RAG_TYPE = "suggestion-rag"
RAG_OPTIONS = ["suggestion-rag", "search-rag"]
SUPPORTED_FILE_TYPES = ["pdf", "txt"]
API_TIMEOUT = 120


@dataclass
class SessionKeys:
    """Centralized session state keys."""
    CHAT_HISTORY = "chat_history"
    IS_PROCESSING = "is_processing_query"
    FILE_UPLOADER_KEY = "file_uploader_key"
    API_BASE_URL = "api_base_url"
    SELECTED_RAG = "selected_rag"


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

class SessionStateManager:
    """Manages session state initialization and access."""
    
    @staticmethod
    def initialize() -> None:
        """Initialize all required session state variables."""
        defaults = {
            SessionKeys.CHAT_HISTORY: [],
            SessionKeys.IS_PROCESSING: False,
            SessionKeys.FILE_UPLOADER_KEY: 0,
            SessionKeys.API_BASE_URL: os.getenv("BASE_URL", DEFAULT_API_URL),
            SessionKeys.SELECTED_RAG: DEFAULT_RAG_TYPE
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def get_api_url() -> str:
        """Get the configured API base URL."""
        return st.session_state.get(SessionKeys.API_BASE_URL, DEFAULT_API_URL)
    
    @staticmethod
    def get_selected_rag() -> str:
        """Get the selected RAG workflow."""
        return st.session_state.get(SessionKeys.SELECTED_RAG, DEFAULT_RAG_TYPE)
    
    @staticmethod
    def add_user_message(content: str) -> None:
        """Add a user message to chat history."""
        st.session_state[SessionKeys.CHAT_HISTORY].append({
            "role": "user",
            "content": content
        })
    
    @staticmethod
    def add_assistant_message(content: str, metadata: Dict[str, Any]) -> None:
        """Add an assistant message to chat history."""
        st.session_state[SessionKeys.CHAT_HISTORY].append({
            "role": "assistant",
            "content": content,
            "metadata": metadata
        })
    
    @staticmethod
    def set_processing(is_processing: bool) -> None:
        """Set the processing state."""
        st.session_state[SessionKeys.IS_PROCESSING] = is_processing
    
    @staticmethod
    def increment_uploader_key() -> None:
        """Increment file uploader key to reset it."""
        st.session_state[SessionKeys.FILE_UPLOADER_KEY] += 1


# ============================================================================
# API SERVICE
# ============================================================================

class APIService:
    """Handles all API communication."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def fetch_documents(self) -> List[Dict[str, Any]]:
        """Fetch the list of uploaded documents."""
        try:
            response = requests.get(
                f"{self.base_url}/document",
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("documents", [])
        except Exception as e:
            print(f"Error fetching documents: {e}")
        return []
    
    def upload_document(self, file) -> bool:
        """Upload a single document."""
        try:
            files = {"file": (file.name, file.getvalue())}
            response = requests.post(
                f"{self.base_url}/document",
                files=files,
                timeout=API_TIMEOUT
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error uploading {file.name}: {e}")
            return False
    
    def send_chat_query(self, message: str, rag_type: str) -> Optional[Dict[str, Any]]:
        """Send a chat query and return the response."""
        try:
            payload = {
                "message": message,
                "rag": rag_type,
            }
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=API_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"API Error: {e}")
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable UI component builders."""
    
    @staticmethod
    def show_temporary_message(message_func, text: str, duration: int = 3) -> None:
        """Display a temporary message."""
        placeholder = st.empty()
        with placeholder:
            message_func(text)
        time.sleep(duration)
        placeholder.empty()
    
    @staticmethod
    def render_document_list(documents: List[Dict[str, Any]]) -> None:
        """Display the list of uploaded documents."""
        if documents:
            with st.expander("Knowledge Base", expanded=False):
                for doc in documents:
                    filename = doc.get("filename", "Unknown")
                    st.markdown(f"• {filename}")
        else:
            st.info("No documents uploaded yet")
    
    @staticmethod
    def render_answer_metadata(metadata: Dict[str, Any]) -> None:
        """Display metadata about the answer."""
        confidence = str(metadata.get("confidence", ""))
        suggestions = metadata.get("suggestions", [])
        missing_info = metadata.get("missing_info", [])
        
        if not (confidence or suggestions or missing_info):
            return
        
        with st.expander("Answer Details", expanded=False):
            if confidence:
                st.markdown(f"**Confidence Level: {confidence}**")
                st.markdown("")
            
            if suggestions:
                st.markdown("**💡 Suggestions:**")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
                st.markdown("")
            
            if missing_info:
                st.markdown("**⚠️ Missing Information:**")
                for info in missing_info:
                    st.markdown(f"- {info}")


# ============================================================================
# PAGE SECTIONS
# ============================================================================

class SettingsSection:
    """Handles the settings UI section."""
    
    @staticmethod
    def render() -> None:
        """Render the settings section."""
        with st.expander("⚙️ Settings", expanded=False):
            base_url = st.text_input(
                "API Base URL",
                value=st.session_state[SessionKeys.API_BASE_URL],
                help="The base URL for the backend API",
                placeholder=DEFAULT_API_URL
            )
            
            selected_rag = st.selectbox(
                "RAG Workflow",
                options=RAG_OPTIONS,
                index=RAG_OPTIONS.index(
                    st.session_state.get(SessionKeys.SELECTED_RAG, DEFAULT_RAG_TYPE)
                ),
                help="Select type of RAG to use"
            )
            
            if st.button("Save Settings", use_container_width=True):
                SettingsSection._save_settings(base_url, selected_rag)
    
    @staticmethod
    def _save_settings(base_url: str, rag_type: str) -> None:
        """Save settings to session state."""
        st.session_state[SessionKeys.API_BASE_URL] = base_url
        st.session_state[SessionKeys.SELECTED_RAG] = rag_type
        
        if base_url:
            os.environ["BASE_URL"] = base_url
        
        st.success("Settings saved successfully!")
        time.sleep(1)
        st.rerun()


class DocumentUploadSection:
    """Handles document upload UI section."""
    
    def __init__(self, api_service: APIService):
        self.api_service = api_service
    
    def render(self) -> None:
        """Render the document upload section."""
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(SUPPORTED_FILE_TYPES).upper()}",
            key=st.session_state[SessionKeys.FILE_UPLOADER_KEY]
        )
        
        if uploaded_files and st.button("Upload", use_container_width=True, type="primary"):
            self._process_uploads(uploaded_files)
    
    def _process_uploads(self, files) -> None:
        """Process uploaded files."""
        progress_bar = st.progress(0)
        successful_uploads = 0
        
        for idx, file in enumerate(files):
            with st.spinner(f"Processing {file.name}..."):
                if self.api_service.upload_document(file):
                    successful_uploads += 1
                    UIComponents.show_temporary_message(
                        st.success,
                        f"{file.name} uploaded successfully!",
                        duration=2
                    )
                else:
                    st.error(f"Failed to upload: {file.name}")
            
            progress_bar.progress((idx + 1) / len(files))
        
        if successful_uploads > 0:
            UIComponents.show_temporary_message(
                st.success,
                f"Uploaded {successful_uploads} file(s)",
                duration=3
            )
            SessionStateManager.increment_uploader_key()
            st.rerun()


class ChatInterface:
    """Handles the chat interface."""
    
    def __init__(self, api_service: APIService):
        self.api_service = api_service
    
    def render(self) -> None:
        """Render the chat interface."""
        self._render_chat_history()
        self._process_pending_query()
        self._render_chat_input()
    
    def _render_chat_history(self) -> None:
        """Display all messages in chat history."""
        for message in st.session_state[SessionKeys.CHAT_HISTORY]:
            role = message.get("role")
            content = message.get("content", "")
            
            with st.chat_message(role):
                st.markdown(content)
                
                if role == "assistant":
                    metadata = message.get("metadata", {})
                    if metadata:
                        UIComponents.render_answer_metadata(metadata)
    
    def _process_pending_query(self) -> None:
        """Process a query if one is pending."""
        if not st.session_state[SessionKeys.IS_PROCESSING]:
            return
        
        chat_history = st.session_state[SessionKeys.CHAT_HISTORY]
        if not chat_history or chat_history[-1].get("role") != "user":
            return
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                self._handle_query_response(chat_history[-1]["content"])
    
    def _handle_query_response(self, user_message: str) -> None:
        """Handle the API response for a query."""
        rag_type = SessionStateManager.get_selected_rag()
        response = self.api_service.send_chat_query(user_message, rag_type)
        
        if response:
            answer_text = response.get("answer", "")
            metadata = {
                "confidence": response.get("confidence", ""),
                "suggestions": response.get("suggestions", []),
                "missing_info": response.get("missing_info", [])
            }
            
            SessionStateManager.add_assistant_message(answer_text, metadata)
            SessionStateManager.set_processing(False)
            st.rerun()
        else:
            st.error("No response from server. Please check your connection.")
            SessionStateManager.set_processing(False)
    
    def _render_chat_input(self) -> None:
        """Render the chat input field."""
        user_query = st.chat_input(
            "Ask a question about your documents...",
            disabled=st.session_state[SessionKeys.IS_PROCESSING]
        )
        
        if user_query and not st.session_state[SessionKeys.IS_PROCESSING]:
            SessionStateManager.add_user_message(user_query)
            SessionStateManager.set_processing(True)
            st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class KnowledgeBaseApp:
    """Main application controller."""
    
    def __init__(self):
        SessionStateManager.initialize()
        self.api_service = APIService(SessionStateManager.get_api_url())
    
    def run(self) -> None:
        """Run the application."""
        self._configure_page()
        self._render_sidebar()
        self._render_main_content()
    
    def _configure_page(self) -> None:
        """Configure the page settings."""
        st.set_page_config(
            page_title="AI-Powered Knowledge Base",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("AI-Powered Knowledge Base")
        st.divider()
    
    def _render_sidebar(self) -> None:
        """Render the sidebar content."""
        with st.sidebar:
            SettingsSection.render()
            st.divider()
            
            DocumentUploadSection(self.api_service).render()
            st.divider()
            
            self._render_document_list()
    
    def _render_document_list(self) -> None:
        """Render the list of documents."""
        try:
            documents = self.api_service.fetch_documents()
            UIComponents.render_document_list(documents)
        except Exception as e:
            st.warning("Could not load documents")
            print(f"Error loading documents: {e}")
    
    def _render_main_content(self) -> None:
        """Render the main chat interface."""
        ChatInterface(self.api_service).render()


def main() -> None:
    """Application entry point."""
    app = KnowledgeBaseApp()
    app.run()


if __name__ == "__main__":
    main()