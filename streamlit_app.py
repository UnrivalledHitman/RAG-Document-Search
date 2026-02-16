"""
Streamlit UI for Agentic RAG System - With Toggle Source Selection
"""

import streamlit as st
from pathlib import Path
import sys
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))

try:
    from src.config.config import Config
    from src.document_ingestion.document_processor import DocumentProcessor
    from src.vectorstore.vectorstore import VectorStore
    from src.graph_builder.graph_builder import GraphBuilder

    logger.info("✅ All imports successful")
except Exception as e:
    logger.error(f"❌ Import failed: {e}")
    st.error(f"Import failed: {e}")
    st.stop()


# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "is_ready" not in st.session_state:
        st.session_state.is_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "source_choice" not in st.session_state:
        st.session_state.source_choice = "local"  # Default
    if "initialized_with" not in st.session_state:
        st.session_state.initialized_with = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        st.session_state.initialized_with = None


@st.cache_resource
def load_rag_system(source_choice: str, _uploaded_file_paths: Optional[list] = None):
    """Load RAG system with caching"""
    try:
        logger.info(f"Loading RAG system with sources: {source_choice}")

        # Get sources based on choice
        sources = Config.get_sources(source_choice)

        # Add uploaded files to sources
        if _uploaded_file_paths:
            sources.extend(_uploaded_file_paths)
            logger.info(f"Added {len(_uploaded_file_paths)} uploaded files")

        # Step 1: Initialize LLM
        with st.spinner("⏳ Step 1/4: Initializing LLM..."):
            llm = Config.get_llm()
            logger.info("LLM initialized")

        # Step 2: Process documents
        with st.spinner("⏳ Step 2/4: Processing documents..."):
            doc_processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            )
            documents = doc_processor.process_sources(sources)
            logger.info(f"Processed {len(documents)} document chunks")

        # Step 3: Build vector store
        with st.spinner("⏳ Step 3/4: Building vector store..."):
            vector_store = VectorStore()
            vector_store.create_vectorstore(documents)
            logger.info("Vector store created")

        # Step 4: Build graph
        with st.spinner("⏳ Step 4/4: Building agent graph..."):
            graph_builder = GraphBuilder(
                retriever=vector_store.get_retriever(), llm=llm
            )
            graph_builder.build()
            logger.info("Graph built")

        return graph_builder, len(documents), sources

    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0, []


def display_sidebar():
    """Display sidebar with source selection"""
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("---")

        # Source selection with radio buttons
        st.subheader("📚 Document Sources")

        source_choice = st.radio(
            "Select your document sources:",
            options=["local", "web", "both"],
            format_func=lambda x: {
                "local": "📁 Local PDFs Only",
                "web": "🌐 Web URLs Only",
                "both": "📚 Both Local & Web",
            }[x],
            index=["local", "web", "both"].index(st.session_state.source_choice),
            help="Choose which sources to use for answering questions",
        )

        # Update session state
        if source_choice != st.session_state.source_choice:
            st.session_state.source_choice = source_choice
            st.session_state.is_ready = False
            st.session_state.initialized_with = None
            # Clear cache to reload with new sources
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")

        # File uploader
        st.subheader("📎 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents to add to the knowledge base",
        )

        # Process uploaded files
        if uploaded_files:
            import tempfile

            temp_paths = []

            for uploaded_file in uploaded_files:
                # Save to temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)

            # Update session state
            if temp_paths != st.session_state.uploaded_files:
                st.session_state.uploaded_files = temp_paths
                st.session_state.is_ready = False
                st.session_state.initialized_with = None
                st.cache_resource.clear()

            st.success(f"📄 {len(uploaded_files)} file(s) uploaded")
        else:
            if st.session_state.uploaded_files:
                # Clear uploaded files
                st.session_state.uploaded_files = []
                st.session_state.is_ready = False
                st.session_state.initialized_with = None
                st.cache_resource.clear()

        st.markdown("---")

        # Initialize button
        if (
            not st.session_state.is_ready
            or st.session_state.initialized_with != source_choice
        ):
            if st.button(
                "🚀 Initialize System", type="primary", use_container_width=True
            ):
                with st.spinner("Initializing RAG system..."):
                    rag_system, doc_count, sources = load_rag_system(
                        source_choice, st.session_state.uploaded_files
                    )

                    if rag_system:
                        st.session_state.rag_system = rag_system
                        st.session_state.is_ready = True
                        st.session_state.initialized_with = source_choice
                        st.success(f"✅ System ready! Loaded {doc_count} chunks")
                        st.balloons()
                        st.rerun()
        else:
            st.success(f"✅ System Ready")
            st.info(f"Using: **{st.session_state.source_choice}** sources")

            # Reset button
            if st.button("🔄 Reinitialize", use_container_width=True):
                st.session_state.is_ready = False
                st.session_state.initialized_with = None
                st.cache_resource.clear()
                st.rerun()

        st.markdown("---")

        # System info
        st.subheader("ℹ️ System Info")
        if st.session_state.is_ready:
            st.write("**Status:** 🟢 Ready")
            st.write(f"**Sources:** {st.session_state.source_choice}")
            if st.session_state.uploaded_files:
                st.write(
                    f"**Uploaded:** {len(st.session_state.uploaded_files)} file(s)"
                )
        else:
            st.write("**Status:** 🔴 Not initialized")

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def display_chat_interface():
    """Display main chat interface"""
    st.title("🤖 RAG Document Assistant")

    # Display welcome message if not initialized
    if not st.session_state.is_ready:
        st.info(
            "👈 **Please select your document sources in the sidebar and click 'Initialize System' to begin!**"
        )
        st.markdown("---")
        st.markdown(
            """
            ### Features:
            - 📁 **Local PDFs** - Query documents from your data folder
            - 🌐 **Web Sources** - Query online articles and blogs
            - 🔍 **Hybrid Search** - Combines document retrieval with web search
            - 🤖 **AI Agent** - Intelligent tool selection for better answers
            
            ### How to Use:
            1. Select document sources in the sidebar
            2. Click "Initialize System"
            3. Start asking questions!
            """
        )
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** `{source['source']}`")
                        st.markdown(f"_{source['content']}_")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Query RAG system
                    result = st.session_state.rag_system.run(prompt)

                    # Get answer
                    answer = result.get("answer", "No answer generated")

                    # Display answer
                    st.markdown(answer)

                    # Prepare sources for display
                    sources = []
                    retrieved_docs = result.get("retrieved_docs", [])
                    if retrieved_docs:
                        for doc in retrieved_docs[:3]:
                            metadata = getattr(doc, "metadata", {}) or {}
                            source = metadata.get("source", "Unknown")
                            content = doc.page_content[:200].replace("\n", " ")
                            if len(doc.page_content) > 200:
                                content += "..."
                            sources.append({"source": source, "content": content})

                        # Display sources in expander
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** `{source['source']}`")
                                st.markdown(f"_{source['content']}_")
                                st.markdown("---")

                    # Add assistant message with sources
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


def main():
    """Main application"""
    # Initialize session state
    init_session_state()

    # Display sidebar
    display_sidebar()

    # Display chat interface
    display_chat_interface()


if __name__ == "__main__":
    main()
