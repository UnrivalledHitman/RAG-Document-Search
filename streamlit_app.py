"""
Streamlit UI for Agentic RAG System - With Toggle Source Selection
(Updated: Uploaded files are saved directly into /data folder)
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
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "is_ready" not in st.session_state:
        st.session_state.is_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "source_choice" not in st.session_state:
        st.session_state.source_choice = "local"
    if "initialized_with" not in st.session_state:
        st.session_state.initialized_with = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


@st.cache_resource
def load_rag_system(source_choice: str):
    """Load RAG system with caching"""
    try:
        logger.info(f"Loading RAG system with sources: {source_choice}")

        # Get sources based on choice
        sources = Config.get_sources(source_choice)

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

            if not documents:
                raise ValueError("No documents found in data folder.")

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
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("---")

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
        )

        if source_choice != st.session_state.source_choice:
            st.session_state.source_choice = source_choice
            st.session_state.is_ready = False
            st.session_state.initialized_with = None
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")

        # Upload PDFs → Save directly into /data folder
        st.subheader("📎 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            saved_paths = []

            for uploaded_file in uploaded_files:
                save_path = data_dir / uploaded_file.name

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                saved_paths.append(str(save_path))

            if saved_paths != st.session_state.uploaded_files:
                st.session_state.uploaded_files = saved_paths
                st.session_state.is_ready = False
                st.session_state.initialized_with = None
                st.cache_resource.clear()

            st.success(f"📄 {len(uploaded_files)} file(s) saved to data folder")

        if st.session_state.uploaded_files:
            st.info(f"✅ {len(st.session_state.uploaded_files)} file(s) ready in data/")

        st.markdown("---")

        if (
            not st.session_state.is_ready
            or st.session_state.initialized_with != source_choice
        ):
            if st.button(
                "🚀 Initialize System", type="primary", use_container_width=True
            ):
                with st.spinner("Initializing RAG system..."):
                    rag_system, doc_count, sources = load_rag_system(source_choice)

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

            if st.button("🔄 Reinitialize", use_container_width=True):
                st.session_state.is_ready = False
                st.session_state.initialized_with = None
                st.cache_resource.clear()
                st.rerun()

        st.markdown("---")

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

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def display_chat_interface():
    st.title("🤖 RAG Document Assistant")

    if not st.session_state.is_ready:
        st.info(
            "👈 **Please select your document sources in the sidebar and click 'Initialize System' to begin!**"
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"{i}. {source['source']}")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_system.run(prompt)
                    answer = result.get("answer", "No answer generated")
                    st.markdown(answer)

                    sources = []
                    retrieved_docs = result.get("retrieved_docs", [])

                    if retrieved_docs:
                        for doc in retrieved_docs[:3]:
                            metadata = getattr(doc, "metadata", {}) or {}
                            source = metadata.get("source", "Unknown")
                            doc_type = metadata.get("type", "")

                            if doc_type == "tool_result":
                                continue

                            sources.append({"source": source})

                        if sources:
                            st.markdown("**Sources:**")
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"{i}. {source['source']}")

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
    init_session_state()
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()
