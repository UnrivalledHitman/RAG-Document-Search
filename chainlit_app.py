"""
Chainlit UI for Agentic RAG System - With Runtime Source Selection
"""

import chainlit as cl
from pathlib import Path
import sys
import asyncio
from typing import Optional, List
import logging

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
    raise

# Global state
rag_system: Optional[GraphBuilder] = None
is_ready = False
current_sources: List[str] = []


@cl.on_chat_start
async def on_chat_start():
    """Initialize when chat starts with source selection"""
    global is_ready, rag_system, current_sources

    logger.info("Chat started")

    try:
        # Send welcome message with instructions
        welcome_msg = cl.Message(
            content="# 🤖 Welcome to RAG Document Assistant!\n\n"
            "Please select which document sources you'd like to use:\n\n"
            "Reply with:\n"
            "- **1** for 📁 Local PDFs Only\n"
            "- **2** for 🌐 Web URLs Only\n"
            "- **3** for 📚 Both Local & Web\n\n"
            "_(Or just type your question and I'll use all sources)_"
        )
        await welcome_msg.send()

        # Set a flag to wait for source selection
        cl.user_session.set("waiting_for_source_selection", True)

    except Exception as e:
        logger.error(f"Error in on_chat_start: {e}", exc_info=True)
        error_msg = cl.Message(content=f"❌ Startup error: {str(e)}")
        await error_msg.send()


async def load_rag_async(source_choice: str = "both"):
    """Load RAG system in background with selected sources"""
    global rag_system, is_ready, current_sources

    try:
        logger.info(f"Starting RAG initialization with sources: {source_choice}")

        # Get sources based on user choice
        sources = Config.get_sources(source_choice)
        current_sources = sources

        # Show what's being loaded
        source_list = "\n".join([f"  • {src}" for src in sources])
        sources_msg = cl.Message(content=f"**Loading from:**\n{source_list}\n")
        await sources_msg.send()

        # Step 1: Initialize LLM
        step1_msg = cl.Message(content="⏳ **Step 1/4:** Initializing LLM...")
        await step1_msg.send()
        llm = await asyncio.to_thread(Config.get_llm)
        logger.info("LLM initialized")
        await asyncio.sleep(0.5)

        # Step 2: Process documents
        step2_msg = cl.Message(content="⏳ **Step 2/4:** Processing documents...")
        await step2_msg.send()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        documents = await asyncio.to_thread(doc_processor.process_sources, sources)
        logger.info(f"Processed {len(documents)} document chunks")
        await asyncio.sleep(0.5)

        # Step 3: Build vector store
        step3_msg = cl.Message(content="⏳ **Step 3/4:** Building vector store...")
        await step3_msg.send()
        vector_store = VectorStore()
        await asyncio.to_thread(vector_store.create_vectorstore, documents)
        logger.info("Vector store created")
        await asyncio.sleep(0.5)

        # Step 4: Build graph
        step4_msg = cl.Message(content="⏳ **Step 4/4:** Building agent graph...")
        await step4_msg.send()
        graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
        await asyncio.to_thread(graph_builder.build)
        logger.info("Graph built")

        rag_system = graph_builder
        is_ready = True

        # Success message
        ready_msg = cl.Message(
            content=f"✅ **System Ready!**\n\n"
            f"📊 Loaded **{len(documents)} document chunks**\n"
            f"🔧 Using **{source_choice}** sources\n\n"
            f"💡 Ask me anything about your documents!"
        )
        await ready_msg.send()

    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        error_msg = cl.Message(
            content=f"❌ **Initialization Failed**\n\n"
            f"```\n{str(e)}\n```\n\n"
            f"Please check:\n"
            f"• Your data folder contains PDFs (if using local)\n"
            f"• URLs are accessible (if using web)\n"
            f"• API keys are configured"
        )
        await error_msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user questions"""
    logger.info(f"Received message: {message.content[:50]}")

    # Check if waiting for source selection
    waiting_for_selection = cl.user_session.get("waiting_for_source_selection", False)

    if waiting_for_selection:
        user_input = message.content.strip()

        # Map user input to source choice
        source_map = {
            "1": ("local", "📁 Local PDFs Only"),
            "2": ("web", "🌐 Web URLs Only"),
            "3": ("both", "📚 Both Local & Web"),
        }

        if user_input in source_map:
            source_choice, source_label = source_map[user_input]

            # Acknowledge choice
            choice_msg = cl.Message(
                content=f"✅ Selected: **{source_label}**\n\n"
                f"🚀 Initializing RAG System..."
            )
            await choice_msg.send()

            # Clear the flag
            cl.user_session.set("waiting_for_source_selection", False)

            # Start initialization
            asyncio.create_task(load_rag_async(source_choice))
            return
        else:
            # User didn't select 1, 2, or 3 - use default and treat as question
            cl.user_session.set("waiting_for_source_selection", False)

            # Start with default (both)
            default_msg = cl.Message(
                content="📚 Using **all sources** (default)\n\n🚀 Initializing..."
            )
            await default_msg.send()

            asyncio.create_task(load_rag_async("both"))

            # Wait a bit for initialization before processing the question
            await asyncio.sleep(2)

    # Handle normal questions
    if not is_ready or rag_system is None:
        waiting_msg = cl.Message(
            content="⏳ System still loading... please wait a moment."
        )
        await waiting_msg.send()
        return

    question = message.content.strip()
    if not question:
        return

    # Show processing indicator
    msg = cl.Message(content="🤔 Processing your question...")

    try:
        await msg.send()

        # Query RAG system
        result = await asyncio.to_thread(rag_system.run, question)
        logger.info("RAG query completed")

        # Update with answer
        answer_content = result.get("answer", "No answer generated")
        msg.content = f"### 💡 Answer\n\n{answer_content}"
        await msg.update()

        # Send sources if available
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            sources_content = "### 📄 Sources\n\n"

            for i, doc in enumerate(retrieved_docs[:3], 1):
                # Get metadata
                metadata = getattr(doc, "metadata", {}) or {}
                source = metadata.get("source", "Unknown")

                # Truncate content
                content = doc.page_content[:200].replace("\n", " ")
                if len(doc.page_content) > 200:
                    content += "..."

                sources_content += f"**Source {i}:** `{source}`\n{content}\n\n"

            sources_msg = cl.Message(content=sources_content)
            await sources_msg.send()

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        msg.content = f"❌ **Error processing query**\n\n```\n{str(e)}\n```"
        await msg.update()


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat ends"""
    global is_ready, rag_system, current_sources
    logger.info("Chat ended")
    is_ready = False
    rag_system = None
    current_sources = []
