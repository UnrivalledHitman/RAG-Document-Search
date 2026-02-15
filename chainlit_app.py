"""
Chainlit UI for Agentic RAG System - Fixed Version
"""

import chainlit as cl
from pathlib import Path
import sys
import asyncio
from typing import Optional
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


@cl.on_chat_start
async def on_chat_start():
    """Initialize when chat starts"""
    global is_ready, rag_system

    logger.info("Chat started")

    try:
        # Send initial message
        msg = cl.Message(content="🚀 **RAG System Starting...**")
        await msg.send()

        # Start initialization in background
        asyncio.create_task(load_rag_async())

    except Exception as e:
        logger.error(f"Error in on_chat_start: {e}")
        error_msg = cl.Message(content=f"❌ Startup error: {str(e)}")
        await error_msg.send()


async def load_rag_async():
    """Load RAG system in background"""
    global rag_system, is_ready

    try:
        logger.info("Starting RAG initialization")

        # Step 1
        step1_msg = cl.Message(content="⏳ **Step 1/4:** Initializing LLM...")
        await step1_msg.send()
        llm = await asyncio.to_thread(Config.get_llm)
        logger.info("LLM initialized")

        # Step 2
        step2_msg = cl.Message(content="⏳ **Step 2/4:** Processing documents...")
        await step2_msg.send()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        documents = await asyncio.to_thread(
            doc_processor.process_sources, Config.DEFAULT_URLS
        )
        logger.info(f"Processed {len(documents)} documents")

        # Step 3
        step3_msg = cl.Message(content="⏳ **Step 3/4:** Building vector store...")
        await step3_msg.send()
        vector_store = VectorStore()
        await asyncio.to_thread(vector_store.create_vectorstore, documents)
        logger.info("Vector store created")

        # Step 4
        step4_msg = cl.Message(content="⏳ **Step 4/4:** Building graph...")
        await step4_msg.send()
        graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
        await asyncio.to_thread(graph_builder.build)
        logger.info("Graph built")

        rag_system = graph_builder
        is_ready = True

        ready_msg = cl.Message(
            content=f"✅ **System Ready!**\n\nLoaded {len(documents)} chunks. Ask your question!"
        )
        await ready_msg.send()

    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        error_msg = cl.Message(content=f"❌ Error:\n```\n{str(e)}\n```")
        await error_msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user questions"""
    logger.info(f"Received message: {message.content[:50]}")

    if not is_ready or rag_system is None:
        waiting_msg = cl.Message(content="⏳ System still loading... please wait.")
        await waiting_msg.send()
        return

    question = message.content.strip()
    if not question:
        return

    # Initialize msg variable
    msg = cl.Message(content="🤔 Processing your question...")

    try:
        # Show processing
        await msg.send()

        # Query RAG
        result = await asyncio.to_thread(rag_system.run, question)
        logger.info("RAG query completed")

        # Update with answer
        msg.content = f"### 💡 Answer\n\n{result['answer']}"
        await msg.update()

        # Send sources
        if result.get("retrieved_docs"):
            sources = "\n\n".join(
                [
                    f"**Source {i+1}:**\n{doc.page_content[:200]}..."
                    for i, doc in enumerate(result["retrieved_docs"][:3])
                ]
            )
            sources_msg = cl.Message(content=f"### 📄 Sources\n\n{sources}")
            await sources_msg.send()

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()
