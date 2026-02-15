"""LangGraph nodes for RAG workflow + ReAct Agent"""

import uuid
import sys
from typing import List, Optional

# Inject uuid into globals for type hint evaluation
if "uuid" not in globals():
    globals()["uuid"] = uuid
sys.modules.setdefault("uuid", uuid)

from src.states.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.agent: Runnable = self._build_agent()

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retriever node"""
        print(f"🔍 Retrieving documents for: {state.question[:50]}...")
        docs = self.retriever.invoke(state.question)
        print(f"✅ Retrieved {len(docs)} documents")
        return RAGState(question=state.question, retrieved_docs=docs)

    def _build_tools(self) -> List[Tool]:
        """Build retriever tool"""

        def retriever_tool_fn(query: str) -> str:
            """Retrieve relevant documents from the indexed corpus"""
            print(f"🔧 Tool called: retriever with query: {query[:50]}...")
            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No documents found."

            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {}) or {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")

            result = "\n\n".join(merged)
            print(f"✅ Tool returned {len(docs)} documents")
            return result

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from the indexed document corpus. Use this to find relevant information from the user's uploaded documents.",
            func=retriever_tool_fn,
        )

        return [retriever_tool]

    def _build_agent(self) -> Runnable:
        """Create ReAct agent with tools"""
        print("🔧 Building ReAct agent...")
        tools = self._build_tools()

        system_prompt = (
            "You are a helpful RAG assistant. "
            "Use the 'retriever' tool to search through the indexed documents. "
            "Always check the retriever before answering questions about the documents. "
            "Provide clear, concise answers based on the retrieved information. "
            "If you cannot find relevant information, say so honestly."
        )

        agent = create_agent(self.llm, tools=tools, system_prompt=system_prompt)
        print("✅ ReAct agent built successfully")
        return agent

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using ReAct agent"""
        print(f"🤖 Generating answer for: {state.question[:50]}...")

        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=state.question)]}
            )

            messages = result.get("messages", [])
            answer: Optional[str] = None

            if messages:
                answer_msg = messages[-1]
                answer = getattr(answer_msg, "content", None)

            final_answer = answer or "Could not generate answer."
            print(f"✅ Answer generated ({len(final_answer)} chars)")

            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=final_answer,
            )

        except Exception as e:
            print(f"❌ Agent error: {e}")
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=f"Agent error: {str(e)}",
            )
