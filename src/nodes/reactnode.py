"""LangGraph nodes for RAG workflow + ReAct Agent"""

import uuid
import sys
import os
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

# Tavily Search
from langchain_community.tools.tavily_search import TavilySearchResults


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
        """Build retriever + Tavily search tools"""

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
            description="Fetch passages from the indexed document corpus. Use this FIRST for information from uploaded documents.",
            func=retriever_tool_fn,
        )

        # Tavily Search Tool
        tools = [retriever_tool]

        try:
            # Check if API key is set
            if not os.getenv("TAVILY_API_KEY"):
                print("⚠️ TAVILY_API_KEY not set. Web search will not be available.")
                return tools

            # Initialize Tavily
            tavily_search = TavilySearchResults(
                max_results=3,
                search_depth="basic",  # or "advanced" for better results
                include_answer=True,
                include_raw_content=False,
            )

            def tavily_tool_fn(query: str) -> str:
                """Search the web using Tavily"""
                print(f"🌐 Tavily search: {query[:50]}...")
                try:
                    results = tavily_search.invoke(query)

                    if not results:
                        return "No search results found."

                    # Format results
                    formatted = []
                    for i, result in enumerate(results, 1):
                        title = result.get("title", "No title")
                        content = result.get("content", "")
                        url = result.get("url", "")

                        formatted.append(f"[{i}] {title}\n{content}\nSource: {url}")

                    output = "\n\n".join(formatted)
                    print(f"✅ Tavily returned {len(results)} results")
                    return output

                except Exception as e:
                    print(f"❌ Tavily search failed: {e}")
                    return f"Search failed: {str(e)}"

            tavily_tool = Tool(
                name="web_search",
                description="Search the web for current information, news, facts, or general knowledge not available in the documents. Use this for recent events or information beyond the document corpus.",
                func=tavily_tool_fn,
            )

            tools.append(tavily_tool)
            print("✅ Tavily search tool enabled")

        except Exception as e:
            print(f"⚠️ Tavily tool initialization failed: {e}. Using only retriever.")

        return tools

    def _build_agent(self) -> Runnable:
        """Create ReAct agent with tools"""
        print("🔧 Building ReAct agent...")
        tools = self._build_tools()

        system_prompt = (
            "You are a helpful RAG assistant with access to document retrieval and web search. "
            "ALWAYS use 'retriever' FIRST for questions about the uploaded documents. "
            "Use 'web_search' (Tavily) only for current events, recent news, or general knowledge not in the documents. "
            "Provide clear, accurate, and well-sourced answers. "
            "Cite your sources when using web search results."
        )

        agent = create_agent(self.llm, tools=tools, system_prompt=system_prompt)
        print(f"✅ ReAct agent built successfully with {len(tools)} tools")
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
            actual_sources = []

            # Extract answer and tool calls
            for msg in messages:
                # Check if this is a tool call message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        print(f"🔧 Tool used: {tool_name}")

                # Check if this is a tool result message
                if hasattr(msg, "name"):
                    tool_name = getattr(msg, "name", "")
                    content = getattr(msg, "content", "")

                    # Create a document-like object for tool results
                    if content and len(content) > 0:
                        doc = Document(
                            page_content=content[:500],  # Truncate long results
                            metadata={
                                "source": f"{tool_name} tool",
                                "type": "tool_result",
                            },
                        )
                        actual_sources.append(doc)

            # Get final answer
            if messages:
                answer_msg = messages[-1]
                answer = getattr(answer_msg, "content", None)

            final_answer = answer or "Could not generate answer."
            print(f"✅ Answer generated ({len(final_answer)} chars)")

            # Use actual sources if available, otherwise fall back to retrieved_docs
            sources_to_return = (
                actual_sources if actual_sources else state.retrieved_docs
            )

            return RAGState(
                question=state.question,
                retrieved_docs=sources_to_return,
                answer=final_answer,
            )

        except Exception as e:
            print(f"❌ Agent error: {e}")
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=f"Agent error: {str(e)}",
            )
