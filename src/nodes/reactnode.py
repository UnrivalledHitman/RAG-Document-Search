"""LangGraph nodes for RAG workflow with OpenAI-format tool calling"""

from typing import List
from src.states.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from groq import BadRequestError
import wikipedia


class RAGNodes:
    """RAG nodes with tool calling support"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._llm_with_tools = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve documents from vector store"""
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    @tool
    def search_documents(self, query: str) -> str:
        """Search the document corpus for AI/ML information."""
        docs: List[Document] = self.retriever.invoke(query)
        if not docs:
            return "No documents found."
        results = []
        for i, d in enumerate(docs[:5], start=1):
            meta = d.metadata if hasattr(d, "metadata") else {}
            title = meta.get("title") or meta.get("source") or f"Doc{i}"
            results.append(f"[{i}] {title}\n{d.page_content[:500]}")
        return "\n\n".join(results)

    @tool
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for general knowledge."""
        try:
            wikipedia.set_lang("en")
            return wikipedia.summary(query, sentences=3, auto_suggest=False)
        except:
            return f"No Wikipedia result for '{query}'"

    def _get_tools(self):
        """Get tools bound to LLM"""
        if self._llm_with_tools is None:
            tools = [self.search_documents, self.search_wikipedia]
            self._llm_with_tools = self.llm.bind_tools(tools)
        return self._llm_with_tools

    def _call_tool(self, tool_call):
        """Execute a tool call"""
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        query = args.get("query", "")

        if name == "search_documents":
            return self.search_documents.invoke({"query": query})
        elif name == "search_wikipedia":
            return self.search_wikipedia.invoke({"query": query})
        return f"Unknown tool: {name}"

    def _agent_with_tools(self, question: str) -> str:
        """Run agent loop with tools"""
        llm = self._get_tools()
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Use search_documents for doc questions, search_wikipedia for general knowledge."
            ),
            HumanMessage(content=question),
        ]

        for _ in range(3):  # max 3 iterations
            response = llm.invoke(messages)
            messages.append(response)

            # Check for tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self._call_tool(tool_call)
                    messages.append(
                        ToolMessage(
                            content=result, tool_call_id=tool_call.get("id", "")
                        )
                    )
                continue  # Get next response after tool use

            return response.content

        return messages[-1].content if messages else "No answer generated"

    def _direct_generation(self, state: RAGState) -> str:
        """Fallback without tools"""
        if state.retrieved_docs:
            context = "\n\n".join(
                [
                    f"[{i+1}] {doc.page_content[:700]}"
                    for i, doc in enumerate(state.retrieved_docs[:5])
                ]
            )
        else:
            context = "No documents retrieved."

        prompt = f"Context:\n{context}\n\nQuestion: {state.question}\n\nAnswer based on context:"
        response = self.llm.invoke(prompt)
        return response.content

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer with tools or fallback"""
        try:
            answer = self._agent_with_tools(state.question)
        except (BadRequestError, Exception):
            answer = self._direct_generation(state)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer.",
        )
