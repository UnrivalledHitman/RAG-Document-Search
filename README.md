# 🤖 RAG Document Assistant

An intelligent document question-answering system powered by **LangChain**, **LangGraph**, and **Streamlit**. Features a ReAct agent that intelligently combines local document retrieval with real-time web search for comprehensive answers.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://rag-document-search-d2ye3wnwzq7szulcthewuu.streamlit.app/)

## 🧠 Technical Details

### Vector Search

**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2

- Dimension: 384
- Speed: ~14,000 sentences/sec
- Quality: High for general-purpose tasks

**Vector Store**: FAISS (Facebook AI Similarity Search)

- Index type: Flat (brute-force, exact search)
- Distance metric: Cosine similarity
- Retrieval: Top-k documents (k=4 default)

#### Web Search Tool (Tavily)

### LLM Configuration

**Provider**: Groq
**Model**: openai/gpt-oss-20b (Llama-based, 20B parameters)