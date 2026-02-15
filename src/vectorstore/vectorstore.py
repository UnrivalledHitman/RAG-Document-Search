"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
):
    """Load embedding model, trying local first then downloading if needed"""
    model_kwargs = {"device": device}

    try:
        print(f"Searching for {model_name} locally...")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={**model_kwargs, "local_files_only": True},
        )
        print("Model loaded from local storage.")
        return embeddings
    except Exception as e:
        print("Model not found locally. Downloading from Hugging Face...")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs
        )
        print("Download complete and model initialized!")
        return embeddings


class VectorStore:
    """Manages vector store operations"""

    def __init__(self, embeddings=None):
        """Initialize vector store with embeddings"""
        self.embedding = embeddings if embeddings else load_embeddings()
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args:
            documents: List of documents to embed
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """
        Get the retriever instance

        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError(
                "Vector store not initialized. Call create_vectorstore first."
            )
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError(
                "Vector store not initialized. Call create_vectorstore first."
            )
        return self.retriever.invoke(query)
